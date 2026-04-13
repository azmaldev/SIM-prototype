import asyncio
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import time
from typing import Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_module_simple import FusionModule


FACTS = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "France is known for its world-renowned cuisine.",
    "The French Revolution began in 1789.",
    "France is a founding member of the European Union.",
    "The Louvre Museum in Paris houses the Mona Lisa.",
    "France has won the FIFA World Cup twice, in 1998 and 2018.",
    "The French language is spoken by over 300 million people worldwide.",
    "France is the most visited country in the world.",
    "The French flag consists of three vertical bands of blue, white, and red.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "London is the capital of the United Kingdom.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
]


class AsyncSIM:
    def __init__(self, model="gemma3:1b", facts=None):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = facts or FACTS
        self.fusion = FusionModule()
        self._build_index()

        self.metrics = {
            "generation_time": 0.0,
            "retrieval_time": 0.0,
            "total_time": 0.0,
        }

    def _build_index(self):
        embeddings = self.embed_model.encode(self.facts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str) -> str:
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), 1
        )
        return self.facts[indices[0][0]]

    def generate_with_parallel_retrieval(self, prompt: str) -> Tuple[str, Dict]:
        start_total = time.time()

        full_prompt = f"""You are a helpful AI that retrieves facts when needed.

IMPORTANT: When answering, use this format for facts:
[RETRIEVE: what you need] Then answer.

Question: {prompt}
Answer:"""

        start_gen = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=full_prompt,
            stream=False,
            options={"num_predict": 100},
        )
        gen_time = time.time() - start_gen

        output = response["response"]

        match = re.search(r"\[RETRIEVE:\s*([^\]]+)\]", output)

        if match:
            query = match.group(1).strip()

            start_ret = time.time()
            fact = self.retrieve(query)
            ret_time = time.time() - start_ret

            fusion_result = self.fusion.fuse(full_prompt, query, fact)
            new_prompt = fusion_result["fused_prompt"]
            injected_context = fusion_result.get("context")

            response = ollama.generate(
                model=self.model,
                prompt=f"Context: {injected_context}\n\n{new_prompt}"
                if injected_context
                else new_prompt,
                stream=False,
                options={"num_predict": 100},
            )

            output = response["response"]
        else:
            ret_time = 0.0

        total_time = time.time() - start_total

        self.metrics = {
            "generation_time": gen_time,
            "retrieval_time": ret_time,
            "total_time": total_time,
            "has_retrieval": match is not None,
        }

        return output, self.metrics


class OptimizedSIM:
    def __init__(self, model="gemma3:1b", facts=None):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = facts or FACTS
        self.fusion = FusionModule()
        self._build_index()

    def _build_index(self):
        embeddings = self.embed_model.encode(self.facts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query: str) -> str:
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), 1
        )
        return self.facts[indices[0][0]]

    def generate_optimized(self, question: str) -> Tuple[str, float]:
        start = time.time()

        prompt = f"""You must retrieve facts before answering.

Format: [RETRIEVE: fact needed] Then answer.

Question: {question}
Answer:"""

        response = ollama.generate(
            model=self.model, prompt=prompt, stream=False, options={"num_predict": 100}
        )

        output = response["response"]

        match = re.search(r"\[RETRIEVE:\s*([^\]]+)\]", output)
        if match:
            query = match.group(1).strip()
            fact = self.retrieve(query)
            fusion_result = self.fusion.fuse(prompt, query, fact)
            final_prompt = fusion_result["fused_prompt"]

            response = ollama.generate(
                model=self.model,
                prompt=final_prompt,
                stream=False,
                options={"num_predict": 100},
            )
            output = response["response"]

        return output, time.time() - start


def benchmark():
    print("\n" + "=" * 70)
    print("Latency Comparison: RAG vs Async SIM vs Optimized SIM")
    print("=" * 70)

    questions = [
        "What is the capital of France?",
        "Where is the Eiffel Tower?",
        "When did the French Revolution happen?",
    ]

    print("\n[RAG - Baseline]")
    rag_times = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        response = ollama.generate(
            model="gemma3:1b",
            prompt=f"Context: Paris is the capital of France\n\nQuestion: {q}\nAnswer:",
            stream=False,
            options={"num_predict": 100},
        )
        elapsed = time.time() - start
        rag_times.append(elapsed)
        print(f"  Q{i}: {elapsed:.2f}s")

    rag_avg = sum(rag_times) / len(rag_times)

    print("\n[Async SIM]")
    sim_async = AsyncSIM(model="gemma3:1b")
    sim_times = []
    for i, q in enumerate(questions, 1):
        output, metrics = sim_async.generate_with_parallel_retrieval(q)
        sim_times.append(metrics["total_time"])
        print(
            f"  Q{i}: {metrics['total_time']:.2f}s (gen: {metrics['generation_time']:.2f}s, ret: {metrics['retrieval_time']:.2f}s)"
        )

    sim_avg = sum(sim_times) / len(sim_times)

    print("\n[Optimized SIM]")
    sim_opt = OptimizedSIM(model="gemma3:1b")
    sim_opt_times = []
    for i, q in enumerate(questions, 1):
        output, elapsed = sim_opt.generate_optimized(q)
        sim_opt_times.append(elapsed)
        print(f"  Q{i}: {elapsed:.2f}s")

    sim_opt_avg = sum(sim_opt_times) / len(sim_opt_times)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nRAG:               {rag_avg:.2f}s (baseline)")
    print(f"Async SIM:         {sim_avg:.2f}s ({(sim_avg / rag_avg - 1) * 100:+.1f}%)")
    print(
        f"Optimized SIM:     {sim_opt_avg:.2f}s ({(sim_opt_avg / rag_avg - 1) * 100:+.1f}%)"
    )

    print(f"\nSIM Overhead: {sim_opt_avg - rag_avg:.2f}s per query")

    if sim_opt_avg < rag_avg * 1.2:
        print("\nSIM latency within acceptable range (<20% overhead)")
    else:
        print(
            f"\nSIM latency overhead too high ({(sim_opt_avg / rag_avg - 1) * 100:.1f}%)"
        )


if __name__ == "__main__":
    benchmark()
