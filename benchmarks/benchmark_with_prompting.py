import json
import time
from typing import Tuple
import re
import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys

sys.path.insert(0, "..")
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

RETRIEVAL_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a fact retrieval system.

IMPORTANT RULES:
1. When answering factual questions, ALWAYS retrieve the fact first
2. Use the format: [RETRIEVE: what you want to retrieve]
3. The system will provide the fact, then you continue answering
4. Only answer based on retrieved facts, never guess

Example:
Q: What is the capital of France?
A: [RETRIEVE: capital of France] The capital of France is Paris, a beautiful city.

Now answer:"""


class RAGSystem:
    def __init__(self, model="gemma3:1b", facts=None):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = facts or []
        self._build_index()

    def _build_index(self):
        if not self.facts:
            return
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

    def generate(self, question: str) -> Tuple[str, float]:
        start = time.time()
        fact = self.retrieve(question)
        prompt = f"Context: {fact}\n\nQuestion: {question}\nAnswer:"

        response = ollama.generate(
            model=self.model, prompt=prompt, stream=False, options={"num_predict": 100}
        )

        return response["response"], time.time() - start


class SIMSystemOptimized:
    def __init__(self, model="gemma3:1b", facts=None):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = facts or []
        self.fusion = FusionModule()
        self._build_index()

    def _build_index(self):
        if not self.facts:
            return
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

    def generate(self, question: str) -> Tuple[str, float]:
        start = time.time()

        prompt = f"""{RETRIEVAL_SYSTEM_PROMPT}

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

        return output, time.time() - start


def benchmark():
    print("\n" + "=" * 70)
    print("Optimized SIM Benchmark (WITH EXPLICIT PROMPTING)")
    print("=" * 70)

    rag = RAGSystem(model="gemma3:1b", facts=FACTS)
    sim = SIMSystemOptimized(model="gemma3:1b", facts=FACTS)

    questions = [
        "What is the capital of France?",
        "Where is the Eiffel Tower?",
        "When did the French Revolution happen?",
        "What is the capital of Germany?",
        "Who won the World Cup in 1998?",
    ]

    results = {"RAG": [], "SIM": []}

    for system_name, system in [("RAG", rag), ("SIM", sim)]:
        print(f"\n[{system_name}]")

        for i, q in enumerate(questions, 1):
            answer, latency = system.generate(q)
            results[system_name].append(
                {"question": q, "answer": answer[:80], "latency": latency}
            )
            print(f"  {i}/5 OK")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    rag_latency = sum(r["latency"] for r in results["RAG"]) / len(results["RAG"])
    sim_latency = sum(r["latency"] for r in results["SIM"]) / len(results["SIM"])

    print(f"\nRAG avg latency: {rag_latency:.2f}s")
    print(f"SIM avg latency: {sim_latency:.2f}s")
    print(f"Difference: {sim_latency - rag_latency:+.2f}s")

    with open("benchmark_optimized.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to benchmark_optimized.json")


if __name__ == "__main__":
    benchmark()
