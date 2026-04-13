import json
import time
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import sys

sys.path.insert(0, "..")
from fusion_module_simple import FusionModule


class BaselineSystem:
    def __init__(self, model="gemma3:1b"):
        self.model = model

    def generate(self, question: str) -> Tuple[str, float]:
        start = time.time()
        prompt = f"Question: {question}\nAnswer:"

        response = ollama.generate(
            model=self.model, prompt=prompt, stream=False, options={"num_predict": 100}
        )

        latency = time.time() - start
        return response["response"], latency


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

        latency = time.time() - start
        return response["response"], latency


class SIMSystem:
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
        prompt = f"Question: {question}\n[RETRIEVE: {question}]\nAnswer:"

        response = ollama.generate(
            model=self.model, prompt=prompt, stream=False, options={"num_predict": 100}
        )

        output = response["response"]
        latency = time.time() - start

        return output, latency


class BenchmarkRunner:
    def __init__(self, model="gemma3:1b", num_samples=10):
        self.model = model
        self.num_samples = num_samples
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "num_samples": num_samples,
            "systems": {},
        }

        self.facts = [
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

    def run(self):
        print("\n" + "=" * 70)
        print("SIM vs RAG vs Baseline Benchmark")
        print("=" * 70)

        baseline = BaselineSystem(self.model)
        rag = RAGSystem(self.model, self.facts)
        sim = SIMSystem(self.model, self.facts)

        questions = [
            "What is the capital of France?",
            "Where is the Eiffel Tower?",
            "What is France known for?",
            "When did the French Revolution happen?",
            "What is the capital of Germany?",
            "What is the capital of Japan?",
            "Who won the World Cup in 1998?",
            "What is in the Louvre Museum?",
            "How many people speak French?",
            "Is France visited by many tourists?",
        ][: self.num_samples]

        for system_name, system in [("Baseline", baseline), ("RAG", rag), ("SIM", sim)]:
            print(f"\n[{system_name}] Running {len(questions)} questions...")

            system_results = {
                "answers": [],
                "avg_latency": 0.0,
                "total_latency": 0.0,
            }

            for i, question in enumerate(questions, 1):
                answer, latency = system.generate(question)
                system_results["answers"].append(
                    {"question": question, "answer": answer[:100], "latency": latency}
                )
                system_results["total_latency"] += latency
                print(f"  {i}/{len(questions)} OK", end="\r")

            system_results["avg_latency"] = system_results["total_latency"] / len(
                questions
            )
            self.results["systems"][system_name] = system_results
            print(f"  {system_name}: {system_results['avg_latency']:.2f}s avg latency")

        return self.results

    def save(self, filename="benchmark_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")
        return filename

    def print_summary(self):
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        for system_name, data in self.results["systems"].items():
            print(f"\n{system_name}:")
            print(f"  Average Latency: {data['avg_latency']:.2f}s")
            print(f"  Total Latency: {data['total_latency']:.2f}s")
            print(f"  Samples: {len(data['answers'])}")

        baseline_latency = self.results["systems"]["Baseline"]["avg_latency"]
        rag_latency = self.results["systems"]["RAG"]["avg_latency"]
        sim_latency = self.results["systems"]["SIM"]["avg_latency"]

        print("\n" + "-" * 70)
        print("LATENCY COMPARISON:")
        print(f"  Baseline:        {baseline_latency:.2f}s")
        print(
            f"  RAG:             {rag_latency:.2f}s ({(rag_latency / baseline_latency - 1) * 100:+.1f}%)"
        )
        print(
            f"  SIM:             {sim_latency:.2f}s ({(sim_latency / baseline_latency - 1) * 100:+.1f}%)"
        )


def main():
    runner = BenchmarkRunner(model="gemma3:1b", num_samples=10)
    results = runner.run()
    runner.print_summary()
    runner.save()


if __name__ == "__main__":
    main()
