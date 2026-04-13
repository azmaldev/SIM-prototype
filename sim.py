import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import sys
from fusion_module import FusionModule, AdvancedFusionModule

FACTS = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "France is known for its world-renowned cuisine, including croissants, baguettes, and fine wines.",
    "The French Revolution began in 1789 and led to significant political and social changes.",
    "France is a founding member of the European Union and the United Nations.",
    "The Louvre Museum in Paris houses the Mona Lisa and is the world's largest art museum.",
    "France has won the FIFA World Cup twice, in 1998 and 2018.",
    "The French language is spoken by over 300 million people worldwide.",
    "France is the most visited country in the world with over 89 million international tourists annually.",
    "The French flag consists of three vertical bands of blue, white, and red.",
]


class SIM:
    def __init__(self, model="gemma3:1b", use_advanced_fusion=False):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = FACTS

        if use_advanced_fusion:
            self.fusion = AdvancedFusionModule()
        else:
            self.fusion = FusionModule()

        self._build_index()

    def _build_index(self):
        embeddings = self.embed_model.encode(self.facts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, k=1):
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )
        return self.facts[indices[0][0]]

    def stream_generate(self, prompt, injected_context=None):
        full_prompt = prompt
        if injected_context:
            full_prompt = f"Context: {injected_context}\n\n{prompt}"

        response = ollama.generate(
            model=self.model,
            prompt=full_prompt,
            stream=True,
            options={"num_predict": 200},
        )
        return response

    def run(self, prompt, use_fusion=True):
        buffer = ""
        final_prompt = prompt

        print("Streaming output:\n")

        response = self.stream_generate(prompt)
        for chunk in response:
            token = chunk["response"]
            buffer += token

            sys.stdout.write(token)
            sys.stdout.flush()

            match = re.search(r"\[RETRIEVE:\s*([^\]]+)\]", buffer)
            if match:
                query = match.group(1).strip()
                print(f"\n\n[RETRIEVE: Detected query: '{query}']")

                fact = self.retrieve(query)
                print(f"[RETRIEVE: Retrieved fact: '{fact}']\n")

                if use_fusion:
                    fusion_result = self.fusion.fuse(prompt, query, fact)
                    print(f"[FUSION: Strategy = {fusion_result['type']}]")
                    print(f"[FUSION: Confidence = {fusion_result['confidence']:.2f}]\n")

                    new_prompt = fusion_result["fused_prompt"]
                    injected_context = fusion_result["context"]
                else:
                    new_prompt = prompt.replace(f"[RETRIEVE: {query}]", "")
                    injected_context = fact

                print("Continuing with context...\n")

                new_response = self.stream_generate(
                    new_prompt, injected_context=injected_context
                )

                for inner_chunk in new_response:
                    inner_token = inner_chunk["response"]
                    sys.stdout.write(inner_token)
                    sys.stdout.flush()

                return

        print()


if __name__ == "__main__":
    print("Initializing SIM with Ollama (gemma3:1b), FAISS, and Fusion Module...")
    sim = SIM(use_advanced_fusion=True)

    test_prompt = (
        "The capital of France is [RETRIEVE: capital of France]. It is known for"
    )

    print(f"Running test with prompt:\n{test_prompt}\n")
    print("=" * 60)

    sim.run(test_prompt, use_fusion=True)
