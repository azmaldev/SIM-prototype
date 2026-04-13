import re
import numpy as np
from typing import Tuple, Dict
from sentence_transformers import SentenceTransformer


class FusionModule:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_key_entity(self, text: str) -> str:
        words = text.split()

        for word in words:
            clean_word = word.strip('.,;:!?"')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                return clean_word

        return words[0].strip('.,;:!?"') if words else text

    def _extract_query_context(self, prompt: str, query: str) -> Tuple[str, str, str]:
        pattern = rf"\[RETRIEVE:\s*{re.escape(query)}\]"
        match = re.search(pattern, prompt)

        if not match:
            return "", query, ""

        before = prompt[: match.start()]
        after = prompt[match.end() :]

        return before, query, after

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            emb1 = self.embed_model.encode([text1])[0]
            emb2 = self.embed_model.encode([text2])[0]

            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )
            return float(similarity)
        except Exception:
            return float(text1.lower() == text2.lower())

    def fuse(self, prompt: str, query: str, retrieved_fact: str) -> Dict:
        key_entity = self.extract_key_entity(retrieved_fact)
        before, _, after = self._extract_query_context(prompt, query)
        confidence = self._calculate_semantic_similarity(query, key_entity)

        if confidence >= 0.5:
            fused_prompt = f"{before}{key_entity}{after}"
            return {
                "type": "token_replacement",
                "fused_prompt": fused_prompt,
                "context": None,
                "confidence": confidence,
            }
        else:
            fallback_prompt = prompt.replace(f"[RETRIEVE: {query}]", "")
            return {
                "type": "context_injection",
                "fused_prompt": fallback_prompt,
                "context": retrieved_fact,
                "confidence": confidence,
            }
