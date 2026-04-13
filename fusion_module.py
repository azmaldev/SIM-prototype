import re
from typing import Tuple
from sentence_transformers import SentenceTransformer
import nltk
from nltk import pos_tag, word_tokenize
import numpy as np

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")


class FusionModule:
    """
    Intelligently merges retrieved facts into the generation stream
    while maintaining syntactic and semantic coherence.
    """

    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_key_entity(self, text: str) -> str:
        """
        Extract the most important noun/entity from retrieved text.

        Example:
          Input:  "Paris is the capital and largest city of France."
          Output: "Paris"
        """
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        proper_nouns = [word for word, pos in tagged if pos == "NNP"]
        if proper_nouns:
            return proper_nouns[0]

        nouns = [word for word, pos in tagged if pos in ["NN", "NNS"]]
        if nouns:
            return nouns[0]

        return tokens[0] if tokens else text

    def extract_query_context(self, prompt: str, query: str) -> Tuple[str, str, str]:
        """
        Extract context around [RETRIEVE: query] to understand what kind of
        information is expected at that position.

        Returns: (before_context, query, after_context)

        Example:
          prompt = "The capital of France is [RETRIEVE: capital of France]. It is known for"
          Returns: ("The capital of France is", "capital of France", ". It is known for")
        """
        pattern = rf"\[RETRIEVE:\s*{re.escape(query)}\]"
        match = re.search(pattern, prompt)

        if not match:
            return "", query, ""

        start_pos = match.start()
        end_pos = match.end()

        before = prompt[:start_pos]
        after = prompt[end_pos:]

        return before, query, after

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Measure semantic similarity between two texts.
        Used to validate if retrieved fact matches the query context.
        """
        emb1 = self.embed_model.encode([text1])[0]
        emb2 = self.embed_model.encode([text2])[0]

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity

    def fuse(self, prompt: str, query: str, retrieved_fact: str) -> dict:
        """
        Main fusion function: intelligently merge retrieved fact into prompt.

        Algorithm:
          1. Extract the entity/key noun from retrieved fact
          2. Identify the expected position in prompt
          3. Replace [RETRIEVE: ...] with extracted entity
          4. Validate coherence
          5. Return fused prompt
        """
        key_entity = self.extract_key_entity(retrieved_fact)

        before, _, after = self.extract_query_context(prompt, query)

        fused = f"{before}{key_entity}{after}"

        similarity = self.semantic_similarity(query, key_entity)

        if similarity < 0.3:
            return {
                "type": "context_injection",
                "fused_prompt": prompt.replace(f"[RETRIEVE: {query}]", ""),
                "context": retrieved_fact,
                "confidence": similarity,
            }

        return {
            "type": "token_replacement",
            "fused_prompt": fused,
            "context": None,
            "confidence": similarity,
        }


class AdvancedFusionModule(FusionModule):
    """
    Enhanced fusion with language model validation.
    Uses a small model to score fusion quality.
    """

    def __init__(self, scoring_model="gpt2"):
        super().__init__()
        self.scoring_model = scoring_model
