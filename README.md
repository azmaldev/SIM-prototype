# SIM: Streaming Interleaved Memory

A working prototype of **Streaming Interleaved Memory** — token-level dynamic retrieval during AI generation.

This project demonstrates a novel approach where an AI model can interrupt its own generation to retrieve relevant facts from a vector database in real-time, injecting retrieved information back into the context and continuing generation seamlessly.

## Overview

When the model encounters a `[RETRIEVE: query]` pattern in the output, it:
1. Pauses generation
2. Searches a FAISS vector database for the most relevant fact
3. Injects the fact as context
4. Resumes generation

This creates truly interleaved memory retrieval during streaming generation.

## Installation

```bash
# Clone the repository
git clone https://github.com/mahammadazmal/SIM-prototype.git
cd SIM-prototype

# Install dependencies
pip install ollama faiss-cpu numpy sentence-transformers

# Install and start Ollama
# See: https://ollama.ai/download
```

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai) (with gemma3:1b model)
- FAISS (faiss-cpu)
- sentence-transformers

## How to Run

```bash
python sim.py
```

The default test prompt will run:
> "The capital of France is [RETRIEVE: capital of France]. It is known for"

The model will detect the retrieval pattern, fetch the relevant fact from the database, inject it, and continue generating.

## Architecture

- **sim.py**: Main script containing the SIM class
- **FAISS Index**: Pre-loaded with 10 facts using sentence-transformers embeddings
- **Ollama Integration**: Streams gemma3:1b tokens in real-time

## Research

Built as a proof-of-concept for a research paper by **Mahammad Azmal** (2026).

---

*"Memory should not be a retrieval step before generation — it should be interleaved within it."*
