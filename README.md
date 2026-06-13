# Quantum Prompt Optimization

This project is a prototype implementing a low-cost, quantum-inspired extension of GAAPO (Genetic Algorithm for Automated Prompt Optimization) tailored for the ETHOS multilabel hate-speech classification task. 

## Overview

Traditional prompt optimization often relies on expensive, iterative LLM-based prompt generation. This project bypasses that by converting prompt optimization into a structured, gene-based search problem. This formulation makes the optimization compatible with a quantum-inspired search approach.

Key highlights:
- **Gene-Based Prompts:** Prompt structures are encoded into discrete "genes" (e.g., persona, reasoning style, definition depth, few-shot examples, output format, and constraints).
- **Quantum-Inspired Search:** An 11-qubit circuit (simulated via Qiskit) is used to sample candidate prompt chromosomes. Each qubit represents a binary decision in the chromosome, and single-qubit rotation gates govern the sampling probabilities.
- **Cost Efficiency:** The free Gemini model (`gemini-2.5-flash-lite`) is used strictly to evaluate a small batch of promising candidate prompts on a reduced ETHOS subset, minimizing expensive LLM calls.
- **Evaluation:** The generated prompts are scored based on exact-match multilabel accuracy, while also tracking extra and missing labels to measure precision.

## Architecture

- `src/main.py`: The entry point that orchestrates the generational search, evaluates candidate prompts, and updates the sampling probabilities (theta) based on the elite candidates.
- `src/prompt_builder.py` / `src/decoder.py` / `src/prompt_schema.py`: Handles translating raw bitstrings from the quantum circuit into structured prompt components and finally into a complete prompt string.
- `src/quantum_sampler.py`: Uses Qiskit to build a quantum circuit, parameterize rotation gates, and sample candidate prompt bitstrings.
- `src/evaluator.py`: Interfaces with the Gemini API to test the prompt against a subset of the ETHOS dataset.

## Requirements

The project requires the following Python libraries:
- `numpy`
- `qiskit`
- `qiskit_aer`
- `google-genai`
- `datasets==2.16.0` (Note: This specific version is required to ensure compatibility with the ETHOS dataset's legacy loader script).

You will also need a Gemini API key. Ensure that `GEMINI_API_KEY` is exported in your environment.

## Usage

1. **Set your API Key:**
   ```powershell
   $env:GEMINI_API_KEY="your-api-key-here"
   ```
2. **Run the Search:**
   ```powershell
   python src/main.py
   ```
