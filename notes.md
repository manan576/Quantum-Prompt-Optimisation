This prototype implements a low-cost quantum-inspired extension of GAAPO for the ETHOS multilabel hate-speech task. Because only a free LLM is available, the system avoids repeated LLM-based prompt generation and instead uses a Qiskit-assisted structured prompt search. Prompt candidates are represented through discrete genes such as persona, reasoning style, definition depth, output format, and constraints. Qiskit is used to sample candidate prompt structures, while the free LLM is used only to evaluate a small number of prompts on a reduced ETHOS subset. The prototype uses exact-match multilabel accuracy for scoring.

Prototype constraints
Only one dataset: ETHOS
Reduced subset due to free LLM budget
No APO / OPRO in version 1
Qiskit used for prompt-structure sampling
LLM calls minimized through small validation batches

During dataset integration, the ETHOS loader failed under the latest Hugging Face datasets package because support for legacy dataset-loading scripts has been removed in recent versions. To preserve compatibility with the ETHOS multilabel dataset, the prototype pins datasets==2.16.0 and uses that environment for reproducible loading.

“Where is your novelty?”

you can say:

My novelty starts by converting prompt optimization from free-text evolution into a structured gene-based search problem. That makes it compatible with quantum-inspired search and also reduces the number of expensive LLM-based prompt rewrites.

That is a very good answer.

The prototype uses an 11-qubit Qiskit circuit to sample candidate prompt chromosomes. Each qubit corresponds to one binary decision in the chromosome, and single-qubit rotation gates control the probability of measuring 0 or 1. Measuring the circuit produces bitstring candidates, which are then decoded into structured prompt configurations. This forms the quantum-inspired search layer of the system