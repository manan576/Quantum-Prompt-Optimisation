import numpy as np

from quantum_sampler import sample_bitstrings, update_theta
from decoder import decode_bitstring
from prompt_builder import build_prompt, load_examples
from evaluator import evaluate_candidates

N_QUBITS = 11


def get_bad_baseline_candidate():
    examples = load_examples()

    bad_chromosome = {
        "persona": "none",
        "reasoning": "direct",
        "definitions": "none",
        "fewshot": 0,
        "output_format": "labels_plus_reason",
        "constraint": "none"
    }

    bad_prompt = build_prompt(bad_chromosome, examples)

    return {
        "bitstring": "manual_bad",
        "chromosome": bad_chromosome,
        "prompt": bad_prompt
    }

def ranking_key(result):
    return (result["avg_score"], -result["avg_extra"], -result["avg_missing"])



def normalize_bitstring(bitstring):
    return bitstring[::-1]

def generate_prompt_candidates(theta, shots=16, max_candidates=10):
    counts = sample_bitstrings(theta, shots=shots)
    raw_bitstrings = list(counts.keys())

    unique_candidates = []
    seen = set()

    for raw in raw_bitstrings:
        normalized = normalize_bitstring(raw)

        if normalized not in seen and len(normalized) == N_QUBITS:
            seen.add(normalized)
            unique_candidates.append(normalized)

        if len(unique_candidates) >= max_candidates:
            break

    examples = load_examples()

    results = []
    for bitstring in unique_candidates:
        chromosome = decode_bitstring(bitstring)
        prompt = build_prompt(chromosome, examples)

        results.append({
            "bitstring": bitstring,
            "chromosome": chromosome,
            "prompt": prompt
        })

    return results

if __name__ == "__main__":
    theta = np.ones(N_QUBITS) * (np.pi / 4)
    n_generations = 4

    best_overall = None
    history = []

    print("\n" + "#" * 100)
    print("BASELINE BAD PROMPT")
    print("#" * 100)

    baseline_candidate = get_bad_baseline_candidate()
    baseline_result = evaluate_candidates([baseline_candidate], subset_size=2)[0]

    print("Chromosome:", baseline_result["chromosome"])
    print("Average Score:", baseline_result["avg_score"])
    print("Average Extra Labels:", baseline_result["avg_extra"])
    print("Average Missing Labels:", baseline_result["avg_missing"])

    for d in baseline_result["details"]:
        print("-" * 40)
        print("Message:", d["message"])
        print("True Labels:", d["true_labels"])
        print("LLM Response:", d["response"])
        print("Predicted Labels:", d["predicted_labels"])
        print("Score:", d["score"])
        print("Extra Labels:", d["extra_labels"])
        print("Missing Labels:", d["missing_labels"])

    for generation in range(n_generations):
        print("\n" + "#" * 100)
        print(f"GENERATION {generation + 1}")
        print("#" * 100)

        candidates = generate_prompt_candidates(theta, shots=20, max_candidates=10)
        filtered = [c for c in candidates if c["chromosome"]["output_format"] == "labels_only"][:2]

        print(f"Evaluating {len(filtered)} candidates...\n")

        results = evaluate_candidates(filtered, subset_size=2)
        results.sort(key=ranking_key, reverse=True)

        for i, result in enumerate(results, start=1):
            print("=" * 80)
            print(f"CANDIDATE {i}")
            print("Bitstring:", result["bitstring"])
            print("Chromosome:", result["chromosome"])
            print("Average Score:", result["avg_score"])
            print("Average Extra Labels:", result["avg_extra"])
            print("Average Missing Labels:", result["avg_missing"])

            for d in result["details"]:
                print("-" * 40)
                print("Message:", d["message"])
                print("True Labels:", d["true_labels"])
                print("LLM Response:", d["response"])
                print("Predicted Labels:", d["predicted_labels"])
                print("Score:", d["score"])
                print("Extra Labels:", d["extra_labels"])
                print("Missing Labels:", d["missing_labels"])

        elite = results[0]
        elite_bitstrings = [elite["bitstring"]]

        history.append({
            "generation": generation + 1,
            "bitstring": elite["bitstring"],
            "chromosome": elite["chromosome"],
            "prompt": elite["prompt"],
            "avg_score": elite["avg_score"],
            "avg_extra": elite["avg_extra"],
            "avg_missing": elite["avg_missing"]
        })

        if best_overall is None or ranking_key(elite) > ranking_key(best_overall):
            best_overall = elite.copy()

        old_theta = theta.copy()
        theta = update_theta(theta, elite_bitstrings, learning_rate=0.15)

        print("\nElite bitstring(s):", elite_bitstrings)
        print("Old theta:", np.round(old_theta, 3))
        print("Updated theta:", np.round(theta, 3))

    print("\n" + "=" * 100)
    print("BEST PROMPT FOUND OVER ALL GENERATIONS")
    print("=" * 100)
    print("Bitstring:", best_overall["bitstring"])
    print("Chromosome:", best_overall["chromosome"])
    print("Average Score:", best_overall["avg_score"])
    print("Average Extra Labels:", best_overall["avg_extra"])
    print("Average Missing Labels:", best_overall["avg_missing"])
    print("\nBEST PROMPT TEXT:\n")
    print(best_overall["prompt"])

    print("\n" + "=" * 100)
    print("ELITE HISTORY")
    print("=" * 100)
    for h in history:
        print(
            f"Generation {h['generation']}: "
            f"score={h['avg_score']}, extra={h['avg_extra']}, missing={h['avg_missing']}, "
            f"chromosome={h['chromosome']}"
        )

    print("\n" + "=" * 100)
    print("BASELINE VS BEST FOUND")
    print("=" * 100)

    comparison_candidates = [
        baseline_candidate,
        {
            "bitstring": best_overall["bitstring"],
            "chromosome": best_overall["chromosome"],
            "prompt": best_overall["prompt"]
        }
    ]

    comparison_results = evaluate_candidates(comparison_candidates, subset_size=2)

    for label, result in zip(["BAD BASELINE", "BEST FOUND"], comparison_results):
        print("\n" + "-" * 80)
        print(label)
        print("Chromosome:", result["chromosome"])
        print("Average Score:", result["avg_score"])
        print("Average Extra Labels:", result["avg_extra"])
        print("Average Missing Labels:", result["avg_missing"])

        for d in result["details"]:
            print("-" * 40)
            print("Message:", d["message"])
            print("True Labels:", d["true_labels"])
            print("LLM Response:", d["response"])
            print("Predicted Labels:", d["predicted_labels"])
            print("Score:", d["score"])
            print("Extra Labels:", d["extra_labels"])
            print("Missing Labels:", d["missing_labels"])