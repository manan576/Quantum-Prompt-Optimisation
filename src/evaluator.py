import json
import os
import time
from google import genai

LABEL_NAMES = {
    "violence",
    "directed_vs_generalized",
    "gender",
    "race",
    "national_origin",
    "disability",
    "religion",
    "sexual_orientation"
}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def load_validation_data(path="data/ethos_val.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_labels_from_response(response_text):
    text = response_text.strip().lower()

    text = text.replace("\n", " ")
    text = text.replace("labels:", "")
    text = text.replace("label:", "")
    text = text.replace(".", " ")
    text = text.replace(";", ",")
    text = text.replace("|", ",")
    text = text.replace("`", "")
    text = text.strip()

    if text == "none":
        return []

    parts = [x.strip() for x in text.split(",")]
    labels = [x for x in parts if x in LABEL_NAMES]

    return sorted(set(labels))

def exact_match_score(predicted_labels, true_labels):
    return 1 if set(predicted_labels) == set(true_labels) else 0

def call_llm(prompt_text, model_name="gemini-2.5-flash-lite", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text
            )
            return response.text.strip()
        except Exception as e:
            print(f"LLM call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print("Waiting 25 seconds before retrying...")
                time.sleep(25)
            else:
                raise

def evaluate_prompt_on_subset(prompt_template, val_data, subset_size=2):
    scores = []
    extra_counts = []
    missing_counts = []
    detailed_results = []

    subset = val_data[:subset_size]

    for example in subset:
        user_message = example["text"]
        true_labels = example["labels"]

        full_prompt = prompt_template.replace("{user_message}", user_message)

        response = call_llm(full_prompt)
        time.sleep(13)

        predicted_labels = parse_labels_from_response(response)
        score = exact_match_score(predicted_labels, true_labels)

        extra = len(set(predicted_labels) - set(true_labels))
        missing = len(set(true_labels) - set(predicted_labels))

        scores.append(score)
        extra_counts.append(extra)
        missing_counts.append(missing)

        detailed_results.append({
            "message": user_message,
            "true_labels": true_labels,
            "response": response,
            "predicted_labels": predicted_labels,
            "score": score,
            "extra_labels": extra,
            "missing_labels": missing
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_extra = sum(extra_counts) / len(extra_counts) if extra_counts else 0.0
    avg_missing = sum(missing_counts) / len(missing_counts) if missing_counts else 0.0

    return avg_score, avg_extra, avg_missing, detailed_results

def evaluate_candidates(candidates, subset_size=2):
    val_data = load_validation_data()
    results = []

    for candidate in candidates:
        avg_score, avg_extra, avg_missing, details = evaluate_prompt_on_subset(
            candidate["prompt"],
            val_data,
            subset_size=subset_size
        )

        results.append({
            "bitstring": candidate["bitstring"],
            "chromosome": candidate["chromosome"],
            "prompt": candidate["prompt"],
            "avg_score": avg_score,
            "avg_extra": avg_extra,
            "avg_missing": avg_missing,
            "details": details
        })

    return results

if __name__ == "__main__":
    val_data = load_validation_data()
    print("Loaded validation samples:", len(val_data))