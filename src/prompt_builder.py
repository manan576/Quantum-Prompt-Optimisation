import json

LABEL_NAMES = [
    "violence",
    "directed_vs_generalized",
    "gender",
    "race",
    "national_origin",
    "disability",
    "religion",
    "sexual_orientation"
]

SHORT_DEFINITIONS = """
Possible hate speech categories:
- violence
- directed_vs_generalized
- gender
- race
- national_origin
- disability
- religion
- sexual_orientation
"""

DETAILED_DEFINITIONS = """
Possible hate speech categories:
- violence: content encouraging, supporting, or referring to physical harm or violent acts
- directed_vs_generalized: whether the hate is aimed at a specific target or a broader group
- gender: hateful or offensive content targeting gender
- race: hateful or offensive content targeting race
- national_origin: hateful or offensive content targeting nationality or national origin
- disability: hateful or offensive content targeting disability
- religion: hateful or offensive content targeting religion
- sexual_orientation: hateful or offensive content targeting sexual orientation
"""

def load_examples(path="data/ethos_train.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_fewshot_examples(examples, k):
    if k == 0:
        return ""

    chosen = examples[:k]
    parts = ["Examples:"]

    for i, ex in enumerate(chosen, start=1):
        parts.append(f"Example {i}:")
        parts.append(f"Message: {ex['text']}")
        parts.append(f"Labels: {', '.join(ex['labels']) if ex['labels'] else 'none'}")
        parts.append("")

    return "\n".join(parts)

def build_prompt(chromosome, fewshot_examples=None):
    sections = []

    persona = chromosome["persona"]
    reasoning = chromosome["reasoning"]
    definitions = chromosome["definitions"]
    fewshot = chromosome["fewshot"]
    output_format = chromosome["output_format"]
    constraint = chromosome["constraint"]

    if persona == "expert_moderator":
        sections.append("You are an expert content moderator.")
    elif persona == "critic_team":
        sections.append("You are a critic team analyzing whether the message contains hate speech.")
    elif persona == "reviewer_team":
        sections.append("You are a reviewer team checking hate speech classification carefully.")

    if reasoning == "direct":
        sections.append("Determine whether the message contains hate speech and assign the correct labels.")
    elif reasoning == "step_by_step":
        sections.append("Analyze the message step by step, then determine whether it contains hate speech and assign the correct labels.")
    elif reasoning == "protected_group_first":
        sections.append("First identify whether the message targets a protected group, then determine whether it contains hate speech and assign the correct labels.")
    elif reasoning == "multi_role":
        sections.append("First analyze the message carefully, then review the reasoning, then assign the correct hate speech labels.")

    sections.append("The possible labels are: " + ", ".join(LABEL_NAMES) + ".")

    if definitions == "short":
        sections.append(SHORT_DEFINITIONS.strip())
    elif definitions == "detailed":
        sections.append(DETAILED_DEFINITIONS.strip())

    if constraint == "exact_labels_only":
        sections.append("Use only labels from the given label set. Do not invent new labels.")
    elif constraint == "avoid_bias":
        sections.append("Avoid biased assumptions and base your classification only on the content of the message.")

    if fewshot > 0 and fewshot_examples is not None:
        sections.append(format_fewshot_examples(fewshot_examples, fewshot))

    if output_format == "labels_only":
        sections.append("Output only the final labels as a comma-separated list. If no label applies, output: none.")
    elif output_format == "labels_plus_reason":
        sections.append("Output the final labels as a comma-separated list, followed by a short explanation.")

    sections.append("Now classify the following message:\n{user_message}")

    return "\n\n".join(sections)

if __name__ == "__main__":
  examples = load_examples()

  sample_chromosome = {
        "persona": "expert_moderator",
        "reasoning": "step_by_step",
        "definitions": "short",
        "fewshot": 1,
        "output_format": "labels_only",
        "constraint": "exact_labels_only"
  }

  prompt = build_prompt(sample_chromosome, examples)
  print(prompt)