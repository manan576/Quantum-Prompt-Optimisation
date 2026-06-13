PERSONA_MAP = {
    "00": "none",
    "01": "expert_moderator",
    "10": "critic_team",
    "11": "reviewer_team"
}

REASONING_MAP = {
    "00": "direct",
    "01": "step_by_step",
    "10": "protected_group_first",
    "11": "multi_role"
}

DEFINITIONS_MAP = {
    "00": "none",
    "01": "short",
    "10": "detailed",
    "11": "short"
}

FEWSHOT_MAP = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 1
}

OUTPUT_MAP = {
    "0": "labels_only",
    "1": "labels_plus_reason"
}

CONSTRAINT_MAP = {
    "00": "none",
    "01": "exact_labels_only",
    "10": "avoid_bias",
    "11": "exact_labels_only"
}

def decode_bitstring(bitstring):
    """
    Expected bitstring length = 11
    Layout:
    [0:2]   persona
    [2:4]   reasoning
    [4:6]   definitions
    [6:8]   fewshot
    [8:9]   output_format
    [9:11]  constraint
    """

    if len(bitstring) != 11:
        raise ValueError(f"Expected 11-bit string, got length {len(bitstring)}")

    chromosome = {
        "persona": PERSONA_MAP[bitstring[0:2]],
        "reasoning": REASONING_MAP[bitstring[2:4]],
        "definitions": DEFINITIONS_MAP[bitstring[4:6]],
        "fewshot": FEWSHOT_MAP[bitstring[6:8]],
        "output_format": OUTPUT_MAP[bitstring[8:9]],
        "constraint": CONSTRAINT_MAP[bitstring[9:11]]
    }

    return chromosome

if __name__ == "__main__":
    sample = "01100110101"
    decoded = decode_bitstring(sample)
    print("Bitstring:", sample)
    print("Decoded chromosome:")
    print(decoded)


from prompt_builder import build_prompt, load_examples

if __name__ == "__main__":
    sample = "01100110101"
    decoded = decode_bitstring(sample)

    print("Bitstring:", sample)
    print("Decoded chromosome:")
    print(decoded)

    examples = load_examples()
    prompt = build_prompt(decoded, examples)

    print("\nGenerated prompt:\n")
    print(prompt)