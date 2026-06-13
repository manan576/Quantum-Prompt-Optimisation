GENES = {
    "persona": [
        "none",
        "expert_moderator",
        "critic_team",
        "reviewer_team"
    ],
    "reasoning": [
        "direct",
        "step_by_step",
        "protected_group_first",
        "multi_role"
    ],
    "definitions": [
        "none",
        "short",
        "detailed"
    ],
    "fewshot": [
        0,
        1,
        2
    ],
    "output_format": [
        "labels_only",
        "labels_plus_reason"
    ],
    "constraint": [
        "none",
        "exact_labels_only",
        "avoid_bias"
    ]
}

def print_schema():
    for gene_name, values in GENES.items():
        print(f"{gene_name}: {values}")

if __name__ == "__main__":
    print_schema()