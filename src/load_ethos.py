from datasets import load_dataset
import random
import json
import os

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

def convert_example(raw_example):
    text = raw_example["text"]
    labels = []

    for label_name in LABEL_NAMES:
        value = raw_example[label_name]
        if value is True or value == 1 or (isinstance(value, (int, float)) and value >= 0.5):
            labels.append(label_name)

    return {
        "text": text,
        "labels": labels
    }

def load_ethos_processed():
    dataset = load_dataset("iamollas/ethos", "multilabel")
    processed = [convert_example(x) for x in dataset["train"]]
    return processed

def make_small_split(data, seed=42):
    random.seed(seed)
    data_copy = data[:]
    random.shuffle(data_copy)

    train = data_copy[:10]
    val = data_copy[10:20]
    test = data_copy[20:40]

    return train, val, test

def save_split(train, val, test, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/ethos_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(f"{out_dir}/ethos_val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    with open(f"{out_dir}/ethos_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    data = load_ethos_processed()
    train, val, test = make_small_split(data, seed=42)
    save_split(train, val, test)

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")
    print("Sample example:")
    print(train[0])