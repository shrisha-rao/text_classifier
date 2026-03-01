#!/usr/bin/env python3
"""
Generate synthetic training and test data using OpenAI API for zero-shot text classification.
Labels are split into disjoint sets for train and test (true zero-shot setup).
"""

import json
import os
import random
import time
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# Configuration
NUM_TOTAL_LABELS = 50
NUM_TRAIN_LABELS = 40
NUM_TEST_LABELS = NUM_TOTAL_LABELS - NUM_TRAIN_LABELS  #10
NUM_TRAIN_SAMPLES = 5000
NUM_TEST_SAMPLES = 500
MIN_LABELS_PER_SAMPLE = 1
MAX_LABELS_PER_SAMPLE = 3
OUTPUT_TRAIN = "data/train_data.json"
OUTPUT_TEST = "data/test_data.json"
# OUTPUT_COMBINED = "data/synthetic_data.json"
OPENAI_MODEL = "gpt-4o-mini"  # Cheaper + faster
SEED = 42
MAX_RETRIES = 3

random.seed(SEED)
os.makedirs("data", exist_ok=True)

# OPENAI_API_KEY environment variable
client = OpenAI()

assert MIN_LABELS_PER_SAMPLE <= MAX_LABELS_PER_SAMPLE
assert MIN_LABELS_PER_SAMPLE >= 1


def generate_label_list():
    """Generate diverse labels using structured JSON prompt."""
    prompt = (
        f"Generate exactly {NUM_TOTAL_LABELS} diverse topics/categories for text classification. "
        "Use single words or short phrases (2-3 words max) suitable as labels. "
        "Examples: 'Machine Learning', 'Climate Change', 'Stock Market', 'Basketball'. "
        "Return ONLY a JSON object: {\"labels\": [\"label1\", \"label2\", ...]} with exactly"
        f" {NUM_TOTAL_LABELS} strings.")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,  # Lower for consistency
                max_tokens=300,
                response_format={"type": "json_object"})
            content = response.choices[0].message.content
            data = json.loads(content)
            labels = data["labels"]

            if len(labels) >= NUM_TOTAL_LABELS:
                return [lbl.strip() for lbl in labels]
            else:
                print(
                    f"Warning: Got {len(labels)} labels, retrying... (attempt {attempt+1})"
                )
        except Exception as e:
            print(f"Label generation attempt {attempt+1} failed: {e}")
            time.sleep(5)

    # Hard fallback
    print("Using fallback labels")
    return [f"Topic_{i}" for i in range(1, 51)]


def split_labels(labels):
    """Randomly split into disjoint train/test sets."""
    shuffled = labels.copy()
    random.shuffle(shuffled)
    train_labels = shuffled[:NUM_TRAIN_LABELS]
    test_labels = shuffled[NUM_TRAIN_LABELS:NUM_TRAIN_LABELS + NUM_TEST_LABELS]
    assert len(set(train_labels)
               & set(test_labels)) == 0, "Label overlap detected!"
    return train_labels, test_labels


def generate_sample(label_set):
    """Generate one realistic sample with given labels."""
    max_k = min(MAX_LABELS_PER_SAMPLE, len(label_set))
    k = random.randint(MIN_LABELS_PER_SAMPLE, max_k)
    # k = random.randint(MIN_LABELS_PER_SAMPLE, MAX_LABELS_PER_SAMPLE)

    chosen_labels = random.sample(label_set, k)
    labels_str = ", ".join(chosen_labels)

    prompt = (
        f"Write 1-2 specific, detailed sentences as if from a professional news article "
        f"that implicitly relate to: {labels_str}. "
        "The sentences should be have minimum of 5 words to and maximum of 25 words"
        "Include concrete elements such as numbers, locations, people, trends, or events. "
        "Do NOT mention the topic names directly. "
        "Do NOT be generic. "
        "Return only the sentences."
        """Examples: 1. "The central bank raised interest rates to combat inflation"
                     2. "A breakthrough in quantum computing was announced by researchers"
                     3. "The local team celebrated their championship victory"
                     4. "Doctors reported a decline in seasonal flu cases""")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=OPENAI_MODEL,
                                                      messages=[{
                                                          "role":
                                                          "user",
                                                          "content":
                                                          prompt
                                                      }],
                                                      temperature=0.8,
                                                      max_tokens=80)
            text = response.choices[0].message.content.strip()
            text = text.strip('"').strip("'").strip()

            # Basic quality check
            # word_count = len(text.split())
            # if (3 <= word_count <= 120 and len(text) <= 500):
            if text and len(text.strip()) > 0:
                return {"text": text, "labels": chosen_labels}
        except Exception as e:
            print(f"Sample generation attempt {attempt+1} failed: {e}")
            time.sleep(10.0)

    # Fallback
    return {
        "text": f"Sample text about {labels_str.lower()}.",
        "labels": chosen_labels
    }
    # raise RuntimeError("Model failed to generate valid sample.")


def generate_dataset(label_set, num_samples, desc, label_counter):
    """Generate dataset with retries and label coverage tracking."""
    data = []
    for _ in tqdm(range(num_samples), desc=desc):
        sample = generate_sample(label_set)
        data.append(sample)

        # Track label usage
        for label in sample["labels"]:
            label_counter[label] += 1

        time.sleep(0.1)  # Rate limiting
    return data


def print_stats(data, dataset_name, label_counter):
    """Print dataset statistics."""
    print(f"\n{dataset_name} Stats:")
    print(f"Samples: {len(data)}")
    print(
        f"Avg labels/sample: {sum(len(s['labels']) for s in data)/len(data):.1f}"
    )

    total_mentions = sum(label_counter.values())
    print(f"Total label mentions: {total_mentions}")

    # Top 5 and bottom 5 labels
    sorted_labels = sorted(label_counter.items(),
                           key=lambda x: x[1],
                           reverse=True)
    print(f"Most frequent: {sorted_labels[:3]}")
    print(f"Least frequent: {sorted_labels[-3:]}")

    coverage = len([l for l in label_counter if label_counter[l] > 0
                    ]) / len(label_counter)
    print(f"Label coverage: {coverage:.1%}")


def plot_label_dist(label_counter, dataset_name, save_path):
    """Plot label frequency distribution."""
    labels, counts = zip(*label_counter.items())
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts)
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name}: Label Distribution")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("🚀 Generating synthetic data for zero-shot classification...")

    # 1. Generate labels
    print("\n1. Generating label set...")
    all_labels = generate_label_list()
    train_labels, test_labels = split_labels(all_labels)

    print(f"Generated {len(all_labels)} total labels")
    print(f"Train labels: {len(train_labels)} (no overlap with test)")
    print(f"Test labels:  {len(test_labels)} (zero-shot evaluation)")

    # 2. Generate datasets
    print("\n2. Generating training data...")
    train_counter = Counter()
    train_data = generate_dataset(train_labels, NUM_TRAIN_SAMPLES,
                                  "Train samples", train_counter)

    print("\n3. Generating test data...")
    test_counter = Counter()
    test_data = generate_dataset(test_labels, NUM_TEST_SAMPLES, "Test samples",
                                 test_counter)

    # 3. Print statistics
    print_stats(train_data, "TRAIN", train_counter)
    print_stats(test_data, "TEST", test_counter)

    # 4. Save datasets
    print("\n4. Saving datasets...")
    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(OUTPUT_TEST, "w") as f:
        json.dump(test_data, f, indent=2)

    # Combined dataset
    # combined_data = train_data + test_data
    # with open(OUTPUT_COMBINED, "w") as f:
    #     json.dump(combined_data, f, indent=2)

    print(f"Saved:")
    print(f"{OUTPUT_TRAIN} ({len(train_data)} samples)")
    print(f"{OUTPUT_TEST} ({len(test_data)} samples)")
    # print(f"   {OUTPUT_COMBINED} ({len(combined_data)} total)")

    # 5. Plot distributions
    # print("\n5. Generating label distribution plots...")
    # plot_label_dist(train_counter, "Train Label Distribution",
    #                 "data/train_label_dist.png")
    # plot_label_dist(test_counter, "Test Label Distribution",
    #                 "data/test_label_dist.png")
    # print("Plots saved: data/*_label_dist.png")

    # print("\nData generation complete!")
    # print("Use `data/synthetic_data.json` with your existing train.py")
    # print("Use `data/test_data.json` with your benchmark.py")


if __name__ == "__main__":
    main()
