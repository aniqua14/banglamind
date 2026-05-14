# src/data_loader.py

import pandas as pd
from datasets import load_dataset


def load_bangla_emotion_data():
    """
    Loads the dair-ai/emotion dataset from Hugging Face Hub.
    Note: This is an English dataset used to build and validate our pipeline.
    In Phase 2, we will layer in Bangla-specific text and BanglaBERT tokenization.
    """

    print("Downloading dataset...")

    # Removed trust_remote_code — no longer supported in newer datasets versions
    dataset = load_dataset("dair-ai/emotion")

    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    val_df = pd.DataFrame(dataset['validation'])

    print(f"Train size : {len(train_df)}")
    print(f"Test size  : {len(test_df)}")
    print(f"Val size   : {len(val_df)}")

    return train_df, val_df, test_df


def filter_four_emotions(df):
    """
    Keep only our 4 target emotions and map numeric labels to names.

    Original label mapping:
        0 = sadness   ✓ keep
        1 = joy       ✓ keep
        2 = love      ✗ remove
        3 = anger     ✓ keep
        4 = fear      ✓ keep
        5 = surprise  ✗ remove
    """

    target_labels = [0, 1, 3, 4]
    filtered_df = df[df['label'].isin(target_labels)].copy()

    label_map = {
        0: 'sadness',
        1: 'joy',
        3: 'anger',
        4: 'fear'
    }
    filtered_df['emotion'] = filtered_df['label'].map(label_map)
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


if __name__ == "__main__":
    train_df, val_df, test_df = load_bangla_emotion_data()

    # Apply emotion filter
    train_df = filter_four_emotions(train_df)
    val_df = filter_four_emotions(val_df)
    test_df = filter_four_emotions(test_df)

    print("\n--- After filtering to 4 emotions ---")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n--- Class distribution (train) ---")
    print(train_df['emotion'].value_counts())

    print("\n--- Sample texts per emotion ---")
    for emotion in ['joy', 'sadness', 'fear', 'anger']:
        sample = train_df[train_df['emotion'] == emotion]['text'].iloc[0]
        print(f"\n[{emotion}]: {sample}")

    print("\n--- Text length stats ---")
    train_df['char_count'] = train_df['text'].apply(len)
    train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
    print(train_df[['char_count', 'word_count']].describe().round(2))
