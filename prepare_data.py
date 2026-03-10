"""Prepare multilingual emotion datasets for training, validation, and testing.

This script converts the notebook preprocessing pipeline into a reproducible CLI workflow.
It loads EmpatheticDialogues + DailyDialog exports, remaps labels, creates a multilingual
(EN/ES) dataset, performs stratified splitting by language and label, balances the train set,
and saves CSV files under the data directory.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.utils import resample


# Label remapping from fine-grained emotions to target classes.
EMPATHETIC_LABEL_MAP = {
    "angry": "anger",
    "annoyed": "anger",
    "furious": "anger",
    "jealous": "anger",
    "disgusted": "disgust",
    "afraid": "fear",
    "anxious": "fear",
    "apprehensive": "fear",
    "terrified": "fear",
    "caring": "happiness",
    "confident": "happiness",
    "content": "happiness",
    "excited": "happiness",
    "faithful": "happiness",
    "grateful": "happiness",
    "joyful": "happiness",
    "hopeful": "happiness",
    "proud": "happiness",
    "trusting": "happiness",
    "sad": "sadness",
    "ashamed": "sadness",
    "devastated": "sadness",
    "disappointed": "sadness",
    "embarrassed": "sadness",
    "guilty": "sadness",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "sentimental": "sadness",
    "impressed": "surprise",
    "surprised": "surprise",
}


def load_empathetic_dialogues(data_dir: Path) -> pd.DataFrame:
    """Load and align EmpatheticDialogues segments with dialogue-level emotion labels."""
    seg_path = data_dir / "MPATHY" / "MPATHY_translation_en2es.csv"
    label_path = data_dir / "MPATHY" / "MPATHY_dialoginfo.csv"

    data_seg = pd.read_csv(seg_path)
    data_label = pd.read_csv(label_path)

    df = pd.DataFrame()
    df["uid"] = data_seg["UID"]
    df["text"] = data_seg["SEG"]
    df["translation"] = data_seg["translation"]

    # Each dialogue-level label is repeated for all turns in that dialogue.
    expanded_labels = []
    for _, row in data_label.iterrows():
        turns = int(row["turns"]) + 1
        expanded_labels.extend([row["emotion"]] * turns)

    if len(expanded_labels) != len(df):
        raise ValueError(
            "EmpatheticDialogues alignment mismatch: "
            f"{len(expanded_labels)} labels for {len(df)} rows."
        )

    df["label"] = expanded_labels

    # Remove classes that were explicitly dropped in the notebook workflow.
    df = df[~df["label"].isin(["anticipating", "prepared"])].copy()

    # Remap fine-grained labels to the reduced emotion set.
    df["label"] = df["label"].replace(EMPATHETIC_LABEL_MAP)

    return df[["uid", "label", "text", "translation"]].reset_index(drop=True)


def load_dailydialog(data_dir: Path) -> pd.DataFrame:
    """Load DailyDialog data and normalize the neutral class naming."""
    seg_path = data_dir / "DAILYD" / "DAILYD_translation_en2es.csv"
    label_path = data_dir / "DAILYD" / "DAILYD_dialoginfo.csv"

    data_seg = pd.read_csv(seg_path)
    data_label = pd.read_csv(label_path)

    df = pd.DataFrame()
    df["uid"] = data_seg["UID"]
    df["text"] = data_seg["SEG"]
    df["translation"] = data_seg["translation"]
    df["label"] = data_label["emotion"].replace({"no emotion": "neutral"})

    return df[["uid", "label", "text", "translation"]].reset_index(drop=True)


def build_multilingual_dataset(ed_df: pd.DataFrame, dd_df: pd.DataFrame) -> pd.DataFrame:
    """Merge corpora and duplicate each row into EN and ES text variants."""
    data = pd.concat([ed_df, dd_df], ignore_index=True)

    data_en = data[["label", "text"]].copy()
    data_en["language"] = "en"

    data_es = data[["label", "translation"]].copy().rename(columns={"translation": "text"})
    data_es["language"] = "es"

    data_multi = pd.concat([data_en, data_es], ignore_index=True)

    # Remove repeated utterances to prevent leakage and duplicated samples.
    data_multi = data_multi.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return data_multi


def _sample_exact(df: pd.DataFrame, n_samples: int, random_state: int) -> pd.DataFrame:
    """Sample exactly n rows with a defensive error if the class is too small."""
    if len(df) < n_samples:
        raise ValueError(
            f"Not enough samples to draw {n_samples}; available={len(df)}."
        )
    return df.sample(n=n_samples, random_state=random_state)


def split_train_val_test(
    df: pd.DataFrame,
    num_samples_per_class_test: int,
    num_samples_per_class_val: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create balanced splits by class and language, matching notebook logic."""
    test_list_en, test_list_es = [], []
    val_list_en, val_list_es = [], []
    train_list_en, train_list_es = [], []

    for label in sorted(df["label"].unique()):
        class_samples_en = df[(df["label"] == label) & (df["language"] == "en")]
        class_samples_es = df[(df["label"] == label) & (df["language"] == "es")]

        class_samples_en = class_samples_en[~class_samples_en.duplicated(subset=["text"])]
        class_samples_es = class_samples_es[~class_samples_es.duplicated(subset=["text"])]

        test_data_en = _sample_exact(class_samples_en, num_samples_per_class_test, random_state)
        test_data_es = _sample_exact(class_samples_es, num_samples_per_class_test, random_state)

        remaining_en = class_samples_en.drop(test_data_en.index)
        remaining_es = class_samples_es.drop(test_data_es.index)

        val_data_en = _sample_exact(remaining_en, num_samples_per_class_val, random_state)
        val_data_es = _sample_exact(remaining_es, num_samples_per_class_val, random_state)

        train_data_en = remaining_en.drop(val_data_en.index)
        train_data_es = remaining_es.drop(val_data_es.index)

        # Keep train text disjoint from held-out sets.
        train_data_en = train_data_en[~train_data_en["text"].isin(test_data_en["text"])]
        train_data_es = train_data_es[~train_data_es["text"].isin(test_data_es["text"])]

        test_list_en.append(test_data_en)
        test_list_es.append(test_data_es)
        val_list_en.append(val_data_en)
        val_list_es.append(val_data_es)
        train_list_en.append(train_data_en)
        train_list_es.append(train_data_es)

    train_df = pd.concat(train_list_en + train_list_es).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_list_en + val_list_es).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_list_en + test_list_es).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


def balance_train_dataset(
    train_df: pd.DataFrame,
    neutral_frac: float,
    upsample_ratios: dict[str, int],
    random_state: int,
) -> pd.DataFrame:
    """Downsample neutral and upsample minority classes to reduce imbalance."""
    neutral_class = train_df[train_df["label"] == "neutral"]

    neutral_downsampled_en = neutral_class[neutral_class["language"] == "en"].sample(
        frac=neutral_frac,
        random_state=random_state,
    )
    neutral_downsampled_es = neutral_class[neutral_class["language"] == "es"].sample(
        frac=neutral_frac,
        random_state=random_state,
    )
    neutral_downsampled = pd.concat([neutral_downsampled_en, neutral_downsampled_es])

    upsampled_parts = []
    for label, ratio in upsample_ratios.items():
        class_samples = train_df[train_df["label"] == label]
        upsampled_parts.append(
            resample(
                class_samples,
                replace=True,
                n_samples=len(class_samples) * ratio,
                random_state=random_state,
            )
        )

    # Keep classes not listed in upsample_ratios untouched, except neutral.
    for label in train_df["label"].unique():
        if label not in upsample_ratios and label != "neutral":
            upsampled_parts.append(train_df[train_df["label"] == label])

    balanced = pd.concat([neutral_downsampled] + upsampled_parts)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced


def print_distribution(df: pd.DataFrame, name: str) -> None:
    """Print class and language breakdown for quick sanity checks."""
    print(f"\n{name} class distribution:\n{df['label'].value_counts()}")
    print(f"\n{name} language x label distribution:")
    print(df.groupby(["language", "label"]).size().unstack(fill_value=0))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare multilingual emotion datasets.")
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument("--num-test", type=int, default=250, help="Samples per class/language for test set.")
    parser.add_argument("--num-val", type=int, default=250, help="Samples per class/language for validation set.")
    parser.add_argument("--neutral-frac", type=float, default=0.6, help="Downsampling fraction for neutral class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def main() -> None:
    """Run the full preprocessing, splitting, and balancing pipeline."""
    args = parse_args()
    data_dir = Path(args.data_dir)

    upsample_ratios = {
        "anger": 2,
        "fear": 3,
        "disgust": 9,
        "happiness": 1,
        "sadness": 1,
        "surprise": 4,
    }

    ed_df = load_empathetic_dialogues(data_dir)
    dd_df = load_dailydialog(data_dir)

    data_multi = build_multilingual_dataset(ed_df, dd_df)
    dataset_multi_path = data_dir / "dataset_multilingual_emotion.csv"
    data_multi.to_csv(dataset_multi_path, index=False)
    print(f"Saved multilingual dataset to: {dataset_multi_path}")

    train_df, val_df, test_df = split_train_val_test(
        data_multi,
        num_samples_per_class_test=args.num_test,
        num_samples_per_class_val=args.num_val,
        random_state=args.seed,
    )

    print_distribution(train_df, "Train (before balance)")
    print_distribution(val_df, "Validation")
    print_distribution(test_df, "Test")

    train_df = balance_train_dataset(
        train_df,
        neutral_frac=args.neutral_frac,
        upsample_ratios=upsample_ratios,
        random_state=args.seed,
    )

    print_distribution(train_df, "Train (after balance)")

    train_path = data_dir / "train_dataset.csv"
    val_path = data_dir / "val_dataset.csv"
    test_path = data_dir / "test_dataset.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train split to: {train_path}")
    print(f"Saved validation split to: {val_path}")
    print(f"Saved test split to: {test_path}")


if __name__ == "__main__":
    main()
