"""Evaluate a trained multilingual emotion classifier on the test split.

This script computes aggregate metrics and can render a confusion matrix figure.
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark import sentence_en, sentence_es


def parse_args() -> argparse.Namespace:
    """Parse evaluation settings from command line."""
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier on test data.")
    parser.add_argument("--model-dir", default="output/model", help="Path to trained model directory.")
    parser.add_argument("--test-file", default="data/test_dataset.csv", help="Path to test CSV file.")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length.")
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Evaluate sentence_es and sentence_en quick sets from benchmark.py.",
    )
    parser.add_argument(
        "--cm-output",
        default="output/model/confusion_matrix.png",
        help="Path to save confusion matrix image.",
    )
    parser.add_argument(
        "--classification-report-output",
        default="output/model/classification_report.txt",
        help="Path to save per-class precision/recall/F1 report.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: str):
    """Load model/tokenizer pair from local artifacts."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_label_names(model) -> list[str]:
    """Get ordered label names from model config."""
    if not getattr(model.config, "id2label", None):
        raise ValueError("Model config does not contain id2label metadata.")

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return [id2label[i] for i in sorted(id2label.keys())]


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> list[str]:
    """Predict labels in mini-batches for efficient test-time inference."""
    label_names = get_label_names(model)
    predictions: list[str] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_ids = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        predictions.extend(label_names[idx] for idx in predicted_ids)

    return predictions


def predict_emotion(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    label_names: list[str],
    max_length: int,
) -> tuple[str, float]:
    """Predict emotion label and confidence score for one sentence."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    predicted_class_id = int(torch.argmax(probabilities, dim=1).item())
    confidence_score = float(probabilities[0, predicted_class_id].item())
    predicted_label = label_names[predicted_class_id]

    return predicted_label, confidence_score


def run_quick_benchmark(
    examples: Sequence[Sequence[str]],
    name: str,
    model,
    tokenizer,
    device: torch.device,
    label_names: list[str],
    max_length: int,
) -> None:
    """Compute simple accuracy over helper sentence lists."""
    correct = 0
    for target_label, text in examples:
        pred_label, _ = predict_emotion(text, model, tokenizer, device, label_names, max_length)
        if target_label == pred_label:
            correct += 1

    accuracy = 100.0 * correct / len(examples)
    print(f"{name} quick-set accuracy: {accuracy:.2f}% ({correct}/{len(examples)})")


def cm_analysis(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: str,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """Generate and save a confusion matrix heatmap with percentage annotations."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=annot, fmt="", cmap="Oranges")
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved confusion matrix to: {output}")


def main() -> None:
    """Run test set evaluation and report metrics."""
    args = parse_args()

    test_df = pd.read_csv(args.test_file)
    y_true = test_df["label"].tolist()

    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)
    model_label_names = get_label_names(model)
    y_pred = predict_batch(test_df["text"].tolist(), model, tokenizer, device, args.max_length)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    print("Evaluation results")
    print(f"Accuracy : {accuracy * 100:.3f}%")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    labels = [label for label in model_label_names if label in set(y_true)]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    print("\nPer-class classification report")
    print(report)

    report_path = Path(args.classification_report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Saved classification report to: {report_path}")

    cm_analysis(y_true, y_pred, labels, args.cm_output)

    if args.run_benchmark:
        print("\nQuick benchmark results")
        run_quick_benchmark(sentence_es, "Spanish", model, tokenizer, device, model_label_names, args.max_length)
        run_quick_benchmark(sentence_en, "English", model, tokenizer, device, model_label_names, args.max_length)


if __name__ == "__main__":
    main()
