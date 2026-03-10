"""Run inference with a trained multilingual emotion classifier.

This script supports single-text prediction and batch prediction from a text file.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse inference options from command line."""
    parser = argparse.ArgumentParser(description="Predict emotions from text.")
    parser.add_argument("--model-dir", default="output/model", help="Path to trained model directory.")
    parser.add_argument("--text", action="append", help="Text input. Repeat --text for multiple sentences.")
    parser.add_argument("--text-file", help="Optional file with one sentence per line.")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length.")
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: str):
    """Load tokenizer and model from a local checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_label_names(model, model_dir: str) -> list[str]:
    """Resolve label names from model config or fallback class file."""
    if getattr(model.config, "id2label", None):
        # id2label keys can be strings in serialized configs.
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        return [id2label[i] for i in sorted(id2label.keys())]

    classes_path = Path(model_dir) / "label_classes.npy"
    if classes_path.exists():
        return np.load(classes_path, allow_pickle=True).tolist()

    raise FileNotFoundError(
        "Could not resolve label names. Missing both model.config.id2label and label_classes.npy."
    )


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


def main() -> None:
    """Run prediction for user-provided text."""
    args = parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)
    label_names = get_label_names(model, args.model_dir)

    inputs: list[str] = []
    if args.text:
        inputs.extend(args.text)

    if args.text_file:
        text_file = Path(args.text_file)
        lines = text_file.read_text(encoding="utf-8").splitlines()
        inputs.extend([line.strip() for line in lines if line.strip()])

    for text in inputs:
        predicted_label, confidence = predict_emotion(
            text,
            model,
            tokenizer,
            device,
            label_names,
            args.max_length,
        )
        print(f"Text: {text}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {confidence:.4f}\n")

    if not inputs:
        print("No input provided. Use --text or --text-file.")


if __name__ == "__main__":
    main()
