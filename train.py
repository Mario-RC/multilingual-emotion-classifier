"""Train a multilingual emotion classifier from prepared CSV datasets.

This script is the training equivalent of the notebook's model section and keeps
all model artifacts in a reusable folder for evaluation and inference scripts.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_scheduler,
)


class EmotionDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset wrapper for tokenized text + encoded labels."""

    def __init__(self, encodings: dict[str, list[int]], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class CustomTrainer(Trainer):
    """Trainer with weighted loss and cosine schedule support."""

    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use class weights to reduce the impact of class imbalance.
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int, num_warmup_steps: int | None = None):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.args.learning_rate,
        )

        if num_warmup_steps is None:
            num_warmup_steps = int(0.1 * num_training_steps)

        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        return optimizer, lr_scheduler


def compute_metrics(eval_pred):
    """Compute task metrics from logits and labels."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }


def parse_args() -> argparse.Namespace:
    """Parse training configuration from command line."""
    parser = argparse.ArgumentParser(description="Train multilingual emotion classifier.")
    parser.add_argument("--train-file", default="data/train_dataset.csv", help="Path to train CSV file.")
    parser.add_argument("--val-file", default="data/val_dataset.csv", help="Path to validation CSV file.")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-large", help="HF model checkpoint.")
    parser.add_argument("--output-dir", default="output/model", help="Output model directory.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay value.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability for hidden and attention layers.")
    parser.add_argument("--logging-steps", type=int, default=500, help="Logging frequency in steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Run the full supervised training flow and save artifacts."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)

    # Encode string labels into numeric IDs and persist class order.
    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df["label"])
    val_df["label_id"] = label_encoder.transform(val_df["label"])
    np.save(output_dir / "label_classes.npy", label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tokenize text using fixed max length for compatibility with the notebook approach.
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )

    train_dataset = EmotionDataset(train_encodings, train_df["label_id"].values)
    val_dataset = EmotionDataset(val_encodings, val_df["label_id"].values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = {i: label for i, label in enumerate(label_encoder.classes_.tolist())}
    label2id = {label: i for i, label in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Align with notebook regularization choices.
    model.config.hidden_dropout_prob = args.dropout
    model.config.attention_probs_dropout_prob = args.dropout

    training_args = TrainingArguments(
        output_dir=str(run_dir / "results"),
        run_name=run_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=str(run_dir / "logs"),
        logging_steps=args.logging_steps,
        save_total_limit=2,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
    )

    # Compute inverse-frequency class weights and normalize them.
    label_counts = train_df["label_id"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / label_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    num_training_steps = int(len(train_dataset) / args.batch_size * args.epochs)
    trainer.create_optimizer_and_scheduler(num_training_steps=max(num_training_steps, 1))

    trainer.train()
    eval_results = trainer.evaluate()

    # Save reusable artifacts for evaluate.py and predict.py.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "eval_results.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("Training completed.")
    print(f"Saved model to: {output_dir}")
    print(f"Run artifacts saved to: {run_dir}")
    print(f"Saved validation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
