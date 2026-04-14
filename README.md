# Emotional Classifier

This project trains and evaluates a multilingual emotion classifier (English + Spanish)
using DailyDialog and EmpatheticDialogues-derived CSV files.

Dataset source:

- The DAILYD and MPATHY CSV resources come from `CHANEL-JSALT-2020/datasets`:
	`https://github.com/CHANEL-JSALT-2020/datasets`

The original notebook workflow (`emotional_classifier.ipynb`) has been split into
four scripts for a cleaner and reproducible pipeline:

- `prepare_data.py`
- `train.py`
- `evaluate.py`
- `predict.py`

Additionally, `benchmark.py` contains curated EN/ES benchmark sentences used by
`evaluate.py --run-benchmark` and `predict.py --run-benchmark`.

The notebook `emotional_classifier.ipynb` remains fully functional and can still be used.

## 1. Requirements

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

## 2. Hyperparameter Config

The pipeline reads defaults from:

- `configs/hyperparameters.json`

You can edit this file to set dataset, training, and pipeline hyperparameters
without changing Python scripts.

Use a custom config file path if needed:

```bash
CONFIG_FILE=configs/hyperparameters.json bash run_pipeline.sh
```

Environment variables still override JSON values when explicitly provided.

## 3. Data Preparation

Generate multilingual dataset and train/validation/test splits:

```bash
python3 prepare_data.py
```

Useful options:

```bash
python3 prepare_data.py --num-test 250 --num-val 250 --neutral-frac 0.6 --seed 42
```

Save distribution plots:

```bash
python3 prepare_data.py --save-plots
```

Outputs in `data/`:

- `dataset_multilingual_emotion.csv`
- `dataset_multi.csv` (legacy notebook-compatible filename)
- `train_dataset.csv`
- `val_dataset.csv`
- `test_dataset.csv`

Optional plot outputs in `data/plots/` when `--save-plots` is enabled.

## 4. Training

Train the classifier from prepared splits:

```bash
python3 train.py
```

Example with explicit model and output path:

```bash
python3 train.py \
	--model-name FacebookAI/xlm-roberta-large \
	--output-dir output/model \
	--batch-size 32 \
	--epochs 3 \
	--learning-rate 5e-6
```

Main artifacts saved under `--output-dir`:

- model weights and config
- tokenizer files
- `label_classes.npy`
- `eval_results.json`
- `classification_report.txt` (written by evaluation)
- timestamped run folders in `runs/run_YYYYMMDD_HHMMSS/` (logs + checkpoints)
- `runs/run_YYYYMMDD_HHMMSS/training_metrics.png`

Resume from latest checkpoint in `--output-dir/runs`:

```bash
python3 train.py --output-dir output/model --resume-from-checkpoint
```

## 5. Evaluation

Evaluate the trained model on the test split:

```bash
python3 evaluate.py --model-dir output/model
```

By default, the confusion matrix image is saved to:

- `output/model/confusion_matrix.png`

Per-class metrics report is saved to:

- `output/model/classification_report.txt`

Run quick benchmark sets from `benchmark.py`:

```bash
python3 evaluate.py --model-dir output/model --run-benchmark
```

## 6. Prediction

Predict one or multiple texts:

```bash
python3 predict.py --model-dir output/model --text "Hola, ¿qué tal estás?"
```

Multiple texts:

```bash
python3 predict.py \
	--model-dir output/model \
	--text "I feel great today." \
	--text "Me siento genial hoy." \
	--text "I am worried about tomorrow." \
    --text "Estoy preocupado por el mañana."
```

From a file (one sentence per line):

```bash
python3 predict.py --model-dir output/model --text-file input_sentences.txt
```

Run benchmark quick sets with prediction path:

```bash
python3 predict.py --model-dir output/model --run-benchmark
```

## 7. Full Pipeline Script

Run the complete workflow with one command:

```bash
bash run_pipeline.sh
```

The pipeline executes all runtime scripts in order:

- `prepare_data.py`
- `train.py`
- `evaluate.py`
- `evaluate.py --run-benchmark`
- `predict.py --run-benchmark`

You can override model settings through environment variables:

```bash
MODEL_NAME=FacebookAI/xlm-roberta-large MODEL_DIR=output/model bash run_pipeline.sh
```

Additional environment variables supported by `run_pipeline.sh`:

- `BATCH_SIZE` (default `32`)
- `EPOCHS` (default `3.0`)
- `LEARNING_RATE` (default `5e-6`)
- `DROPOUT` (default `0.2`)
- `SEED` (default `42`)
- `MAX_LENGTH` (default `128`)
- `SAVE_PLOTS` (`1`/`0`, default `1`)
- `RESUME_FROM_CHECKPOINT` (`1`/`0`, default `0`)
