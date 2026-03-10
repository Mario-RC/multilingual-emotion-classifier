# Emotional Classifier

This project trains and evaluates a multilingual emotion classifier (English + Spanish)
using DailyDialog and EmpatheticDialogues-derived CSV files.

Dataset source note:

- The DAILYD and MPATHY CSV resources come from `CHANEL-JSALT-2020/datasets`:
	`https://github.com/CHANEL-JSALT-2020/datasets`

The original notebook workflow (`emotional_classifier.ipynb`) has been split into
four scripts for a cleaner and reproducible pipeline:

- `prepare_data.py`
- `train.py`
- `evaluate.py`
- `predict.py`

Additionally, `benchmark.py` contains curated EN/ES benchmark sentences used by
`evaluate.py --run-benchmark`.

## 1. Requirements

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

Generate multilingual dataset and train/validation/test splits:

```bash
python3 prepare_data.py
```

Useful options:

```bash
python3 prepare_data.py --num-test 250 --num-val 250 --neutral-frac 0.6 --seed 42
```

Outputs in `data/`:

- `dataset_multilingual_emotion.csv`
- `train_dataset.csv`
- `val_dataset.csv`
- `test_dataset.csv`

## 3. Training

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
- timestamped run folders in `runs/run_YYYYMMDD_HHMMSS/` (logs + checkpoints)

## 4. Evaluation

Evaluate the trained model on the test split:

```bash
python3 evaluate.py --model-dir output/model
```

By default, the confusion matrix image is saved to:

- `output/model/confusion_matrix.png`

Run quick benchmark sets from `benchmark.py`:

```bash
python3 evaluate.py --model-dir output/model --run-benchmark
```

## 5. Prediction

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

## 6. Full Pipeline Script

Run the complete workflow with one command:

```bash
bash run_pipeline.sh
```

You can override model settings through environment variables:

```bash
MODEL_NAME=FacebookAI/xlm-roberta-large MODEL_DIR=output/model bash run_pipeline.sh
```

## Notes

- Keep your dataset CSV files under `data/` as expected by defaults, or pass custom paths.
- If training is unstable due to GPU memory, reduce `--batch-size`.
- The scripts are deterministic when using the default seed (`42`) where sampling applies.
