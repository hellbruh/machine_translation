import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


SEED = 42
MAX_LENGTH = 128
BASE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
OUTPUTS_DIR = Path("outputs")
MODEL_DIRS = {
    "model_news": OUTPUTS_DIR / "model_news",
    "model_books": OUTPUTS_DIR / "model_books",
    "model_combined": OUTPUTS_DIR / "model_combined",
}
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

DATASET_CONFIGS = {
    "news": {
        "path": "Helsinki-NLP/news_commentary",
        "lang_pair": "en-ru",
        "train_limit": 3000,
        "test_limit": 300,
    },
    "books": {
        "path": "Helsinki-NLP/opus_books",
        "lang_pair": "en-ru",
        "train_limit": 3000,
        "test_limit": 300,
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for model_dir in MODEL_DIRS.values():
        model_dir.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_translation_dataset(dataset_path: str, lang_pair: str):
    last_error = None
    attempts = [
        lambda: load_dataset(dataset_path, lang_pair),
        lambda: load_dataset(dataset_path, name=lang_pair),
        lambda: load_dataset(dataset_path, lang_pair=lang_pair),
    ]
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as error:
            last_error = error
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Could not load dataset {dataset_path} ({lang_pair}): {last_error}")


def get_translation_column_name(dataset: Dataset) -> str:
    for candidate in ("translation", "translations"):
        if candidate in dataset.column_names:
            return candidate
    raise ValueError(f"Translation column not found in columns: {dataset.column_names}")


def normalize_translation_example(example: Dict, translation_column: str) -> Dict[str, str]:
    translation = example[translation_column]
    if isinstance(translation, list):
        translation = translation[0]
    if not isinstance(translation, dict):
        raise ValueError("Unexpected translation field format")
    return {"source_text": translation["en"], "target_text": translation["ru"]}


def build_small_splits(dataset_key: str) -> Tuple[Dataset, Dataset]:
    config = DATASET_CONFIGS[dataset_key]
    dataset_dict = load_translation_dataset(config["path"], config["lang_pair"])

    base_split_name = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    base_dataset = dataset_dict[base_split_name]
    translation_column = get_translation_column_name(base_dataset)

    normalized = base_dataset.map(
        lambda row: normalize_translation_example(row, translation_column),
        remove_columns=base_dataset.column_names,
    )
    normalized = normalized.shuffle(seed=SEED)

    sample_limit = min(len(normalized), config["train_limit"])
    sampled = normalized.select(range(sample_limit))
    split_dataset = sampled.train_test_split(test_size=0.2, seed=SEED)

    test_limit = min(len(split_dataset["test"]), config["test_limit"])
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"].select(range(test_limit))
    return train_dataset, test_dataset


def tokenize_batch(batch: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[List[int]]]:
    model_inputs = tokenizer(
        batch["source_text"],
        max_length=MAX_LENGTH,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_LENGTH,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_tokenized_datasets(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> Tuple[Dataset, Dataset]:
    tokenized_train = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    return tokenized_train, tokenized_eval


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(get_device())
    return model, tokenizer


def translate_texts(
    texts: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
) -> List[str]:
    predictions = []
    model.eval()
    device = get_device()

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **batch,
                max_length=MAX_LENGTH,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    metric = evaluate.load("sacrebleu")
    result = metric.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    return float(result["score"])


def save_predictions(
    model_name: str,
    test_dataset_name: str,
    sources: List[str],
    predictions: List[str],
    references: List[str],
    sample_count: int = 10,
) -> None:
    records = []
    for source, prediction, reference in zip(sources[:sample_count], predictions[:sample_count], references[:sample_count]):
        records.append(
            {
                "source_text": source,
                "prediction": prediction,
                "reference": reference,
            }
        )

    output_path = PREDICTIONS_DIR / f"{model_name}_{test_dataset_name}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def evaluate_model_on_dataset(
    model_name: str,
    model_path: str,
    test_dataset_name: str,
    test_dataset: Dataset,
) -> Dict[str, object]:
    model, tokenizer = load_model_and_tokenizer(model_path)
    sources = test_dataset["source_text"]
    references = test_dataset["target_text"]
    predictions = translate_texts(sources, model, tokenizer)
    bleu = compute_bleu(predictions, references)
    save_predictions(model_name, test_dataset_name, sources, predictions, references)
    return {
        "model_name": model_name,
        "test_dataset": test_dataset_name,
        "bleu": round(bleu, 4),
    }


def train_model(output_dir: Path, train_dataset: Dataset, eval_dataset: Dataset) -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    tokenized_train, tokenized_eval = prepare_tokenized_datasets(train_dataset, eval_dataset, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def run_baseline(sample_size: int = 100) -> None:
    _, news_test = build_small_splits("news")
    sample_size = min(sample_size, len(news_test))
    news_sample = news_test.select(range(sample_size))
    result = evaluate_model_on_dataset("baseline", BASE_MODEL_NAME, "news_test", news_sample)

    print("Baseline BLEU on news_test:")
    print(f"BLEU = {result['bleu']}")


def run_train_news() -> None:
    train_dataset, test_dataset = build_small_splits("news")
    train_model(MODEL_DIRS["model_news"], train_dataset, test_dataset)


def run_train_books() -> None:
    train_dataset, test_dataset = build_small_splits("books")
    train_model(MODEL_DIRS["model_books"], train_dataset, test_dataset)


def run_train_combined() -> None:
    news_train, news_test = build_small_splits("news")
    books_train, books_test = build_small_splits("books")
    combined_train = concatenate_datasets([news_train, books_train]).shuffle(seed=SEED)
    combined_eval = concatenate_datasets([news_test, books_test]).shuffle(seed=SEED)
    train_model(MODEL_DIRS["model_combined"], combined_train, combined_eval)


def resolve_local_model(path: Path) -> str:
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Model directory does not exist: {path}")


def run_evaluate_all() -> None:
    _, news_test = build_small_splits("news")
    _, books_test = build_small_splits("books")

    local_combined_candidates = [
        MODEL_DIRS["model_combined"],
        OUTPUTS_DIR / "model_conbined",
    ]
    combined_path = None
    for candidate in local_combined_candidates:
        if candidate.exists():
            combined_path = str(candidate)
            break
    if combined_path is None:
        raise FileNotFoundError(
            "Combined model not found. Run `python main.py train_combined` first."
        )

    model_paths = {
        "baseline": BASE_MODEL_NAME,
        "news-finetuned": resolve_local_model(MODEL_DIRS["model_news"]),
        "books-finetuned": resolve_local_model(MODEL_DIRS["model_books"]),
        "model-finetuned": combined_path,
    }

    test_datasets = {
        "news_test": news_test,
        "books_test": books_test,
    }

    results = []
    for model_name, model_path in model_paths.items():
        for test_dataset_name, test_dataset in test_datasets.items():
            results.append(
                evaluate_model_on_dataset(
                    model_name=model_name,
                    model_path=model_path,
                    test_dataset_name=test_dataset_name,
                    test_dataset=test_dataset,
                )
            )

    results_df = pd.DataFrame(results, columns=["model_name", "test_dataset", "bleu"])
    results_df.to_csv(METRICS_DIR / "results.csv", index=False)
    print(results_df)


def run_translate(text: str, model_path: str) -> None:
    resolved_model_path = model_path
    if model_path != BASE_MODEL_NAME:
        resolved_model_path = str(Path(model_path))

    model, tokenizer = load_model_and_tokenizer(resolved_model_path)
    translated_text = translate_texts([text], model, tokenizer, batch_size=1)[0]
    print(translated_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple EN->RU machine translation project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Evaluate baseline model on a small news sample")
    baseline_parser.add_argument("--sample-size", type=int, default=100)

    subparsers.add_parser("train_news", help="Fine-tune on news_commentary")
    subparsers.add_parser("train_books", help="Fine-tune on opus_books")
    subparsers.add_parser("train_combined", help="Fine-tune on combined datasets")
    subparsers.add_parser("evaluate_all", help="Evaluate all models and save BLEU results")

    translate_parser = subparsers.add_parser("translate", help="Translate one English sentence to Russian")
    translate_parser.add_argument("--text", required=True, help="English text")
    translate_parser.add_argument(
        "--model-path",
        required=True,
        help="Model path or Helsinki-NLP/opus-mt-en-ru",
    )

    return parser


def main() -> None:
    set_seed(SEED)
    ensure_output_dirs()
    args = build_parser().parse_args()

    if args.command == "baseline":
        run_baseline(sample_size=args.sample_size)
    elif args.command == "train_news":
        run_train_news()
    elif args.command == "train_books":
        run_train_books()
    elif args.command == "train_combined":
        run_train_combined()
    elif args.command == "evaluate_all":
        run_evaluate_all()
    elif args.command == "translate":
        run_translate(text=args.text, model_path=args.model_path)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
