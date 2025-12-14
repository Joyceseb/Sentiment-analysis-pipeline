import torch
import logging
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        device = "cpu"
        logger.info("Using CPU for evaluation.")

        logger.info("Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        model_path = "./final_model"
        logger.info(f"Loading model from {model_path}")

        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,
            )

        logger.info("Tokenizing test dataset...")
        tokenized_test = dataset["test"].map(tokenize_function, batched=True)

        # Optional: small subset for faster evaluation
        eval_dataset = (
            tokenized_test
            .shuffle(seed=42)
            .select(range(500))
        )

        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)

            metrics = {}
            metrics.update(
                accuracy.compute(predictions=predictions, references=labels)
            )
            metrics.update(
                precision.compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )
            )
            metrics.update(
                recall.compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )
            )
            metrics.update(
                f1.compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )
            )

            return metrics

        eval_args = TrainingArguments(
            output_dir="./eval_results",
            per_device_eval_batch_size=4,
            dataloader_num_workers=0,
            use_cpu=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting evaluation...")
        results = trainer.evaluate()

        logger.info("Evaluation results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")

        import json
        import os
        
        # Ensure output directory exists
        if not os.path.exists("./eval_results"):
            os.makedirs("./eval_results")
            
        output_file = "./eval_results/eval_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
