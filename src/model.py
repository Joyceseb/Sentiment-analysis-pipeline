import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Force CPU
        device = "cpu"
        logger.info("Using CPU to avoid MPS/CUDA issues.")

        # Load dataset
        logger.info("Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        # Load pre-trained model + tokenizer
        model_name = "bert-base-uncased"
        logger.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Move model to CPU
        model.to(device)

        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )

        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Smaller subsets for fast training
        small_train_dataset = (
            tokenized_datasets["train"]
            .shuffle(seed=42)
            .select(range(500))
        )
        small_eval_dataset = (
            tokenized_datasets["test"]
            .shuffle(seed=42)
            .select(range(200))
        )

        # Training parameters
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",        # FIXED
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            logging_dir="./logs",
            logging_steps=20,
            report_to="none",
            save_total_limit=1,
            dataloader_num_workers=0,
            no_cuda=True,
        )


        # Metric
        accuracy_metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished.")

        # --- Ensure model directory exists ---
        if not os.path.exists("./final_model"):
            os.makedirs("./final_model")

        # --- Force save model + tokenizer ---
        logger.info("Saving model to ./final_model ...")
        trainer.model.save_pretrained("./final_model")
        tokenizer.save_pretrained("./final_model")

        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
