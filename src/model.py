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
        # Force CPU to avoid MPS memory issues
        device = "cpu"  # Force CPU usage
        logger.info("Using CPU to avoid MPS memory issues")
        
        # Load the dataset
        logger.info("Loading IMDB dataset...")
        dataset = load_dataset("imdb")

        # Load model and tokenizer
        model_name = "bert-base-uncased"
        logger.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Move model to CPU
        model.to(device)

        # Tokenization with shorter sequences to save memory
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=256  # Reduced from 512 to save memory
            )

        logger.info("Tokenizing datasets...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Take even smaller subsets for quick training
        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))  # Reduced from 1000
        small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))   # Reduced from 1000

        # Training arguments with reduced memory footprint
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            per_device_train_batch_size=4,  # Reduced from 8
            per_device_eval_batch_size=4,   # Reduced from 8
            num_train_epochs=2,
            logging_dir="./logs",
            logging_steps=20,
            report_to="none",
            save_total_limit=1,
            dataloader_num_workers=0,  # Disable multiprocessing to save memory
            no_cuda=True,  # Explicitly disable CUDA/MPS
        )

        # Define accuracy metric
        accuracy_metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)

        # Trainer setup
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model("./final_model")
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


