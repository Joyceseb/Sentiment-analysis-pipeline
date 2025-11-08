from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_metric
import numpy as np


def main():
    # 1️⃣ Load pretrained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2️⃣ Load and preprocess dataset (example: IMDb)
    dataset = load_dataset("imdb")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 3️⃣ Define evaluation metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 4️⃣ Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
    )

    # 5️⃣ Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),  # smaller subset for demo
        eval_dataset=encoded_dataset["test"].shuffle(seed=42).select(range(500)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6️⃣ Train (fine-tune) the model
    trainer.train()

    # 7️⃣ Evaluate on validation/test set
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # 8️⃣ Save the fine-tuned model
    save_path = "./models/bert_sentiment"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
