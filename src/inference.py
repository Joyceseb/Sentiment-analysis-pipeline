import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os
# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Default model path ---
MODEL_PATH = "./final_model"


def load_model(model_path=MODEL_PATH):
    """
    ✅ Load fine-tuned model from saved path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")

    logger.info(f"Loading model and tokenizer from {model_path} ...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer


def tokenize_input(texts, tokenizer):
    """
    ✅ Tokenize new text inputs.
    """
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )


def run_inference(model, inputs):
    """
    ✅ Run model inference (forward pass without gradient computation).
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return logits


def decode_predictions(logits):
    """
    ✅ Decode logits into sentiment labels.
    """
    predictions = torch.argmax(logits, dim=1).tolist()
    label_map = {0: "Negative 😡", 1: "Positive 😊"}
    decoded = [label_map[p] for p in predictions]
    return decoded


def predict(texts, model, tokenizer):
    """
    ✅ Full prediction pipeline: tokenize → infer → decode → return result.
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenize_input(texts, tokenizer)
    logits = run_inference(model, inputs)
    results = decode_predictions(logits)
    return results


def main():
    """
    ✅ Entry point: loads model and allows user to test new text input.
    """
    try:
        model, tokenizer = load_model()

        logger.info("Model loaded successfully. Ready for inference!")

        while True:
            text = input("\nEnter text to analyze sentiment (or type 'quit'): ")
            if text.lower() == "quit":
                print("Exiting inference mode.")
                break

            results = predict(text, model, tokenizer)
            print(f"Sentiment: {results[0]}")

    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()