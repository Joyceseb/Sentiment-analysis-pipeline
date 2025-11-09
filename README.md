# Introduction

With this repository, you will be able to build and train a complete Sentiment Analysis Pipeline using BERT.

The pipeline allows you to load, clean, and process raw text data, then fine-tune a pretrained BERT model for sentiment classification.

You will also be able to deploy the trained model for inference and perform online sentiment predictions on new text inputs and validate everything with unit tests. 

______________________________________________________________________________________________________________

You can explore the main components of the pipeline:

**Data Extraction** → Load and validate raw data with robust error handling. (data_extraction.py)

**Data Processing** → Clean, normalize, and tokenize text with Hugging Face AutoTokenizer. (data_processing.py)

**Model Training** → Fine-tune bert-base-uncased for binary sentiment classification and save the best model. (model.py)

**Inference** → Use the saved model to predict sentiment on custom inputs. (see README example)

**Testing** → Unit tests covering data loading, preprocessing, and model forward pass. (tests/test_data_extraction.py, tests/test_data_processing.py, tests/test_model.py)

______________________________________________________________________________________________________________

# Project Structure 

<img width="164" height="149" alt="image" src="https://github.com/user-attachments/assets/c2243090-0aec-4737-bf1b-81728814e8fb" />

______________________________________________________________________________________________________________

# Project Setup

**Step 1 — Clone the Repository**

git clone https://github.com/your-username/sentiment-analysis-pipeline.git

cd sentiment-analysis-pipeline

This will copy the full project locally and prepare the workspace.

______________________________________________________________________________________________________________

**Step 2 — Create a Virtual Environment**

python -m venv venv

source venv/bin/activate   # On Linux/Mac

venv\Scripts\activate      # On Windows

A virtual environment isolates project dependencies and avoids version conflicts.

______________________________________________________________________________________________________________

**Step 3 — Install Dependencies**

pip install -r requirements.txt

These libraries are required for data manipulation, tokenization, model training, and testing

# Step-by-Step Project Workflow

**Step 1 — Data Extraction (data_extraction.py)**

**Goal**: Load and verify raw text data.

- Load CSV files into pandas DataFrames.

- Handle errors such as missing files, empty files, invalid formats, or permission issues.

- Display clear success or error messages to ensure data reliability.

**Example**:

"""
from data_extraction import load_data

df = load_data("dataset.csv")

"""

**Step 2 — Data Processing (data_processing.py)**

**Goal**: Prepare clean and tokenized data for BERT.

- Clean text by removing URLs, mentions, hashtags, and special characters.

- Normalize text by converting to lowercase and adjusting spacing.

- Tokenize text using the Hugging Face AutoTokenizer (bert-base-uncased).

- Split the dataset into training and validation sets (default 80/20).

**Example:**

"""

from data_processing import preprocess_data

train_df, val_df, train_enc, val_enc = preprocess_data(df)

"""

**Step 3 — Model Training (model.py)**

**Goal**: Fine-tune a BERT model on the sentiment dataset.

- Load the IMDB dataset or a custom dataset.

- Use BertForSequenceClassification for binary sentiment classification.

- Define the evaluation metric (accuracy).

- Train the model using the Hugging Face Trainer API.

- Save the fine-tuned model in the ./final_model/ directory.

**Example (run directly):**

"""

python model.py

""" 

**Step 4 — Inference (Prediction)**

**Goal**: Use the trained model to predict sentiment for new text samples.

**Example:**

"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./final_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "This movie was absolutely amazing!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()

print("Sentiment:", "Positive" if prediction == 1 else "Negative")

"""

**Step 5 — Unit Testing (tests/)**

**Goal**: Validate the functionality and robustness of each module.

**Test File	Description**

- test_data_extraction.py	: Tests the CSV loading function, including handling of missing files, invalid formats, and other edge cases.
- 
- test_data_processing.py	: Verifies text cleaning, tokenization, and the correct split between training and validation sets.
- 
- test_model.py	: Checks the proper loading of the model and tokenizer, ensures that the forward pass executes correctly, and verifies the shape of the output logits.
