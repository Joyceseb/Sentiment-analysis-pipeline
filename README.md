# Introduction

With this repository, you will be able to build and train a complete Sentiment Analysis Pipeline using BERT.

The pipeline allows you to load, clean, and process raw text data, then fine-tune a pretrained BERT model for sentiment classification.

You will also be able to deploy the trained model for inference and perform online sentiment predictions on new text inputs and validate everything with unit tests. 


You can explore the main components of the pipeline:
Data Extraction → Load and validate raw data with robust error handling. (data_extraction.py)

Data Processing → Clean, normalize, and tokenize text with Hugging Face AutoTokenizer. (data_processing.py)

Model Training → Fine-tune bert-base-uncased for binary sentiment classification and save the best model. (model.py)

Inference → Use the saved model to predict sentiment on custom inputs. (see README example)

Testing → Unit tests covering data loading, preprocessing, and model forward pass. (tests/test_data_extraction.py, tests/test_data_processing.py, tests/test_model.py)

# Project Structure 

<img width="164" height="149" alt="image" src="https://github.com/user-attachments/assets/c2243090-0aec-4737-bf1b-81728814e8fb" />

