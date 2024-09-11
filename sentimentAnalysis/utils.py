import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Define the path to the saved model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'fine_tuned_bert_model')

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

# Set the model to evaluation mode
model.eval()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move tensors to the configured device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class
