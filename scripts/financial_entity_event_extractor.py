import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Set to local-only mode to prevent repo id errors
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Use your path (adjust for your environment)
MODEL_PATH = r"A:\Infosys\models\finbert_improved"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)

# Financial entity extraction function
def extract_entities(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    pred_labels = [model.config.id2label[p.item()] for p in predictions[0]]
    filtered = [(tok, lab) for tok, lab in zip(tokens, pred_labels) if lab != "O"]
    return filtered

# Example usage
financial_text = "Apple Inc. reported Q3 revenue growth and an increase in market cap."
extracted = extract_entities(financial_text)
print("Financial Entities Extracted:")
for token, label in extracted:
    print(f"{token}: {label}")

sample_texts = [
    "Tesla stock price rose after quarterly earnings.",
    "Alphabet saw a drop in market capitalization."
]
for txt in sample_texts:
    print(f"\nEntities from: {txt}")
    print(extract_entities(txt))
