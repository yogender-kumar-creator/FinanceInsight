from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

# Folder where your fine-tuned weights are saved
MODEL_DIR = r"A:\Infosys\models\finbert_finetuned"
BASE_MODEL = "yiyanghkust/finbert-tone"

print("🔹 Loading base FinBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔹 Loading fine-tuned model weights...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Build pipeline for inference
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)

# Example test sentences
texts = [
    "The company reported strong quarterly profits, exceeding expectations.",
    "The firm's revenue fell significantly due to weak market conditions.",
    "The company plans to expand its services to new markets.",
    "Investors were disappointed by the declining sales figures."
]

print("\n🔹 Making predictions...\n")
for t in texts:
    preds = pipe(t)
    print(f"Text: {t}")
    for p in preds[0]:
        print(f"  {p['label']}: {p['score']:.4f}")
    print()
