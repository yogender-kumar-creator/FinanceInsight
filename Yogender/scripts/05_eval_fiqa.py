import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix

# Set your paths (update if needed)
DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
MODEL_DIR = r"A:\Infosys\models\finbert_finetuned"
BASE_MODEL = "yiyanghkust/finbert-tone"

# Load the labeled data
df = pd.read_csv(DATA_PATH)
label_map = {label: i for i, label in enumerate(df['label'].unique())}
rev_label_map = {v: k for k, v in label_map.items()}
df['label_id'] = df['label'].map(label_map)

# Create test split (last 20%)
test_size = int(0.2 * len(df))
test_df = df.tail(test_size).copy()  # copy for safe access

# Make huggingface Dataset
test_dataset = Dataset.from_pandas(test_df[['text','label_id']].rename(columns={'label_id':'labels'}))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load your finetuned model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
trainer = Trainer(model=model)

# Predict
preds = trainer.predict(test_dataset)
y_true = test_df['label_id'].to_list()
y_pred = preds.predictions.argmax(axis=-1)

# Metrics
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))
print("--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))

# Error analysis (using test_df['text'] for error-free reporting)
incorrect = []
for text, gt, pred in zip(test_df['text'], y_true, y_pred):
    if gt != pred:
        incorrect.append((text, rev_label_map[gt], rev_label_map[pred]))
error_path = r"A:\Infosys\outputs\eda\fiqa_errors.csv"
pd.DataFrame(incorrect, columns=['text', 'true_label', 'pred_label']).to_csv(error_path, index=False)
print(f"❌ Misclassified samples saved to: {error_path}")
