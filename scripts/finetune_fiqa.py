from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
BASE_MODEL = "yiyanghkust/finbert-tone"

# Load data
df = pd.read_csv(DATA_PATH)
label_map = {label: i for i, label in enumerate(df['label'].unique())}
df['label_id'] = df['label'].map(label_map)
dataset = Dataset.from_pandas(df[['text', 'label_id']].rename(columns={"label_id": "labels"}))

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Train/test split
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Training args - tune to improve accuracy
training_args = TrainingArguments(
    output_dir="models/finbert_finetuned",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
)


model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(label_map))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Starting fine-tuning FinBERT...")
trainer.train()
trainer.save_model(training_args.output_dir)
print("✅ Fine-tuning complete and saved.")
