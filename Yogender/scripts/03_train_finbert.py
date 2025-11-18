import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from imblearn.over_sampling import RandomOverSampler
import torch
from torch.nn import CrossEntropyLoss

# 1. LOAD AND BALANCE DATA
DATA_PATH = "data/processed/preprocessed_fiqa.csv"
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={'clean_text': 'text'})

label_map = {label: i for i, label in enumerate(df['label'].unique())}
df['label_id'] = df['label'].map(label_map)

# Balance classes
ros = RandomOverSampler()
X_bal, y_bal = ros.fit_resample(df[['text']], df['label_id'])
df_bal = pd.DataFrame({'text': X_bal['text'], 'label_id': y_bal})

# Optional: Simple Text Augmentation
def simple_synonym(text):
    synonyms = {'gain':'profit', 'loss':'decline'}
    return ' '.join([synonyms.get(w, w) for w in text.split()])

df_bal['text_aug'] = df_bal['text'].apply(simple_synonym)
df_aug = pd.concat([
    df_bal[['text','label_id']],
    pd.DataFrame({'text': df_bal['text_aug'], 'label_id': df_bal['label_id']})
])

# 2. SPLIT DATA
train_df, test_df = train_test_split(df_aug, test_size=0.2, stratify=df_aug['label_id'], random_state=42)
train_dataset = Dataset.from_pandas(train_df[['text','label_id']].rename(columns={'label_id':'labels'}))
test_dataset = Dataset.from_pandas(test_df[['text','label_id']].rename(columns={'label_id':'labels'}))

# 3. TOKENIZE
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 4. CLASS WEIGHTS
label_counts = train_df['label_id'].value_counts()
weights = torch.tensor([1.0/label_counts[i] for i in range(len(label_map))], dtype=torch.float32)
q

# 5. DEFINE CUSTOM TRAINER TO OVERRIDE LOSS
from transformers import Trainer
from torch.nn import CrossEntropyLoss

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = CrossEntropyLoss(weight=weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# 6. TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="models/finbert_improved",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    num_train_epochs=7,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs_improved",
    logging_steps=25,
)

# 7. MODEL
model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone",
    num_labels=len(label_map)
)

# 8. TRAIN
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()
trainer.save_model(training_args.output_dir)

# 9. EVALUATE
outputs = trainer.predict(test_dataset)
y_true = test_dataset['labels']
y_pred = outputs.predictions.argmax(axis=-1)

print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))
print(confusion_matrix(y_true, y_pred))

# 10. ERROR ANALYSIS
rev_label_map = {v: k for k, v in label_map.items()}
incorrect = []
for text, gt, pred in zip(test_dataset['text'], y_true, y_pred):
    if gt != pred:
        incorrect.append((text, rev_label_map[int(gt)], rev_label_map[int(pred)]))
pd.DataFrame(incorrect, columns=['text','true_label','pred_label']).to_csv('outputs/fiqa_errors_improved.csv', index=False)
