import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
MODEL_DIR = r"A:\Infosys\models\finbert_finetuned"
BASE_MODEL = "yiyanghkust/finbert-tone"

def main():
    print("✅ Script started.")

    # Load dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded. Rows: {len(df)}")
    except Exception as e:
        print(f"❌ Data file error: {e}")
        return

    label_map = {label: i for i, label in enumerate(df['label'].unique())}
    rev_label_map = {v: k for k, v in label_map.items()}
    df['label_id'] = df['label'].map(label_map)

    # Prepare test set (last 20%)
    test_size = int(0.2 * len(df))
    test_df = df.tail(test_size)
    test_dataset = Dataset.from_pandas(test_df[['text', 'label_id']].rename(columns={'label_id': 'labels'}))

    print(f"✅ Test set ready, samples: {len(test_dataset)}")

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        print("✅ Tokenizer loaded.")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

    try:
        test_dataset = test_dataset.map(tokenize, batched=True)
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'text'])
        print("✅ Tokenization complete.")
    except Exception as e:
        print(f"❌ Tokenization error: {e}")
        return

    # Model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return

    trainer = Trainer(model=model)
    print("✅ Trainer ready.")

    # Predictions
    try:
        preds = trainer.predict(test_dataset)
        y_true = test_dataset['labels']
        y_pred = preds.predictions.argmax(-1)
        print("✅ Predictions made.")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return

    # Metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

    # Error analysis
    incorrect = []
    for text, gt, pred in zip(test_dataset['text'], y_true, y_pred):
        gt_int = int(gt)
        pred_int = int(pred)
        if gt_int != pred_int:
            incorrect.append((text, rev_label_map[gt_int], rev_label_map[pred_int]))

    error_path = r"A:\Infosys\outputs\eda\fiqa_errors.csv"
    error_df = pd.DataFrame(incorrect, columns=['text', 'true_label', 'pred_label'])
    error_df.to_csv(error_path, index=False)
    print(f"❌ Misclassified samples saved to: {error_path}")

    print("✅ Script finished.")

if __name__ == "__main__":
    main()
