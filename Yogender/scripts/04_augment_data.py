import pandas as pd
from textattack.augmentation import EasyDataAugmenter
from tqdm import tqdm
import os

DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
OUTPUT_PATH = r"A:\Infosys\data\augmented\augmented_fiqa.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df = pd.read_csv(DATA_PATH)
augmenter = EasyDataAugmenter()

augmented_rows = []

print("🔹 Augmenting data (this may take a few minutes)...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    text, label = row['text'], row['label']
    try:
        aug_texts = augmenter.augment(text)
        for t in aug_texts:
            augmented_rows.append((t, label))
    except Exception:
        pass

aug_df = pd.DataFrame(augmented_rows, columns=["text", "label"])
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Augmented dataset saved at: {OUTPUT_PATH}")
print(f"Total samples: {len(final_df)}")
