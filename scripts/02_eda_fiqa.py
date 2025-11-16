# a:\Infosys\scripts\02_eda_fiqa.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# === CONFIG ===
DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
OUTPUT_DIR = r"A:\Infosys\outputs\eda"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
print(df.head())

# --- Label distribution ---
print("\nLabel Distribution:")
print(df['label'].value_counts())

# === Create output folder ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === WordClouds for each sentiment ===
for sentiment in df['label'].unique():
    subset = df[df['label'] == sentiment]
    text_blob = " ".join(subset['text'].dropna().astype(str))
    if len(text_blob.strip()) == 0:
        print(f"[WARN] No text for sentiment: {sentiment}")
        continue
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment}", fontsize=14)
    save_path = os.path.join(OUTPUT_DIR, f"wordcloud_{sentiment}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {save_path}")

# === Basic stats ===
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
stats = df.groupby('label')['text_length'].describe()
print("\nSentence length stats by label:")
print(stats)

# === Plot label distribution ===
plt.figure(figsize=(6,4))
df['label'].value_counts().plot(kind='bar', color=['#4daf4a','#377eb8','#e41a1c'])
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
label_plot_path = os.path.join(OUTPUT_DIR, 'label_distribution.png')
plt.savefig(label_plot_path)
plt.close()
print(f"✅ Saved {label_plot_path}")

print("\n✅ EDA complete. Outputs saved to:", OUTPUT_DIR)
