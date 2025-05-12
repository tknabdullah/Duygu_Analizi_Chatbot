import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("punkt")

# Temizleme fonksiyonu 
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    return " ".join(tokens)

# Veriyi yükle
df = pd.read_csv("train.csv", header=None, names=["Comment", "Label"], encoding="ISO-8859-9")
df["Cleaned"] = df["Comment"].apply(clean_text)

# TF-IDF (unigram + bigram) vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train = vectorizer.fit_transform(df["Cleaned"])
y_train = df["Label"]

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model başarıyla eğitildi ve kaydedildi.")



