import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Gerekli NLTK verilerini indir
nltk.download("punkt")
nltk.download("stopwords")

# Türkçe stop words
stop_words = set(stopwords.words("turkish"))

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

# Test verisini yükleme
df_test = pd.read_csv("test.csv", encoding="ISO-8859-9")

# Sütunları kontrol et
print("Sütunlar:", df_test.columns)

# 'comment' sütununu temizle
df_test["Cleaned"] = df_test["comment"].apply(clean_text)

# Modeli ve TF-IDF vectorizer'ı yükle
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Temizlenmiş verileri vektörleştir
X_test = vectorizer.transform(df_test["Cleaned"])

# Tahmin yap
y_pred = model.predict(X_test)

# Gerçek etiketleri ve tahminleri integer'a çevir
y_test = df_test["Label"].astype(int)
y_pred = [int(p) for p in y_pred]

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: {accuracy:.4f}")

# İlk 10 örneği yazdır
for i in range(min(10, len(df_test))):
    print(f"Yorum: {df_test['comment'].iloc[i]}")
    print(f"Gerçek Etiket: {y_test.iloc[i]}")
    print(f"Tahmin: {y_pred[i]}")
    print("-" * 50)
