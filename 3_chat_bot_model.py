import tkinter as tk
from tkinter import messagebox
import nltk
import pickle
import string
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    return " ".join(tokens)

# Model ve vectorizer'ı yükle
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Tahmin fonksiyonu
def analyze_sentiment():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Uyarı", "Lütfen bir yorum girin.")
        return

    cleaned = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned])
    prediction = model.predict(input_vector)
    proba = model.predict_proba(input_vector)
    confidence = max(proba[0])

    if prediction[0] == "1":
        result_label.config(text=f"%{confidence * 100:.1f} güven ile Pozitif 🎉", fg="green")
    else:
        result_label.config(text=f"%{confidence * 100:.1f} güven ile Negatif 😞", fg="red")

# Arayüz
root = tk.Tk()
root.title("Film Yorumları Duygu Analizi Chatbot")
root.geometry("800x500")
root.configure(bg="#f7f7f7")

# Başlık etiketini ekleyelim
header_label = tk.Label(root, text="Film Yorumları Duygu Analizi", font=("Arial", 18, "bold"), bg="#f7f7f7", fg="#4A90E2")
header_label.pack(pady=20)

# Yorum girişi etiketi
label = tk.Label(root, text="Yorumunuzu girin:", font=("Arial", 12), bg="#f7f7f7", fg="#333")
label.pack(pady=5)

# Yorum girişi kutusu
entry = tk.Entry(root, width=60, font=("Arial", 12), bd=2, relief="solid", justify="center")
entry.pack(pady=10)

# Buton tasarımı
analyze_button = tk.Button(root, text="Tahmin Et", command=analyze_sentiment, font=("Arial", 14, "bold"), bg="#4A90E2", fg="white", relief="raised", width=20)
analyze_button.pack(pady=20)

# Sonuç etiketini ekleyelim
result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f7f7f7")
result_label.pack(pady=20)

# Alt bilgi etiketi
footer_label = tk.Label(root, text="Duygu analizi chatbotu\nFilm yorumlarına dayalı olarak analizinizi yapın.", font=("Arial", 10), bg="#f7f7f7", fg="#888")
footer_label.pack(side="bottom", pady=10)

root.mainloop()




