# Duygu_Analizi_Chatbot
İSTE Python Projesi
Bu proje, Türkçe film yorumları üzerinden pozitif veya negatif duygu analizi yapan bir chatbot uygulamasıdır. Makine öğrenmesi ve doğal dil işleme (NLP) teknikleri kullanılarak geliştirilmiştir.

Proje Özellikleri
Türkçe yorumlardan duygu analizi yapar (Pozitif / Negatif)

TF-IDF (unigram + bigram) tabanlı metin vektörleştirme

Lojistik Regresyon algoritmasıyla model eğitimi

Tkinter ile kullanıcı dostu masaüstü arayüz

Eğitilen model .pkl formatında kaydedilip arayüzde kullanılır

Kullanılan Teknolojiler ve Kütüphaneler
Python 3.x

scikit-learn

pandas

nltk

tkinter

pickle

.
├── train.csv           # Eğitim verisi (yorum, etiket)
├── test.csv            # Test verisi
├── train.py            # Model eğitimi ve kaydetme
├── test.py             # Model testi ve doğruluk analizi
├── chatbot.py          # Ana uygulama (GUI ve chatbot)
├── model.pkl           # Eğitilmiş model ve TF-IDF vectorizer
└── README.md           # Proje açıklaması

