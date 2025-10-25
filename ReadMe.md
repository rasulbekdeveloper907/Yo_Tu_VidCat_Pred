# 🎬 YouTube Video Category Prediction

## 🧠 Loyiha maqsadi

Ushbu loyiha **YouTube videolari** haqidagi ma'lumotlardan (likes,
comments, subscribers, country, region, views va boshqalar) foydalanib,
**videoning kategoriyasini (`CategoryName`) oldindan aniqlash** uchun
**multi-class classification model** yaratishni maqsad qiladi.

Bu loyiha **data analysis**, **feature engineering**, va **machine
learning** bosqichlarini o'z ichiga oladi.

------------------------------------------------------------------------

## 📚 Mazmun

-   [⚙️ Talablar (Requirements)](#️-talablar-requirements)
-   [📁 Fayl tuzilmasi (Structure)](#-fayl-tuzilmasi-structure)
-   [📊 Ma'lumot (Dataset)](#-malumot-dataset)
-   [🧹 Feature Engineering va
    Preprocessing](#-feature-engineering-va-preprocessing)
-   [🧠 Model arxitekturasi va
    modellar](#-model-arxitekturasi-va-modellar)
-   [📈 Baholash (Evaluation)](#-baholash-evaluation)
-   [🚀 Ishga tushirish (How to run)](#-ishga-tushirish-how-to-run)
-   [🔍 Misol: yangi video uchun
    bashorat](#-misol-yangi-video-uchun-bashorat)
-   [📊 Natijalarni talqin qilish
    (Interpretation)](#-natijalarni-talqin-qilish-interpretation)
-   [🔧 Keyingi yaxshilanishlar (Future
    improvements)](#-keyingi-yaxshilanishlar-future-improvements)
-   [📜 Litsenziya (License)](#-litsenziya-license)

------------------------------------------------------------------------

## ⚙️ Talablar (Requirements)

Python 3.8+ va quyidagi kutubxonalar kerak bo'ladi:

``` bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost plotly
```

Yoki `requirements.txt` orqali:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📁 Fayl tuzilmasi (Structure)

    youtube-category-prediction/
    ├─ data/
    │  └─ youtube_dataset.csv         # Asosiy dataset
    ├─ notebooks/
    │  └─ eda_plotly.ipynb            # EDA va grafik tahlil
    ├─ src/
    │  ├─ preprocess.py               # Ma'lumotni tozalash va tayyorlash
    │  ├─ train.py                    # Modelni o‘qitish va baholash
    │  └─ predict.py                  # Bashorat funksiyasi
    ├─ models/
    │  └─ RandomForestClassifier.joblib
    ├─ requirements.txt
    ├─ README.md
    └─ LICENSE

------------------------------------------------------------------------

## 📊 Ma'lumot (Dataset)

**Ustunlar (columns):**

  -----------------------------------------------------------------------
  Column                                Turi              Tavsif
  ------------------------------------- ----------------- ---------------
  `Video ID`                            object            Videoning
                                                          unikal ID

  `Video Title`                         object            Video nomi

  `Channel`                             object            Kanal nomi

  `Published Date`                      datetime          Video joylangan
                                                          sana

  `Views`                               int64             Ko'rishlar soni

  `Likes`                               int64             Layklar soni

  `Comments`                            int64             Kommentlar soni

  `Subscribers`                         int64             Obunachilar
                                                          soni

  `Channel Views`                       int64             Kanal umumiy
                                                          ko'rishlar soni

  `Country`                             object            Kanal
                                                          joylashgan
                                                          mamlakat

  `Region`                              object            Hudud

  `CategoryID`                          int64             Kategoriya ID

  `CategoryName`                        object            🎯 Target ustun
                                                          --- bashorat
                                                          qilinadigan
                                                          kategoriya
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🧹 Feature Engineering va Preprocessing

Model uchun ishlatilgan asosiy xususiyatlar:

``` python
features = [
    'Views', 'Likes', 'Comments', 'Subscribers', 'Channel Views',
    'Country', 'Region', 'CategoryID',
    'Year', 'Month', 'Day', 'DayOfWeek', 'DayName',
    'Views_per_Sub', 'Engagement', 'Like_per_Sub', 'Comment_per_Sub'
]
target = 'CategoryName'
```

**Qo'shimcha ishlovlar:** - `Published Date` ustunidan `Year`, `Month`,
`Day`, `DayOfWeek`, `DayName` kabi yangi feature'lar chiqarilgan.\
- `Country`, `Region`, `DayName` ustunlariga **Label Encoding /
OneHotEncoding** qo'llanilgan.\
- Skalerlash (`StandardScaler`) sonli ustunlarga tatbiq etilgan.

------------------------------------------------------------------------

## 🧠 Model arxitekturasi va modellar

Quyidagi **klassifikatsiya modellar** sinovdan o'tkazilgan:

  Model                     Accuracy   Precision   Recall   F1 Score
  ------------------------- ---------- ----------- -------- ----------
  **Random Forest**         1.000      1.000       1.000    1.000
  **XGBoost**               1.000      1.000       1.000    1.000
  **Logistic Regression**   0.9994     0.9994      0.9994   0.9994
  **Decision Tree**         1.000      1.000       1.000    1.000

📊 Eng yaxshi natijani **Random Forest** va **XGBoost** modellari
ko'rsatdi.

------------------------------------------------------------------------

## 📈 Baholash (Evaluation)

Model **multi-class classification** uchun quyidagi metrikalar bilan
baholangan:

-   **Accuracy** → To'g'ri bashoratlar ulushi\
-   **Precision** → Har bir kategoriya uchun aniqlik\
-   **Recall** → To'liq qamrov\
-   **F1 Score** → Aniqlik va qamrovning o'rtacha muvozanati

**Vizual natijalar:** - Confusion Matrix\
- Classification Report\
- Model Comparison Table (yuqoridagi jadval)

------------------------------------------------------------------------

## 🚀 Ishga tushirish (How to run)

``` bash
# 1. Datasetni joylashtiring
/data/youtube_dataset.csv

# 2. Modelni o‘qitish
python src/train.py

# 3. Bashorat qilish
python src/predict.py
```

------------------------------------------------------------------------

## 🔍 Misol: yangi video uchun bashorat

``` python
import joblib
import pandas as pd

loaded_pipeline = joblib.load("models/RandomForestClassifier.joblib")

new_video = pd.DataFrame({
    'Views': [120000],
    'Likes': [3500],
    'Comments': [500],
    'Subscribers': [200000],
    'Channel Views': [1500000],
    'Country': ['US'],
    'Region': ['North America'],
    'CategoryID': [24],
    'Year': [2025],
    'Month': [10],
    'Day': [25],
    'DayOfWeek': [5],
    'DayName': ['Saturday'],
    'Views_per_Sub': [0.6],
    'Engagement': [0.08],
    'Like_per_Sub': [0.017],
    'Comment_per_Sub': [0.002]
})

pred = loaded_pipeline.predict(new_video)
print("🔮 Predicted Category Name:", pred[0])
```

------------------------------------------------------------------------

## 📊 Natijalarni talqin qilish (Interpretation)

-   **Model aniqligi juda yuqori (99.9--100%)**, bu ma'lumotlar yaxshi
    balanslangan yoki kuchli feature engineering qo'llanganini
    ko'rsatadi.\
-   **Eng muhim omillar**: `CategoryID`, `Country`, `Region`,
    `Engagement`, `Views_per_Sub`.\
-   **Random Forest** va **XGBoost** modellarining natijalari mutlaqo
    mukammal bo'lib chiqdi.

------------------------------------------------------------------------

## 🔧 Keyingi yaxshilanishlar (Future improvements)

✅ Modelni real-time bashorat uchun optimallashtirish\
✅ Imbalanced data uchun class-weight balanslash\
✅ NLP orqali video sarlavhasidan (title) semantik feature'lar olish\
✅ SHAP / LIME yordamida feature importance vizualizatsiyasi\
✅ Streamlit / Dash orqali interaktiv dashboard yaratish

------------------------------------------------------------------------

## 📜 Litsenziya (License)

Ushbu loyiha **MIT License** asosida tarqatiladi.\
Kod va hujjatlarni erkin o'zgartirish, qayta ishlatish va ulashish
mumkin.

------------------------------------------------------------------------

## ✨ Yakun

> Ushbu loyiha YouTube videolarining **kategoriya turini aniqlash**
> orqali: - Kontent tahlilini yaxshilaydi,\
> - Trendni oldindan aniqlaydi,\
> - Kanal strategiyasini avtomatlashtirishga yordam beradi.

💡 Maqsad --- **ma'lumot asosida kontent yo'nalishini aniqlashni
avtomatlashtirish.**
