# 🎬 YouTube Video Views Prediction

## 🧠 Loyiha maqsadi
Ushbu loyiha **YouTube videolari** haqidagi ma’lumotlardan (likes, comments, subscribers, channel views, category, country va boshqalar) foydalanib, **yangi videoning taxminiy ko‘rishlar sonini (`Views`) oldindan aytish** uchun regression model yaratishni maqsad qiladi.  

Bu loyiha **data analysis**, **feature engineering**, va **machine learning** bosqichlarini o‘z ichiga oladi.

---

## 📚 Mazmun

- [⚙️ Talablar (Requirements)](#️-talablar-requirements)
- [📁 Fayl tuzilmasi (Structure)](#-fayl-tuzilmasi-structure)
- [📊 Ma'lumot (Dataset)](#-malumot-dataset)
- [🧹 Feature Engineering va Preprocessing](#-feature-engineering-va-preprocessing)
- [🧠 Model arxitekturasi va modellar](#-model-arxitekturasi-va-modellar)
- [📈 Baholash (Evaluation)](#-baholash-evaluation)
- [🚀 Ishga tushirish (How to run)](#-ishga-tushirish-how-to-run)
- [🔍 Misol: yangi video uchun bashorat](#-misol-yangi-video-uchun-bashorat)
- [📊 Natijalarni talqin qilish (Interpretation)](#-natijalarni-talqin-qilish-interpretation)
- [🔧 Keyingi yaxshilanishlar (Future improvements)](#-keyingi-yaxshilanishlar-future-improvements)
- [📜 Litsenziya (License)](#-litsenziya-license)

---

## ⚙️ Talablar (Requirements)

Python 3.8+ va quyidagi kutubxonalar kerak bo‘ladi:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost plotly
```

Yoki `requirements.txt` orqali:

```bash
pip install -r requirements.txt
```

---

## 📁 Fayl tuzilmasi (Structure)

```
video-views-prediction/
├─ data/
│  └─ youtube_dataset.csv         # Asosiy dataset (foydalanuvchi joylashtiradi)
├─ notebooks/
│  └─ eda_plotly.ipynb            # EDA va grafiklar uchun Jupyter Notebook
├─ src/
│  ├─ preprocess.py               # Ma'lumotni tozalash va tayyorlash funksiyalari
│  ├─ train.py                    # Modelni o‘qitish va baholash
│  └─ predict.py                  # Bashorat qilish funksiyalari
├─ models/
│  └─ rf_views_model.pkl          # Saqlangan model fayli
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 📊 Ma'lumot (Dataset)

**Ustunlar (columns):**

| Column          | Turi    | Tavsif |
|-----------------|----------|---------|
| `Video ID`        | object | Videoning unikal identifikatori |
| `Video Title`     | object | Video nomi |
| `Channel`         | object | Kanal nomi |
| `Published Date`  | object | Video joylangan sana |
| `Views`           | int64  | Ko‘rishlar soni |
| `Likes`           | int64  | Layklar soni |
| `Comments`        | int64  | Kommentlar soni |
| `Subscribers`     | int64  | Kanal obunachilari soni |
| `Channel Views`   | int64  | Kanal umumiy ko‘rishlar soni |
| `Country`         | object | Kanal joylashgan mamlakat |
| `Region`          | object | Hudud |
| `CategoryID`      | int64  | Kategoriya ID raqami |
| `CategoryName`    | object | Kategoriya nomi |

**Eslatma:**  
- Sana (`Published Date`) ustuni `datetime` formatiga o‘tkazilishi tavsiya etiladi.  
- Sonli ustunlarda (`Views`, `Likes`, `Comments`, `Subscribers`) qiymatlar `int` tipida bo‘lishi kerak.

---

## 🧹 Feature Engineering va Preprocessing

Modelga kiritiladigan asosiy xususiyatlar:

```python
features = [
    'Likes',
    'Comments',
    'Subscribers',
    'Channel Views',
    'CategoryName',
    'Country'
]
target = 'Views'
```

**Qo‘shimcha ishlovlar:**
- Kategorik ustunlar (`CategoryName`, `Country`) uchun `Label Encoding` yoki `OneHotEncoding`.
- Sana ustunidan:
  - `Year`, `Month`, `Day`, `Weekday` kabi qo‘shimcha feature’lar hosil qilish.
- Skalerlash (`StandardScaler` yoki `MinMaxScaler`).

---

## 🧠 Model arxitekturasi va modellar

Loyihada quyidagi modellardan foydalanish mumkin:

| Model | Turi | Tavsif |
|--------|------|--------|
| `LinearRegression` | Baseline | Oddiy chiziqli regressiya |
| `RandomForestRegressor` | Ensemble | Murakkab, lekin kuchli model |
| `XGBoostRegressor` | Boosting | Yuqori aniqlik va tezlik |
| `CatBoost` / `LightGBM` | Advanced | Kategorik ma’lumotlarga moslashgan ilg‘or modelllar |

---

## 📈 Baholash (Evaluation)

Regression modellari uchun quyidagi metrikalardan foydalaniladi:

| Metrika | Tavsif |
|----------|---------|
| **R² (R-squared)** | Model ma’lumotlardagi dispersiyani qanchalik tushuntira oladi |
| **MAE (Mean Absolute Error)** | O‘rtacha mutlaq xatolik |
| **RMSE (Root Mean Squared Error)** | Xatoliklar dispersiyasini baholaydi |

📊 **Vizualizatsiyalar:**
- Actual vs Predicted scatter plot  
- Residuals histogram  
- Feature importance chart

---

## 🚀 Ishga tushirish (How to run)

```bash
# 1. Datasetni joylashtiring
/data/youtube_dataset.csv

# 2. Modelni o‘qitish
python src/train.py

# 3. Bashorat qilish
python src/predict.py
```

---

## 🔍 Misol: yangi video uchun bashorat

```python
from predict import predict_views

new_video = {
    "Likes": 12000,
    "Comments": 350,
    "Subscribers": 550000,
    "Channel Views": 20000000,
    "CategoryName": "Entertainment",
    "Country": "US"
}

predicted_views = predict_views(new_video)
print(f"Taxminiy ko‘rishlar soni: {predicted_views}")
```

---

## 📊 Natijalarni talqin qilish (Interpretation)

- **R²** → Model ko‘rishlar sonining o‘zgarishini qanchalik tushuntira olishini bildiradi.  
- **Feature importance** → Qaysi omillar eng muhimligini ko‘rsatadi (masalan, `Subscribers` va `Likes` odatda eng muhim).  
- **MAE / RMSE** → Modelning o‘rtacha xato darajasi.

---

## 🔧 Keyingi yaxshilanishlar (Future improvements)

✅ Video title’dan NLP orqali TF-IDF yoki embedding feature’lar olish  
✅ Sana bo‘yicha trend/seasonal komponentlarni qo‘shish  
✅ Hyperparameter tuning (`GridSearchCV`, `Optuna`)  
✅ Model stacking va ensemble yondashuvlarini sinash  
✅ Interpretability uchun SHAP yoki LIME grafiklari  
✅ Web-dashboard (Streamlit, Dash) orqali natijalarni ko‘rsatish

---

## 📜 Litsenziya (License)

Ushbu loyiha **MIT License** asosida tarqatiladi.  
Kod va hujjatlarni erkin o‘zgartirish, qayta ishlatish va ulashish mumkin.

---

## ✨ Yakun

> Ushbu loyiha YouTube videolarining ko‘rishlar sonini bashorat qilish orqali:
> - Kontent strategiyasini tahlil qilish,
> - Kanal o‘sishini rejalashtirish,
> - Trendlarni oldindan ko‘ra olish imkonini beradi.  

💡 Maqsad — **ma’lumot asosida qaror qabul qilishni osonlashtirish.**
