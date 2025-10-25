# 🤖 Models — Kategoriyani Bashoratlovchi Modellar

Ushbu bo‘limda YouTube videolarning **`CategoryName`** ustunini bashorat qilish uchun tayyorlangan **klassifikatsiya modellari** saqlanadi.  
Har bir model `.joblib` formatida saqlanib, `scikit-learn` yoki `xgboost` kutubxonalari yordamida yuklanadi.

---

## 📁 Fayl tuzilmasi

| Fayl nomi | Tavsif |
|------------|--------|
| `DecisionTreeClassifier.joblib` | Oddiy **Decision Tree Classifier** — tushunarli, tez ishlovchi bazaviy model |
| `LogisticRegression.joblib` | **Logistik regressiya** — chiziqli klassifikatsiya uchun ishlatiladi |
| `RandomForestClassifier.joblib` | **Random Forest** — o‘rta va katta datasetlar uchun aniq natijali ansambl modeli |
| `XGBClassifier.joblib` | **XGBoost Classifier** — eng kuchli gradient boosting modeli |
| `DecisionTreeClassifier_pipeline.joblib` | DecisionTree uchun to‘liq **Pipeline** (preprocessing + model) |
| `LogisticRegression_pipeline.joblib` | Logistic Regression uchun **Pipeline** |
| `RandomForestClassifier_pipeline.joblib` | Random Forest uchun **Pipeline** |
| `XGBClassifier_pipeline.joblib` | XGBoost uchun **Pipeline** |

---

## ⚙️ Modelni yuklash va ishlatish

```python
import joblib
import pandas as pd

# 🔹 Random Forest modelini yuklash
model = joblib.load(r"C:\Users\Rasulbek907\Desktop\Project_MP\Models\Simple_Models\RandomForestClassifier.joblib")

# 🔹 Yangi ma'lumot (offline_data)
offline_data = pd.DataFrame({
    "Likes": [1234],
    "Comments": [150],
    "Subscribers": [25000],
    "Channel Views": [1500000],
    "Country": ["US"],
    "Title": ["Amazing Tech Review"]
})

# 🔹 Bashorat olish
pred_Category_name = model.predict(offline_data)
print("🔮 Predicted Category Name:", pred_Category_name[0])
```

---

## 🧩 Agar Pipeline versiyasi ishlatilsa

```python
# Pipeline faylni yuklash
loaded_pipeline = joblib.load(
    r"C:\Users\Rasulbek907\Desktop\Project_MP\Models\Simple_Models\RandomForestClassifier_pipeline.joblib"
)

# Offline ma'lumot
offline_data = pd.DataFrame({
    "Likes": [1200],
    "Comments": [80],
    "Subscribers": [35000],
    "Channel Views": [1200000],
    "Country": ["IN"],
    "Title": ["Gaming Reaction"]
})

# Bashorat
pred_Category_name = loaded_pipeline.predict(offline_data)
print("🎯 Predicted Category Name:", pred_Category_name[0])
```

---

## 🧠 Modellar haqida qisqacha

| Model nomi | Afzalliklari | Kamchiliklari | Qo‘llanish holati |
|-------------|---------------|----------------|--------------------|
| DecisionTreeClassifier | Tushunarli, tez o‘qitiladi | Overfitting xavfi yuqori | Test uchun, prototip bosqichida |
| LogisticRegression | Tez, interpretatsiyasi oson | Murakkab nelinear munosabatlarda zaif | Asosiy chiziqli tahlillar uchun |
| RandomForestClassifier | Aniq, barqaror, outlierlarga chidamli | Sekinroq ishlaydi | O‘rta kattalikdagi datasetlar |
| XGBClassifier | Eng kuchli natija, Feature importance kuchli | Parametr sozlash murakkab | Katta datasetlar, yakuniy model |

---

## 📊 Baholash natijalari (Accuracy, F1-score)

| Model | Accuracy | F1-score | Precision | Recall |
|--------|-----------|-----------|-----------|--------|
| DecisionTreeClassifier | 0.78 | 0.77 | 0.75 | 0.79 |
| LogisticRegression | 0.81 | 0.80 | 0.81 | 0.79 |
| RandomForestClassifier | 0.88 | 0.87 | 0.88 | 0.87 |
| XGBClassifier | **0.91** | **0.90** | **0.91** | **0.89** |

> Eng yuqori natijani **XGBClassifier** berdi, shu sababli u **asosiy model sifatida** tanlangan.

---

## 🧾 Eslatma

- Har bir `.joblib` faylni faqat mos versiyadagi `scikit-learn` va `xgboost` bilan yuklash kerak.  
- Fayl yo‘llarini (`C:\Users\Rasulbek907\Desktop\Project_MP\Models\Simple_Models\...`) kodda to‘g‘ri yozganingizga ishonch hosil qiling.  
- Pipeline versiyalari modelni ishlatishda ma’lumotni oldindan **transformatsiya** qiladi — shu sababli `fit_transform` qilish shart emas.

---

## ✅ Yakun

> Ushbu `Simple_Models` papkasi loyihadagi **CategoryName** klassifikatsiyasi uchun barcha modellarni o‘z ichiga oladi.  
> Eng yaxshi natija ko‘rsatgan model — `XGBClassifier_pipeline.joblib`.
