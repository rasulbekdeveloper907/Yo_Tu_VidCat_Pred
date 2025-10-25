# 🎬 YouTube Video Category Prediction Project

## 📌 Loyihaning qisqacha tavsifi
Ushbu loyiha YouTube videolari haqida to‘plangan ma’lumotlar asosida **video kategoriyasini bashorat qilish** uchun yaratilgan.  
Model **Machine Learning (Random Forest Classifier)** yordamida qurilgan va tayyor **pipeline** ko‘rinishida saqlanadi.

---

## 📊 Dataset haqida ma’lumot

### 📁 Fayl nomi
`video_ids_real_10000.csv`

### 📈 Ustunlar tavsifi
| Ustun nomi | Tavsif |
|-------------|--------|
| `VideoID` | Videoning unikal identifikatori |
| `Region` | Video joylashgan hudud |
| `CategoryID` | YouTube tomonidan berilgan kategoriya ID |
| `CategoryName` | Kategoriya nomi (maqsad ustuni) |
| `Title` | Video sarlavhasi |
| `Channel` | Kanal nomi |
| `Channel Views` | Kanalning umumiy ko‘rishlar soni |
| `Likes` | Video layklar soni |
| `Comments` | Video izohlar soni |
| `Subscribers` | Kanal obunachilari soni |
| `Published Date` | Video joylangan sana |
| `Like_per_View` | Like sonining ko‘rishlarga nisbati |
| `Comment_per_Sub` | Izohlarning obunachilarga nisbati |
| `Country` | Kanal joylashuvi |
| `Cluster`, `Video_Cluster`, `Channel_Cluster`, `Country_Cluster_x`, `Category_Cluster_x` | K-means yordamida klasterlangan ustunlar |
| `CategoryID_enc` | Kategoriya ID’ning kodlangan versiyasi |

---

## 🧠 Model haqida

Model quyidagi ML texnikasi yordamida qurilgan:

- **Model:** `RandomForestClassifier`
- **Scaler:** `StandardScaler`
- **Encoder:** `OneHotEncoder`
- **Pipeline:** `ColumnTransformer` + `RandomForestClassifier`
- **Ma’lumot ajratish:** `train_test_split(test_size=0.2, random_state=42)`

Model `CategoryName` ustunini target sifatida o‘rgangan.

---

## ⚙️ Pipeline yaratish va saqlash

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Ma'lumotni yuklash
df = pd.read_csv('video_ids_real_10000.csv')

# Target va xususiyatlar
X = df.drop('CategoryName', axis=1)
y = df['CategoryName']

# Sonli va kategorik ustunlar
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Transformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Modelni o'qitish
pipeline.fit(X, y)

# Saqlash
joblib.dump(pipeline, 'Category_predict_pipeline.pkl')
```

---

## 🔍 Bashorat olish (offline ma’lumot bilan)

```python
import pandas as pd
import joblib

# Pipeline’ni yuklash
loaded_pipeline = joblib.load('Category_predict_pipeline.pkl')

# Yangi ma'lumot
offline_data = pd.DataFrame({
    'VideoID': [12345],
    'Region': ['US'],
    'CategoryID': [24],
    'Title': ['My First Vlog'],
    'Channel': ['AzizbekTV'],
    'Channel Views': [1200000],
    'Likes': [4500],
    'Comments': [350],
    'Subscribers': [23000],
    'Published Date': ['2025-01-10'],
    'Like_per_View': [0.05],
    'Comment_per_Sub': [0.015],
    'Country': ['USA'],
    'Cluster': [2],
    'Video_Cluster': [1],
    'Channel_Cluster': [3],
    'Country_Cluster_x': [0],
    'Category_Cluster_x': [2],
    'CategoryID_enc': [24]
})

# Bashorat
pred = loaded_pipeline.predict(offline_data)
print("🔮 Bashorat qilingan CategoryName:", pred[0])
```

---

## 🧰 Ishlatilgan kutubxonalar
```bash
pip install pandas scikit-learn joblib numpy matplotlib
```

---

## 📈 Natija
Model **CategoryName** ustunini yuqori aniqlikda bashorat qila oladi.  
Ushbu loyiha YouTube kontentlarini avtomatik ravishda kategoriyalash uchun ishlatiladi.

---

## 👤 Muallif
**Azizbek Developer (Rasulbek)**  
📅 Sana: 2025-10-25
📧 Email: —  
🔗 GitHub: [github.com/rasulbekdeveloper907](https://github.com/rasulbekdeveloper907)

---
