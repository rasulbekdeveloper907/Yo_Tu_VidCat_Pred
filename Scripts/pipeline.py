import sys
import os
import logging

# 📌 Source papkani import yo‘liga qo‘shamiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source')))

# 📌 Endi pipeline.py dagi klasslarni import qilamiz
from pipeline import DataLoader, DataPreProcessing
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 🔹 Dataset joylashgan manzil
    file_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\raw_data\data_merged.csv"

    # 🔹 1. Datasetni yuklaymiz
    loader = DataLoader(file_path)
    raw_data = loader.load_dataset()

    if raw_data.empty:
        logging.error("Dataset yuklanmadi. Iltimos fayl yo'lini tekshiring.")
        sys.exit(1)

    # 🔹 2. Preprocessing pipeline
    preprocessor = DataPreProcessing(raw_data)

    df_cleaned = preprocessor.handle_missing_values()
    df_cleaned = preprocessor.remove_duplicates()
    df_encoded = preprocessor.encode_categorical()
    df_scaled = preprocessor.scale_numeric()

    # 🔹 3. Natijani chiqarish
    final_df = preprocessor.get_processed_data()
    print(final_df.head())

    # 🔹 (ixtiyoriy) Natijani faylga yozish
    output_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\processed\processed_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Final processed data saved to {output_path}")
