import gradio as gr
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------
# Model yuklash
# ---------------------------
MODEL_PATH = Path("Models/GradientBoostingRegressor.joblib")
model = joblib.load(MODEL_PATH)

# ---------------------------
# Predict function
# ---------------------------
def predict_lifespan(name, birth_date, birth_place, death_date, death_place,
                     occupation, awards, alma_mater, education, spouse,
                     children, occupation_cluster, birth_year, death_year,
                     life_span_cluster, edu_award_cluster, bio_cluster):
    df = pd.DataFrame([{
        "name": name,
        "birth_date": birth_date,
        "birth_place": birth_place,
        "death_date": death_date,
        "death_place": death_place,
        "occupation": occupation,
        "awards": awards,
        "alma_mater": alma_mater,
        "education": education,
        "spouse": spouse,
        "children": children,
        "occupation_cluster": occupation_cluster,
        "birth_year": birth_year,
        "death_year": death_year,
        "life_span_cluster": life_span_cluster,
        "edu_award_cluster": edu_award_cluster,
        "bio_cluster": bio_cluster
    }])
    prediction = float(model.predict(df)[0])
    return round(prediction,2)

# ---------------------------
# Dashboard
# ---------------------------
with gr.Blocks(title="💀 Life Span Prediction Dashboard") as demo:
    gr.Markdown("# 🧬 Life Span Prediction\nPredict human life span (years) using GradientBoostingRegressor")

    with gr.Row():
        # Personal Info
        with gr.Column(scale=1):
            gr.Markdown("### 🧑 Personal Info")
            name = gr.Textbox(label="Name", value="John Doe")
            birth_date = gr.Textbox(label="Birth Date (YYYY-MM-DD)", value="1970-01-01")
            birth_place = gr.Textbox(label="Birth Place", value="New York")
            death_date = gr.Textbox(label="Death Date (YYYY-MM-DD)", value="2020-01-01")
            death_place = gr.Textbox(label="Death Place", value="Los Angeles")

        # Career & Bio
        with gr.Column(scale=1):
            gr.Markdown("### 🏆 Career & Bio")
            occupation = gr.Textbox(label="Occupation", value="Scientist")
            awards = gr.Number(label="Awards Count", value=3, precision=0)
            alma_mater = gr.Textbox(label="Alma Mater", value="MIT")
            education = gr.Textbox(label="Education Level", value="PhD")
            spouse = gr.Number(label="Spouse Count", value=1, precision=0)
            children = gr.Number(label="Children Count", value=2, precision=0)
            occupation_cluster = gr.Number(label="Occupation Cluster", value=5, precision=0)
            birth_year = gr.Number(label="Birth Year", value=1970, precision=0)
            death_year = gr.Number(label="Death Year", value=2020, precision=0)
            life_span_cluster = gr.Number(label="Life Span Cluster", value=50, precision=0)
            edu_award_cluster = gr.Number(label="Education & Awards Cluster", value=10, precision=0)
            bio_cluster = gr.Number(label="Bio Cluster", value=8, precision=0)

    # Predict button & output
    predict_btn = gr.Button("Predict 🏁", variant="primary")
    life_span_output = gr.Number(label="🧪 Predicted Life Span (years)", value=0, interactive=False)

    predict_btn.click(
        fn=predict_lifespan,
        inputs=[
            name, birth_date, birth_place, death_date, death_place,
            occupation, awards, alma_mater, education, spouse,
            children, occupation_cluster, birth_year, death_year,
            life_span_cluster, edu_award_cluster, bio_cluster
        ],
        outputs=[life_span_output]
    )

    gr.Markdown("---\nBuilt by **Your Name** | GradientBoostingRegressor | ML Dashboard")

# ---------------------------
# Launch with local + live link
# ---------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)