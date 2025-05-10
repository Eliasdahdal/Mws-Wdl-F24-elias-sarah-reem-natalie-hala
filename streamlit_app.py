import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os

# Config
st.set_page_config(page_title="Olivetti & Fer Dashboard", layout="wide")

# Display user information at the top of the dashboard
st.markdown("""
#### MWS_WDL_S24
Supervisor: Dr. Bassel Alkhatib  
Created by: Elias_335295 - sarah_326852 - Reem_321116 - Hala_332141 - Natalie_336924
""")

# Load Olivetti data
faces = fetch_olivetti_faces()
X = faces.images[..., np.newaxis]
y = to_categorical(faces.target, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=faces.target, random_state=42)

# Models
model_files = {
    "Without Augmentation": "model_olivetti_plain.h5",
    "With Augmentation": "model_olivetti_aug.h5"
}
available_models = {k: v for k, v in model_files.items() if os.path.exists(v)}

# Sidebar
st.sidebar.title("🧭 Navigation")
selected_tab = st.sidebar.radio("Choose Tab", ["Model Info", "Predictions"])
st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox("🧠 Choose Model", list(available_models.keys()))
model_path = available_models[selected_model]
model = load_model(model_path)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
preds = model.predict(X_test)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

# === Model Info Tab ===
if selected_tab == "Model Info":
    st.markdown("## 📊 Olivetti Model Evaluation Dashboard")
    st.markdown("Explore model performance and compare accuracy visually.")

    col1, col2 = st.columns([1, 2])
    col1.metric("🎯 Accuracy", f"{acc * 100:.2f} %")
    col1.markdown(f"**Model File:** `{model_path}`")
    
    # Donut-like half pie (Success vs Error)
    success = round(acc * 100, 2)
    failure = round(100 - success, 2)
    fig_donut = go.Figure(data=[go.Pie(
        values=[success, failure],
        labels=['✅ Correct', '❌ Incorrect'],
        hole=0.5,
        marker_colors=['green', 'red'],
        textinfo='label+percent',
        sort=False,
    )])
    fig_donut.update_layout(
        title_text=f"{selected_model} Prediction Distribution",
        showlegend=False,
        height=400
    )
    col2.plotly_chart(fig_donut, use_container_width=True)

    # Compare models (if both exist)
    if len(available_models) == 2:
        accs = {}
        for name, path in available_models.items():
            m = load_model(path)
            accs[name] = m.evaluate(X_test, y_test, verbose=0)[1] * 100

        st.markdown("### 📊 Accuracy Comparison Between Models")
        fig_bar = go.Figure(data=[
            go.Bar(name='Accuracy (%)', x=list(accs.keys()), y=list(accs.values()), marker_color=['blue', 'orange'])
        ])
        fig_bar.update_layout(yaxis_title="Accuracy %", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Upload both models to compare their performance.")

# === Predictions Tab ===
elif selected_tab == "Predictions":
    st.title("🖼 Prediction Samples")
    num = st.slider("Number of Samples", 5, 20, 10)
    indices = np.random.choice(len(X_test), size=num, replace=False)

    cols = st.columns(5)
    for i, idx in enumerate(indices):
        with cols[i % 5]:
            pred = pred_classes[idx]
            true = true_classes[idx]
            label = "✅ Correct" if pred == true else "❌ Wrong"
            st.image(X_test[idx].squeeze(), width=100, caption=f"{label}\nT:{true} P:{pred}")
