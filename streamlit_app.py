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
st.set_page_config(page_title="Olivetti Dashboard", layout="wide")

# Load data
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
selected_tab = st.sidebar.radio("Choose Tab", ["Model Info", "Predictions", "Comparison"])

st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox("🧠 Choose Model", list(available_models.keys()))
model_path = available_models[selected_model]
model = load_model(model_path)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
preds = model.predict(X_test)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

# TABS
if selected_tab == "Model Info":
    st.title(f"🔍 {selected_model}")
    st.metric("Accuracy", f"{acc * 100:.2f}%")
    st.markdown(f"Model path: `{model_path}`")
    st.info("This model is evaluated on 30% of the Olivetti test dataset.")

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

elif selected_tab == "Comparison":
    st.title("📊 Accuracy Comparison")
    if len(available_models) == 2:
        accs = {}
        for name, path in available_models.items():
            m = load_model(path)
            accs[name] = m.evaluate(X_test, y_test, verbose=0)[1] * 100

        fig = go.Figure(data=[
            go.Bar(name='Accuracy (%)', x=list(accs.keys()), y=list(accs.values()), marker_color=['blue', 'orange'])
        ])
        fig.update_layout(title="Model Accuracy Comparison", yaxis_title="Accuracy %")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Both models must be available to show comparison.")
