import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os

# Set layout
st.set_page_config(page_title="Olivetti Smart Comparison", layout="wide")

# Load Olivetti data
faces = fetch_olivetti_faces()
X = faces.images[..., np.newaxis]
y = to_categorical(faces.target, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=faces.target, random_state=42)

# Check model files
model_files = {
    "Without Augmentation": "model_olivetti_plain.h5",
    "With Augmentation": "model_olivetti_aug.h5"
}
available_models = {name: path for name, path in model_files.items() if os.path.exists(path)}

# UI
st.title("🧠 Smart Olivetti Model Explorer")
st.markdown("Explore and compare Keras models trained on Olivetti Faces.")

# Select model
if available_models:
    selected_model_name = st.selectbox("Select a Model", list(available_models.keys()))
    model_path = available_models[selected_model_name]
    model = load_model(model_path)
    st.success(f"✅ Loaded: {selected_model_name} ({model_path})")

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.metric(label="🎯 Accuracy", value=f"{acc*100:.2f} %")

    # Show predictions
    if st.checkbox("🔍 Show Sample Predictions"):
        indices = np.random.choice(len(X_test), 10, replace=False)
        preds = model.predict(X_test[indices])
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i % 5]:
                true = np.argmax(y_test[idx])
                pred = np.argmax(preds[i])
                st.image(X_test[idx].squeeze(), width=100, caption=f"True: {true} | Pred: {pred}")

else:
    st.error("❌ No models found. Please upload at least one `.h5` model.")
    st.stop()

# Show comparison if both exist
if len(available_models) == 2:
    st.divider()
    st.subheader("📊 Accuracy Comparison")
    accs = {}
    for name, path in available_models.items():
        model_temp = load_model(path)
        accs[name] = model_temp.evaluate(X_test, y_test, verbose=0)[1] * 100

    fig, ax = plt.subplots()
    ax.bar(accs.keys(), accs.values(), color=["blue", "orange"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)
