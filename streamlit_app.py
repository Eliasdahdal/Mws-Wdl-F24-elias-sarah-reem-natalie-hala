import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os

st.set_page_config(page_title="Olivetti Model Comparison", layout="wide")

# --- Load Dataset ---
faces = fetch_olivetti_faces()
X = faces.images[..., np.newaxis]
y = to_categorical(faces.target, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=faces.target, random_state=42)

# --- App Title ---
st.title("🧠 Olivetti Model Evaluation Dashboard")
st.markdown("Compare model performance with and without augmentation on the Olivetti faces dataset.")

# --- Sidebar Options ---
with st.sidebar:
    st.header("🛠 Model Options")
    model_choice = st.radio("Select Model", ["Without Augmentation", "With Augmentation"])
    show_samples = st.checkbox("Show Prediction Samples")
    show_comparison_plot = st.checkbox("Show Accuracy Comparison")

# --- Load Selected Model ---
model_filename = "model_olivetti_plain.h5" if model_choice == "Without Augmentation" else "model_olivetti_aug.h5"

if os.path.exists(model_filename):
    model = load_model(model_filename)
    st.success(f"✅ Loaded model: {model_filename}")
else:
    st.error(f"❌ Model file '{model_filename}' not found. Please upload it.")
    st.stop()

# --- Evaluate Model ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.subheader(f"📊 Model Accuracy: {accuracy * 100:.2f}%")

# --- Show Sample Predictions ---
if show_samples:
    st.subheader("🖼 Sample Predictions")
    indices = np.random.choice(range(len(X_test)), 10, replace=False)
    cols = st.columns(5)
    predictions = model.predict(X_test[indices])
    for i, idx in enumerate(indices):
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(predictions[i])
        with cols[i % 5]:
            st.image(X_test[idx].squeeze(), width=100, caption=f"True: {true_label} | Pred: {pred_label}")

# --- Accuracy Comparison ---
if show_comparison_plot:
    if os.path.exists("model_plain.h5") and os.path.exists("model_aug.h5"):
        model_plain = load_model("model_plain.h5")
        model_aug = load_model("model_aug.h5")
        acc_plain = model_plain.evaluate(X_test, y_test, verbose=0)[1]
        acc_aug = model_aug.evaluate(X_test, y_test, verbose=0)[1]

        st.subheader("📈 Accuracy Comparison")
        fig, ax = plt.subplots()
        ax.bar(["No Augmentation", "With Augmentation"], [acc_plain * 100, acc_aug * 100], color=["blue", "orange"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.warning("Both model files must be available to compare accuracies.")
