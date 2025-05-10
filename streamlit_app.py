import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from tensorflow.keras.utils import to_categorical

st.set_page_config(page_title="Olivetti Faces Explorer", layout="wide")

# Load data
faces = fetch_olivetti_faces()
X = faces.images[..., np.newaxis]
y = to_categorical(faces.target, 40)

st.title("🧠 Olivetti Faces Explorer")
st.markdown("Explore the classic Olivetti face dataset interactively.")

# Sidebar
with st.sidebar:
    st.header("Options")
    view_mode = st.radio("View Mode", ["Random Samples", "Show a Person"])
    person_id = st.slider("Person ID", 0, 39, 0)
    show_stats = st.checkbox("Show Dataset Info")

# Show stats
if show_stats:
    st.subheader("📊 Dataset Info")
    st.write(f"Total images: {X.shape[0]}")
    st.write(f"Image shape: {X.shape[1:]} (grayscale)")
    st.write("Each person has 10 images (40 persons).")

# Show random images
if view_mode == "Random Samples":
    st.subheader("🔀 Random Samples")
    indices = np.random.choice(range(X.shape[0]), size=10, replace=False)
    cols = st.columns(5)
    for i, idx in enumerate(indices):
        with cols[i % 5]:
            st.image(X[idx].squeeze(), width=100, caption=f"ID: {np.argmax(y[idx])}")

# Show person-specific images
if view_mode == "Show a Person":
    st.subheader(f"🧍‍♂️ Images of Person #{person_id}")
    cols = st.columns(5)
    count = 0
    for i in range(X.shape[0]):
        if np.argmax(y[i]) == person_id:
            with cols[count % 5]:
                st.image(X[i].squeeze(), width=100, caption=f"Face #{count + 1}")
            count += 1
