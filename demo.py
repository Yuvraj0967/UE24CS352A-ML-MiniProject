import streamlit as st
import pickle
import os

st.set_page_config(page_title="PvP Chat Sentiment Classifier", page_icon="🎮")

st.title("PvP Game Chat Sentiment & Toxicity Classifier")
st.subheader("UE24CS352A ML Mini-Project | SRN 514 & 547")
st.markdown("---")

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

model, vectorizer = load_artifacts()

# User interface input
user_input = st.text_input("Enter a PvP Chat Message:", "gg EZ trash team uninstall")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess input and predict
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        probs = model.predict_proba(vec_input)[0]
        confidence = max(probs) * 100

        st.write(f"**Input Message:** *\"{user_input}\"*")
        
        # Display styled output based on predicted class
        if prediction == "Derogatory / Toxic":
            st.error(f"**Predicted Class:** {prediction}")
        elif prediction == "Positive":
            st.success(f"**Predicted Class:** {prediction}")
        else:
            st.info(f"**Predicted Class:** {prediction}")
            
        st.info(f"**Confidence Score:** {confidence:.1f}%")
    else:
        st.warning("Please enter a valid chat message.")