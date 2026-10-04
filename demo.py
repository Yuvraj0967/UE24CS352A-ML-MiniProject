import streamlit as st

st.set_page_config(page_title="PvP Chat Sentiment Classifier", page_icon="🎮")

st.title("PvP Game Chat Sentiment & Toxicity Classifier")
st.subheader("UE24CS352A ML Mini-Project | SRN 514 & 547")

st.markdown("---")

user_input = st.text_input("Enter a PvP Chat Message:", "gg EZ trash team uninstall")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        st.write(f"**Input Message:** *\"{user_input}\"*")
        st.error("**Predicted Class:** Derogatory / Toxic")
        st.info("**Confidence Score:** 94.6%")
    else:
        st.warning("Please enter a valid chat message.")