# streamlit_app/app.py
import streamlit as st

st.set_page_config(page_title="Music Recommender", page_icon="", layout="wide")

st.title(" Deep Learning Music Recommendation System")
st.markdown("""
Welcome to the AI-powered Music Recommendation Engine! 
This system uses advanced Deep Learning algorithms to find your next favorite song from a dataset of over 80,000 tracks.

### How to use this app:
1. Go to the **Music Recommend** page (in the sidebar) to search for a song and get personalized recommendations.
2. Check out the **Model Stats** page to see how our algorithms compare in terms of diversity, novelty, and accuracy.

### Our Algorithms:
- **Baseline**: Classic Content-Based filtering using raw audio features.
- **RBM (Restricted Boltzmann Machine)**: Deep learning model that finds hidden, latent musical patterns.
- **Markov Chain**: Predicts the next song in a playlist based on sequential flow.
- **Hybrid**: The ultimate combination of all three for the most accurate and diverse recommendations.
""")