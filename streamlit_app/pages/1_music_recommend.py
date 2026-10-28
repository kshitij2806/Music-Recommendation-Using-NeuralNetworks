# streamlit_app/pages/1_music_recommend.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import os

# --- CACHING FUNCTIONS (Crucial for Speed) ---
@st.cache_data
def load_data():
    df_features = pd.read_parquet('./data/processed/processed_features.parquet')
    track_mapping = joblib.load('./saved_models/preprocessors/track_mapping.pkl')
    trackid_to_index = joblib.load('./saved_models/preprocessors/trackid_to_index.pkl')
    df_raw = pd.read_csv('./data/raw/music_dataset.csv')
    return df_features, track_mapping, trackid_to_index, df_raw

@st.cache_resource
def load_models(X):
    # Baseline
    baseline_model = joblib.load('./saved_models/baseline_nn_model.pkl')
    
    # RBM Architecture
    class RBM(nn.Module):
        def __init__(self, n_visible, n_hidden):
            super().__init__()
            self.encoder = nn.Linear(n_visible, n_hidden)
            self.decoder = nn.Linear(n_hidden, n_visible)
        def forward(self, x):
            return torch.sigmoid(self.decoder(torch.sigmoid(self.encoder(x)))), torch.sigmoid(self.encoder(x))
            
    rbm_model = RBM(X.shape[1], 64)
    rbm_model.load_state_dict(torch.load('./saved_models/rbm_weights.pth', weights_only=True))
    rbm_model.eval()
    rbm_nn_model = joblib.load('./saved_models/rbm_nn_model.pkl')
    
    # Pre-calculate hidden space for all songs
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        _, X_hidden = rbm_model(X_tensor)
    X_hidden_np = X_hidden.numpy()
    
    # Markov Chain
    markov_indices = np.load('./saved_models/markov_indices.npy')
    markov_probs = np.load('./saved_models/markov_probs.npy')
    
    return baseline_model, rbm_nn_model, X_hidden_np, markov_indices, markov_probs

# --- LOAD DATA & MODELS ---
df_features, track_mapping, trackid_to_index, df_raw = load_data()
track_ids = df_features['track_id'].values
X = df_features.drop(columns=['track_id']).values
baseline_model, rbm_nn_model, X_hidden_np, markov_indices, markov_probs = load_models(X)

# --- UI SETUP ---
st.set_page_config(page_title="Recommend Music", page_icon="🎵", layout="wide")
st.title("🎵 Find Your Next Favorite Song")
st.markdown("Search for a track or artist below to get AI-powered recommendations.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Recommendation Algorithm:",
        ["Hybrid (Best)", "RBM (Deep Learning)", "Markov Chain (Sequential)", "Baseline (Content-Based)"]
    )
    num_recs = st.slider("Number of Recommendations:", 5, 20, 10)
    
    # --- FIX: Define default weights first ---
    w_rbm, w_markov, w_baseline = 0.4, 0.4, 0.2 
    
    if model_choice == "Hybrid (Best)":
        st.markdown("**Hybrid Weights:**")
        w_rbm = st.slider("RBM Weight", 0.0, 1.0, 0.4)
        w_markov = st.slider("Markov Weight", 0.0, 1.0, 0.4)
        w_baseline = 1.0 - w_rbm - w_markov
        st.caption(f"Baseline Weight: {w_baseline:.2f} (Auto-calculated)")

# --- SEARCH LOGIC ---
search_query = st.text_input("🔍 Search for a song or artist...", placeholder="e.g., Comedy, Gen Hoshino, acoustic")

selected_track_id = None

if search_query:
    # Filter dataframe efficiently
    mask = df_raw['track_name'].str.contains(search_query, case=False, na=False) | \
           df_raw['artists'].str.contains(search_query, case=False, na=False)
    filtered_df = df_raw[mask]
    
    if len(filtered_df) > 0:
        options = [f"{row['track_name']} by {row['artists']}" for _, row in filtered_df.iterrows()]
        selected_option = st.selectbox("Select a track from the results:", options)
        
        # Get the track_id of the selected option
        selected_row = filtered_df[filtered_df.apply(lambda row: f"{row['track_name']} by {row['artists']}" == selected_option, axis=1)]
        if not selected_row.empty:
            selected_track_id = selected_row.iloc[0]['track_id']
    else:
        st.warning("No tracks found. Try a different search term.")

# --- RECOMMENDATION LOGIC ---
def get_recommendations(track_id, model_name, n_recs, w_r=0.4, w_m=0.4, w_b=0.2):
    idx = trackid_to_index[track_id]
    rec_indices = []
    
    if model_name == "Baseline (Content-Based)":
        _, indices = baseline_model.kneighbors(X[idx].reshape(1, -1), n_neighbors=n_recs+1)
        rec_indices = indices[0][1:]
        
    elif model_name == "RBM (Deep Learning)":
        _, indices = rbm_nn_model.kneighbors(X_hidden_np[idx].reshape(1, -1), n_neighbors=n_recs+1)
        rec_indices = indices[0][1:]
        
    elif model_name == "Markov Chain (Sequential)":
        step1_indices = markov_indices[idx]
        step1_probs = markov_probs[idx]
        two_step_probs = {}
        for i in range(len(step1_indices)):
            b_idx = step1_indices[i]
            prob_a_to_b = step1_probs[i]
            step2_indices = markov_indices[b_idx]
            step2_probs = markov_probs[b_idx]
            for j in range(len(step2_indices)):
                c_idx = step2_indices[j]
                if c_idx == idx: continue
                combined_prob = prob_a_to_b * step2_probs[j]
                two_step_probs[c_idx] = two_step_probs.get(c_idx, 0) + combined_prob
        sorted_tracks = sorted(two_step_probs.items(), key=lambda x: x[1], reverse=True)
        rec_indices = [idx for idx, _ in sorted_tracks[:n_recs]]
        
    elif model_name == "Hybrid (Best)":
        # Get candidates from RBM and Markov
        _, rbm_indices = rbm_nn_model.kneighbors(X_hidden_np[idx].reshape(1, -1), n_neighbors=21)
        rbm_candidates = set(rbm_indices[0][1:])
        
        step1_indices = markov_indices[idx]
        step1_probs = markov_probs[idx]
        markov_candidates_set = set()
        markov_scores_raw = {}
        for i in range(len(step1_indices)):
            b_idx = step1_indices[i]
            prob_a_to_b = step1_probs[i]
            step2_indices = markov_indices[b_idx]
            step2_probs = markov_probs[b_idx]
            for j in range(len(step2_indices)):
                c_idx = step2_indices[j]
                if c_idx == idx: continue
                combined_prob = prob_a_to_b * step2_probs[j]
                markov_candidates_set.add(c_idx)
                markov_scores_raw[c_idx] = markov_scores_raw.get(c_idx, 0) + combined_prob
                
        all_candidates = rbm_candidates | markov_candidates_set
        all_candidates.discard(idx)
        
        # Score with Baseline
        baseline_scores_raw = {}
        input_vec = X[idx]
        for c_idx in all_candidates:
            candidate_vec = X[c_idx]
            sim = np.dot(input_vec, candidate_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(candidate_vec))
            baseline_scores_raw[c_idx] = sim
            
        # Normalize scores
        def normalize(scores_dict):
            if not scores_dict: return {}
            vals = list(scores_dict.values())
            min_val, max_val = min(vals), max(vals)
            if max_val == min_val: return {k: 0.5 for k in scores_dict}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
            
        norm_rbm = {idx: 1.0 for idx in rbm_candidates if idx in all_candidates}
        norm_markov = normalize(markov_scores_raw)
        norm_baseline = normalize(baseline_scores_raw)
        
        final_scores = {}
        for c_idx in all_candidates:
            score_r = norm_rbm.get(c_idx, 0.0)
            score_m = norm_markov.get(c_idx, 0.0)
            score_b = norm_baseline.get(c_idx, 0.0)
            final_scores[c_idx] = (w_r * score_r) + (w_m * score_m) + (w_b * score_b)
            
        sorted_tracks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rec_indices = [idx for idx, _ in sorted_tracks[:n_recs]]
        
    return rec_indices

# --- DISPLAY RESULTS ---
if selected_track_id:
    st.divider()
    st.subheader(f"🎧 Recommendations for: {track_mapping[selected_track_id]['track_name']}")
    st.caption(f"Artist: {track_mapping[selected_track_id]['artists']} | Genre: {track_mapping[selected_track_id]['track_genre']}")
    
    with st.spinner(f"Generating {num_recs} recommendations using {model_choice}..."):
        rec_indices = get_recommendations(selected_track_id, model_choice, num_recs, w_rbm, w_markov, w_baseline)
        
    st.success("Recommendations generated!")
    
    # Display as clean UI cards
    for i, rec_idx in enumerate(rec_indices, 1):
        rec_id = track_ids[rec_idx]
        info = track_mapping[rec_id]
        
        with st.container():
            col1, col2, col3 = st.columns([0.1, 0.6, 0.3])
            with col1:
                st.markdown(f"### {i}")
            with col2:
                st.markdown(f"### {info['track_name']}")
                st.markdown(f"**{info['artists']}**")
            with col3:
                st.markdown(f"🎵 *{info['track_genre']}*")
            st.markdown("---")