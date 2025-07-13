import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from rapidfuzz import process
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Load TMDB API Key
load_dotenv()
TMDB_API_KEY = st.secrets["TMDB_API_KEY"] if "TMDB_API_KEY" in st.secrets else os.getenv("TMDB_API_KEY")

# ------------------ Helper Functions ------------------
def get_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url)
        data = res.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None

def get_trailer_url(title):
    try:
        url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url)
        data = res.json()
        if data['results']:
            media_type = data['results'][0]['media_type']
            media_id = data['results'][0]['id']
            video_url = f"https://api.themoviedb.org/3/{media_type}/{media_id}/videos?api_key={TMDB_API_KEY}"
            video_data = requests.get(video_url).json()
            for video in video_data.get('results', []):
                if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                    return f"https://youtube.com/watch?v={video['key']}"
    except:
        pass
    return None

# ------------------ Preprocessing ------------------
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("data/netflix_titles.csv")
    df = df[['title', 'listed_in', 'duration', 'release_year', 'country', 'rating', 'description']].dropna()

    df['duration'] = df['duration'].str.extract('(\d+)')
    df.dropna(subset=['duration'], inplace=True)
    df['duration'] = df['duration'].astype(float)

    df['genres'] = df['listed_in'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)
    df = pd.concat([df.reset_index(drop=True), genre_encoded], axis=1)
    df.drop(['listed_in', 'genres'], axis=1, inplace=True)

    genre_cols = list(genre_encoded.columns)
    features = df[genre_cols + ['duration']].dropna()
    df = df.loc[features.index]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)
    df['pca_1'] = components[:, 0]
    df['pca_2'] = components[:, 1]

    return df, genre_cols

# ------------------ Recommender ------------------
def recommend_show(df, show_name, n=5):
    if show_name not in df['title'].values:
        return []
    cluster_id = df[df['title'] == show_name]['Cluster'].values[0]
    recommendations = df[(df['Cluster'] == cluster_id) & (df['title'] != show_name)]
    return recommendations.sample(n=min(n, len(recommendations)))

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🎬 Netflix Recommender", layout="wide")
st.title("🎬 Netflix Clustering-Based Recommender")
st.markdown("Recommend similar shows based on genre and duration using K-Means clustering.")

# Load data
df, genre_cols = load_and_preprocess()

# ------------------ Sidebar Filters ------------------
st.sidebar.header("🔎 Filter Shows")
year_opt = sorted(df['release_year'].dropna().unique())
country_opt = sorted(df['country'].dropna().unique())
rating_opt = sorted(df['rating'].dropna().unique())

selected_year = st.sidebar.selectbox("Release Year", [None] + list(year_opt))
selected_country = st.sidebar.selectbox("Country", [None] + list(country_opt))
selected_rating = st.sidebar.selectbox("Rating", [None] + list(rating_opt))

filtered_df = df.copy()
if selected_year:
    filtered_df = filtered_df[filtered_df['release_year'] == selected_year]
if selected_country:
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_rating:
    filtered_df = filtered_df[filtered_df['rating'] == selected_rating]

# ------------------ Search Input ------------------
search_input = st.text_input("🔍 Search for a show:")
if search_input:
    match_result = process.extractOne(search_input, filtered_df['title'].tolist())
    if match_result:
        match, score, _ = match_result
        selected_show = match if score > 60 else None
        if selected_show:
            st.success(f"📌 Matched: **{selected_show}** (Score: {score})")
        else:
            st.warning("No good match found.")
    else:
        selected_show = None
else:
    selected_show = st.selectbox("Or pick a show:", filtered_df['title'].sort_values().unique())

# ------------------ Recommendations ------------------
if selected_show:
    cluster_id = filtered_df[filtered_df['title'] == selected_show]['Cluster'].values[0]
    st.subheader(f"📦 **'{selected_show}'** is in Cluster **{cluster_id}**")

    rec_df = recommend_show(filtered_df, selected_show)
    if not rec_df.empty:
        st.subheader("🎯 Recommended Shows")
        cols = st.columns(5)
        for i, (title, row) in enumerate(rec_df.iterrows()):
            with cols[i % 5]:
                st.markdown(f"**{row['title']}**")
                poster = get_poster(row['title'])
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("*(No image)*")
                st.caption(f"📝 {row['description'][:100]}...")
                trailer_url = get_trailer_url(row['title'])
                if trailer_url:
                    st.markdown(f"[▶ Watch Trailer]({trailer_url})")
    else:
        st.warning("No recommendations found for this show.")

# ------------------ PCA Plot ------------------
with st.expander("📊 Show PCA Cluster Visualization"):
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x='pca_1', y='pca_2', hue='Cluster', palette='tab10', s=60)
        selected_row = filtered_df[filtered_df['title'] == selected_show]
        plt.scatter(
            selected_row['pca_1'], selected_row['pca_2'],
            color='black', marker='X', s=200, label='Selected'
        )
        plt.title("PCA View of Netflix Clusters")
        plt.legend()
        st.pyplot(fig)
    else:
        st.info("No data to display PCA plot for selected filters.")
