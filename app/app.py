import sys, os
# ✅ Add the project root to sys.path first
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from dotenv import load_dotenv
from pipeline.pipeline import AnimeRecommenderPipeline  # now it will resolve properly

st.set_page_config(page_title="Anime Recommender", layout="wide")

load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommenderPipeline()

pipeline = init_pipeline()

st.title("Anime Recommender System")
st.write("Get anime recommendations based on your interests!")

user_query = st.text_input(
    "Enter your anime preferences or interests:",
    placeholder="e.g., 'I like a light-hearted anime with school life and romance'"
)

if user_query:
    with st.spinner("Generating recommendations..."):
        response = pipeline.recommend(user_query)
        st.markdown("### 🎯 Recommendations:")
        st.write(response)
