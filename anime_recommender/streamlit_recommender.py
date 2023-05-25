import streamlit as st
import pandas as pd
import pickle
import os

# Load the models and data when the script starts
AnimesDF = pd.read_csv(f'{os.getcwd()}/raw_data/anime_cleaned.csv')

with open(f"{os.getcwd()}/anime_recommender/trained_models/baseline_model.pickle", 'rb') as f:
    loaded_model = pickle.load(f)

with open(f"{os.getcwd()}/anime_recommender/trained_models/knn_model.pickle", 'rb') as f:
    loaded_knn_model = pickle.load(f)

def get_item_recommendations(algo, algo_items, anime_title, anime_id=100000, k=10):
    anime_title = anime_title.strip().lower()
    matching_animes = AnimesDF[AnimesDF['title'].str.lower() == anime_title]

    if matching_animes.empty:
        st.write("No matching anime found. Please check your input.")
        return

    if anime_id == 100000:
        anime_id = matching_animes['anime_id'].iloc[0]

    iid = algo_items.trainset.to_inner_iid(anime_id)
    neighbors = algo_items.get_neighbors(iid, k=k)
    raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)
    st.write("Here's a list of anime titles you might enjoy")
    df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])
    df = pd.merge(df, AnimesDF, left_on = 'Anime_ID', right_on = 'anime_id', how = 'left')
    return df[['Anime_ID', 'title', 'genre']]

# Set up the Streamlit interface
st.title('Anime Recommendation System')

anime_title = st.text_input('Please enter an anime title')

if st.button('Get recommendations'):
    recommendations = get_item_recommendations(loaded_model, loaded_knn_model, anime_title)
    st.write(recommendations)
