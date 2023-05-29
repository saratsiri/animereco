import streamlit as st
import pandas as pd
import pickle
import os
import re

# Load the models and data when the script starts
AnimesDF = pd.read_csv(f'{os.getcwd()}/raw_data/anime_cleaned.csv')

with open(f"{os.getcwd()}/anime_recommender/trained_models/baseline_model.pickle", 'rb') as f:
    loaded_model = pickle.load(f)

with open(f"{os.getcwd()}/anime_recommender/trained_models/knn_model.pickle", 'rb') as f:
    loaded_knn_model = pickle.load(f)

import re

# Preprocessing step
AnimesDF['title_lower'] = AnimesDF['title'].str.lower()
AnimesDF['title_english_lower'] = AnimesDF['title_english'].str.lower()

def get_item_recommendations(algo, algo_items, anime_title, anime_id=100000, k=20):
    try:
        if not anime_title or anime_title.isspace():
            st.write("Enter something you neckbeard! There's no empty anime name.")
            return

        anime_title = anime_title.strip().lower()

        matching_animes = AnimesDF[AnimesDF['title'].str.lower().str.startswith(anime_title)]

        if matching_animes.empty:
            matching_animes = AnimesDF[AnimesDF['title_english'].str.lower().str.startswith(anime_title)]

        if matching_animes.empty:
            st.write("No matching anime found. Please check your input.")
            return

        best_match_index = matching_animes['title'].str.len().idxmin()
        best_match = matching_animes.loc[best_match_index]
        st.write("Assuming you meant: ", best_match['title'])
        anime_id = best_match['anime_id']

        iid = algo_items.trainset.to_inner_iid(anime_id)
        neighbors = algo_items.get_neighbors(iid, k=k)
        raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)

        df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])
        df = df.merge(AnimesDF[['anime_id', 'title', 'genre', 'score']], left_on='Anime_ID', right_on='anime_id', how='left')

        df["Title"] = "[" + df["title"] + "](https://myanimelist.net/anime/" + df['Anime_ID'].astype(str) + "/" + df['title'].str.replace(' ', '_') + ")"

        table_md = "| Title | Genre | Score |\n| --- | --- | --- |\n"
        for i, row in df[:10].iterrows():
            table_md += f"| {row['Title']} | {row['Genre']} | {row['Score']} |\n"

        st.markdown("## Top Recommendations")
        st.markdown(table_md)

    except Exception as e:
        st.write("An error occurred during the processing. Please try again.")
# Set up the Streamlit interface
st.title('Weeaboo Wonderland')
st.write("""
Please enter an anime title in the input box below and hit 'Enter' on your keyboard.
You'll be presented with a list of recommended anime based on your input.
You can click on the title of any anime in the recommendations to go to its webpage.
""")

anime_title = st.text_input('Please enter an anime title')

# Only run the recommendation function if the user has entered something
if anime_title:
    get_item_recommendations(loaded_model, loaded_knn_model, anime_title)
