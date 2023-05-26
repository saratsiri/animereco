import streamlit as st
import pandas as pd
import pickle
from google.cloud import storage
from io import StringIO
import json
from google.oauth2.service_account import Credentials
import re

# Parse the secrets to a dictionary
creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

# Create a Credentials object from the dictionary
creds = Credentials.from_service_account_info(creds_dict)

# Instantiate a Google Cloud Storage client with the credentials
storage_client = storage.Client(credentials=creds)

@st.cache_data  # Use st.cache_data for data-like objects
def load_csv_from_gcs(bucket_name, blob_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    s = blob.download_as_text()
    return pd.read_csv(StringIO(s))

@st.cache_resource  # Use st.cache_resource for resource-like objects
def load_pickle_from_gcs(bucket_name, blob_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    s = blob.download_as_bytes()
    return pickle.loads(s)

bucket_name = "anime-reco"

AnimesDF = load_csv_from_gcs(bucket_name, "anime_cleaned.csv")
loaded_model = load_pickle_from_gcs(bucket_name, "baseline_model.pickle")
loaded_knn_model = load_pickle_from_gcs(bucket_name, "knn_model.pickle")

import re

def get_item_recommendations(algo, algo_items, anime_title, anime_id=100000, k=20):
    try:
        anime_title = anime_title.strip().lower()

        # Create regex pattern
        pattern = '.*'.join(anime_title)  # Converts 'slamdunk' to 's.*l.*a.*m.*d.*u.*n.*k'
        regex = re.compile(pattern)  # Compiles a regex pattern which can match any characters between the letters of 'slamdunk'

        # Check if the regex pattern is in the title
        matching_animes = AnimesDF[AnimesDF['title'].str.lower().apply(lambda x: bool(regex.search(x)))]

        # If no results, search by the English title
        if matching_animes.empty:
            matching_animes = AnimesDF[AnimesDF['title_english'].str.lower().apply(lambda x: bool(regex.search(x)))]

        if matching_animes.empty:
            st.write("No matching anime found. Please check your input.")
            return

        # If there are multiple matches, select the best one
        if len(matching_animes) > 1:
            st.write("Assuming you meant: ", matching_animes.iloc[0]['title'])
            anime_id = matching_animes.iloc[0]['anime_id']
        else:
            anime_id = matching_animes['anime_id'].iloc[0]

        iid = algo_items.trainset.to_inner_iid(anime_id)
        neighbors = algo_items.get_neighbors(iid, k=k)
        raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)

        df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])
        df = pd.merge(df, AnimesDF, left_on = 'Anime_ID', right_on = 'anime_id', how = 'left')

        df["Title"] = df.apply(lambda row: f"[{row['title']}](https://myanimelist.net/anime/{row['Anime_ID']}/{row['title'].replace(' ', '_')})", axis=1)
        df["Genre"] = df["genre"]
        df["Score"] = df["score"]

        # Create markdown tables
        table_md = "| Title | Genre | Score |\n| --- | --- | --- |\n"
        for i, row in df[:10].iterrows():
            table_md += f"| {row['Title']} | {row['Genre']} | {row['Score']} |\n"

        st.markdown("## Top Recommendations")
        st.markdown(table_md)

    except Exception as e:
        st.write("Stop trying to debug this code, NECKBEARD!")

# Set up the Streamlit interface
st.title('Weeaboo Wonderland')
st.write("""
Please enter an anime title in the input box below and hit 'Enter' on your keyboard.
You'll be presented with a list of recommended anime based on your input.
You can click on the title of any anime in the recommendations to go to its webpage.
""")

anime_title = st.text_input('Please enter an anime title')

if anime_title:
    get_item_recommendations(loaded_model, loaded_knn_model, anime_title)
