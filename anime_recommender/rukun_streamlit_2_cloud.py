import streamlit as st
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader

from collections import defaultdict
from operator import itemgetter
import heapq
import pandas as pd
import os
import numpy as np

import pickle
from google.cloud import storage
from io import StringIO
import json
from google.oauth2.service_account import Credentials

# load numpy array from npy file
from numpy import load

# Page configuration
st.set_page_config(page_title='Anime Recommendation', layout='centered')

# Parse the secrets to a dictionary
creds_dict = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

# Create a Credentials object from the dictionary
creds = Credentials.from_service_account_info(creds_dict)

# Instantiate a Google Cloud Storage client with the credentials
storage_client = storage.Client(credentials=creds)


@st.cache_data
def load_similarity_matrix():
    # Specify the bucket and file name for similarity matrix
    bucket_name = "animerec"
    file_name = "similarity_matrix_full.npy"

    # Load the .npy file from Google Cloud Storage
    similarity_blob = load_npy_from_gcs(bucket_name, file_name)
    similarity_matrix = np.load(similarity_blob.download_as_bytes())
    return similarity_matrix

@st.cache_data
def load_anime():
    # Specify the bucket and file name for anime dataset
    bucket_name = "animerec"
    file_name = "anime_cleaned.csv"

    # Load the CSV file from Google Cloud Storage
    anime_blob = load_csv_from_gcs(bucket_name, file_name)
    AnimesDF = pd.read_csv(anime_blob)
    animeID_to_name = AnimesDF.set_index('anime_id')['title'].to_dict()
    return animeID_to_name

@st.cache_data
def load_score():
    # Specify the bucket and file name for scores dataset
    bucket_name = "animerec"
    file_name = "animelists_cleaned.csv"

    # Load the CSV file from Google Cloud Storage
    scores_blob = load_csv_from_gcs(bucket_name, file_name)
    ScoresDF = pd.read_csv(scores_blob)
    ScoresDF_selected = ScoresDF[ScoresDF["my_score"] > 0][["username", "anime_id", "my_score", "my_last_updated"]]
    return ScoresDF_selected

@st.cache_data
def load_csv_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    csv_data = blob.download_as_text()
    return pd.compat.StringIO(csv_data)

@st.cache_data
def load_npy_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    npy_data = np.load(blob.download_as_bytes())
    return npy_data

# Load the data using the respective functions
similarity_matrix = load_similarity_matrix()
animeID_to_name = load_anime()
ScoresDF_selected = load_score()



@st.cache_data
def load_trainset():
    reader = Reader(rating_scale=(0, 10))
    scoredata = Dataset.load_from_df(ScoresDF_selected[['username', 'anime_id', 'my_score']], reader)
    trainset = scoredata.build_full_trainset()
    return trainset
trainset=load_trainset()



# Sidebar
st.sidebar.title('Anime Recommendation')
test_subject = st.sidebar.text_input('Enter your username')
submit_button = st.sidebar.button('Get Recommendations')

if submit_button:
    # Get the top K items user rated
    k = 20

    test_subject_iid = trainset.to_inner_uid(test_subject)
    test_subject_ratings = trainset.ur[test_subject_iid]
    k_neighbors = heapq.nlargest(k, test_subject_ratings, key=lambda t: t[1])

    candidates = defaultdict(float)

    for itemID, rating in k_neighbors:
        try:
            similaritities = similarity_matrix[itemID]
            for innerID, score in enumerate(similaritities):
                candidates[innerID] += score * (rating / 5.0)
        except:
            continue

    # Utility function to get anime name from animeID
    def getAnimeName(animeID):
        if int(animeID) in animeID_to_name:
            return animeID_to_name[int(animeID)]
        else:
            return ""

    # Build a dictionary of anime the user has watched
    watched = {}
    for itemID, rating in trainset.ur[test_subject_iid]:
        watched[itemID] = 1

    # Add items to list of user's recommendations
    # If they are similar to their favorite anime,
    # AND have not already been watched.
    recommendations = []

    position = 0
    for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            recommendations.append(getAnimeName(trainset.to_raw_iid(itemID)))
            position += 1
            if (position > 10):
                break  # We only want top 10

    # Display recommendations
    if len(recommendations) > 0:
        st.header('Anime Recommendations')
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.info("No recommendations found for the given user.")
