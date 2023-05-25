import streamlit as st
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader

from collections import defaultdict
from operator import itemgetter
import heapq
import pandas as pd
import os

# load numpy array from npy file
from numpy import load

# Page configuration
st.set_page_config(page_title='Anime Recommendation', layout='centered')

@st.cache_data
def load_similarity_matrix():
    # load array
    # Specify the file path
    file_path=f"{os.getcwd()}/processed_data/similarity_matrix_full.npy"
    # load array
    similarity_matrix = load(file_path)
    return similarity_matrix
similarity_matrix=load_similarity_matrix()

@st.cache_data
def load_anime():
    AnimesDF = pd.read_csv(f"{os.getcwd()}/raw_data/anime_cleaned.csv")
    animeID_to_name = AnimesDF.set_index('anime_id')['title'].to_dict()
    return animeID_to_name
animeID_to_name=load_anime()

@st.cache_data
def load_score():
    ScoresDF = pd.read_csv(f"{os.getcwd()}/raw_data/animelists_cleaned.csv")
    ScoresDF_selected = ScoresDF[ScoresDF["my_score"] > 0][["username", "anime_id", "my_score", "my_last_updated"]]
    return ScoresDF_selected
ScoresDF_selected=load_score()

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
