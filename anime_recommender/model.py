import pandas as pd
import pickle
import os
AnimesDF = pd.read_csv(f'{os.getcwd()}/raw_data/anime_cleaned.csv')

def get_item_recommendations(algo, algo_items, anime_title, anime_id=100000, k=10):
    # Clean up the input
    anime_title = anime_title.strip().lower()

    # Try to get the anime_id from the title
    matching_animes = AnimesDF[AnimesDF['title'].str.lower() == anime_title]
    if matching_animes.empty:
        print("No matching anime found. Please check your input.")
        return None

    # If no specific anime_id is provided, use the id of the first matching anime
    if anime_id == 100000:
        anime_id = matching_animes['anime_id'].iloc[0]

    iid = algo_items.trainset.to_inner_iid(anime_id)
    neighbors = algo_items.get_neighbors(iid, k=k)
    raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)
    print("Here's a list of anime titties you might enjoy")
    df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])
    df = pd.merge(df, AnimesDF, left_on = 'Anime_ID', right_on = 'anime_id', how = 'left')

    return df[['Anime_ID', 'title', 'genre']]


# Load the model from a file
with open(f"{os.getcwd()}/anime_recommender/trained_models/baseline_model.pickle", 'rb') as f:
    loaded_model = pickle.load(f)

# Load the KNNBaseline model
with open(f"{os.getcwd()}/anime_recommender/trained_models/knn_model.pickle", 'rb') as f:
    loaded_knn_model = pickle.load(f)

anime_title = input("Please enter an anime title: ")

print(get_item_recommendations(loaded_model, loaded_knn_model, anime_title))
