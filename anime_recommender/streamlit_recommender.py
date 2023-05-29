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

# Check if the input is empty or consists of only spaces
if not anime_title or anime_title.isspace():
    st.write("Enter something you neckbeard! There's no empty anime name.")
else:
    get_item_recommendations(loaded_model, loaded_knn_model, anime_title)
