import pandas as pd #required pandas == 1.4.4
import os

#Importing the CSVs to Dataframe format
UsersDF = pd.read_csv(f'{os.getcwd()}/raw_data/users_cleaned.csv')
AnimesDF = pd.read_csv(f'{os.getcwd()}/raw_data/anime_cleaned.csv')
ScoresDF = pd.read_csv(f'{os.getcwd()}/raw_data/animelists_cleaned.csv')

#Since ScoresDF is a huge DF (2GB of data) I`ll only take the columns that are important for the recommendation system
ScoresDF = ScoresDF[['username', 'anime_id', 'my_score', 'my_status']]

#Analysing all the possible values for the score, this will be used as a parameter later on
lower_rating = ScoresDF['my_score'].min()
upper_rating = ScoresDF['my_score'].max()

#Counting how many relevant scores each user have done, resetting the index (so the series could become a DF again) and changing the column names
UsersAndScores = ScoresDF['username'].value_counts().reset_index().rename(columns={"username": "animes_rated", "index": "username"})

UsersSampled = UsersDF.sample(frac = 1) #, random_state = 2)

UsersAndScoresSampled = pd.merge(UsersAndScores, UsersSampled, left_on = 'username', right_on = 'username', how = 'inner')

#Grouping users whom had the same amount of animes rated
UserRatedsAggregated = UsersAndScoresSampled['animes_rated'].value_counts().reset_index().rename(columns={"animes_rated": "group_size", "index": "animes_rated"}).sort_values(by=['animes_rated'])

#Counting how many relevant scores each anime has, resetting the index (so the series could become a DF again) and changing the column names
RatedsPerAnime = ScoresDF['anime_id'].value_counts().reset_index().rename(columns={"anime_id": "number_of_users", "index": "anime_id"})

#Grouping users whom had the same amount of animes rated
AnimeRatedsAggregated = RatedsPerAnime['number_of_users'].value_counts().reset_index().rename(columns={"number_of_users": "group_size", "index": "number_of_users"}).sort_values(by=['number_of_users'])
#Creating a dataframe of users  and animes with more than 10 interactions
UserRatedsCutten = UsersAndScoresSampled[UsersAndScoresSampled['animes_rated'] >= 10]
AnimeRatedsCutten = RatedsPerAnime[RatedsPerAnime['number_of_users'] >= 10]
#Joining (merging) our new dataframes with the interactions one (this will already deal with the sample problem,
#as it is an inner join). The "HotStart" name comes from a pun about solving the "Cold Start" issue
ScoresDFHotStart = pd.merge(ScoresDF, UserRatedsCutten, left_on = 'username', right_on = 'username', how = 'inner')
ScoresDFHotStart = pd.merge(ScoresDFHotStart, AnimeRatedsCutten, left_on = 'anime_id', right_on = 'anime_id', how = 'inner')

# Save ScoresDFHotStart to a CSV file
ScoresDFHotStart.to_csv(f"{os.getcwd()}/processed_data/ScoresDFHotStart", index = False)
print("Data processing completed.")
