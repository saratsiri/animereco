import pandas as pd
from collections import defaultdict
from scipy.sparse.linalg import svds
import surprise as sp
import pickle
import os

print("Loading data...\n")
# Load preprocessed data
ScoresDFHotStart = pd.read_csv('../processed_data/ScoresDFHotStart.csv')
print("Loading data completed\n")
# Initialize reader and load data into Surprise dataset format
reader = sp.Reader(rating_scale=(0, 10))
data = sp.Dataset.load_from_df(ScoresDFHotStart[['username', 'anime_id', 'my_score']], reader)

# Split data into training and test set
trainset, testset = sp.model_selection.train_test_split(data, test_size=.25)

# Dictionary for analysis (unused in this snippet)
analysis = defaultdict(list)

# Train a BaselineOnly model
trainset_full = data.build_full_trainset()
baseline_model = sp.BaselineOnly()
baseline_model.fit(trainset_full)

# Generate anti-testset and make predictions
anti_testset = trainset_full.build_anti_testset()
print("Training Baseline Model...\n")
predictions = baseline_model.test(anti_testset)
print("Baseline Model Completed\n")

# Convert predictions to dataframe and drop unnecessary column
predictions_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
predictions_df.drop('rui', axis=1, inplace=True)

# Train a KNNBaseline model
print("Training KNN model\n")
sim_options = {'name': 'pearson_baseline', 'user_based': False}
knn_model = sp.KNNBaseline(sim_options=sim_options)
knn_model.fit(trainset_full)
print("KNN Model Completed")
# Save models to pickle files
print("Saving..\n")
with open(f"{os.getcwd()}/trained_models/baseline_model.pickle", 'wb') as f:
    pickle.dump(baseline_model, f)
with open(f"{os.getcwd()}/trained_models/knn_model.pickle", 'wb') as f:
    pickle.dump(knn_model, f)
print("Saving Completed\n")
