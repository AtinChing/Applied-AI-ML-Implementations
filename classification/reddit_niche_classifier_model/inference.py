from tensorflow import keras
import pandas as pd
import numpy as np

# Load model
model = keras.models.load_model("niche_classifier_model.keras")

# Example input (replace this with real preprocessed data)
sample = pd.DataFrame({
    "score": [123],
    "num_comments": [45],
    "upvote_ratio": [0.92],
    "title_length": [84],
    "selftext_length": [203],
    "contains_question": [1],
    "contains_capslock": [0],
    "engagement_score": [0.85],
    "hour_of_posting": [15],
    "num_caps_words": [2],
    "subreddit_AskReddit": [1],
    "subreddit_relationships": [0],
    "subreddit_TIFU": [0],
    # add other one-hot subreddit features as needed
})

pred = model.predict(sample).argmax(axis=1)
print("Predicted class index:", pred)
