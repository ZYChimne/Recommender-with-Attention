import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


class MovielensDataset:
    def __init__(self):
        ratings = tfds.load("movielens/100k-ratings", split="train")
        ratings = ratings.map(lambda x: {
            "movie_id": x["movie_id"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
            "user_gender": int(x["user_gender"]),
            "user_zip_code": x["user_zip_code"],
            "user_occupation_text": x["user_occupation_text"],
            "bucketized_user_age": int(x["bucketized_user_age"]),
            # movie_genres, user_occupation_label
        })
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)
        feature_names = ["movie_id", "user_id", "user_gender", "user_zip_code",
                         "user_occupation_text", "bucketized_user_age"]
        vocabularies = {}
        for feature_name in feature_names:
            vocab = ratings.batch(1_000_000).map(lambda x: x[feature_name])
            vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))
        self.cached_train = train.shuffle(100_000).batch(8192)
        self.cached_test = test.batch(4096)
        self.vocabularies = vocabularies

    def get_data(self):
        return self.cached_train, self.cached_test, self.vocabularies
