"""
Loading raw IMDB review files and preparing train/val/test DataFrames.
"""

import os
import pandas as pd


def load_dataset(folder):
    """
    Reads all text files in the specified folder and returns their content as a list.

    Args:
        folder (str): Path to the folder containing text files.

    Returns:
        list: A list of strings, where each string is the content of a text file.
    """
    reviews = []

    for filename in os.listdir(folder):
        if filename.endswith("txt"):
            file_path = os.path.join(folder, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                reviews.append(file.read())
    return reviews


def build_dataframes(train_pos_path, train_neg_path, test_pos_path, test_neg_path):
    """
    Loads positive/negative reviews for train and test splits and assembles
    them into labeled pandas DataFrames.

    Args:
        train_pos_path (str): Path to training positive reviews.
        train_neg_path (str): Path to training negative reviews.
        test_pos_path (str): Path to testing positive reviews.
        test_neg_path (str): Path to testing negative reviews.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): (train_df, test_df)
    """
    train_pos = load_dataset(train_pos_path)
    train_neg = load_dataset(train_neg_path)
    test_pos = load_dataset(test_pos_path)
    test_neg = load_dataset(test_neg_path)

    train_df = pd.DataFrame({
        "review": train_pos + train_neg,
        "label": [1] * len(train_pos) + [0] * len(train_neg)
    })

    test_df = pd.DataFrame({
        "review": test_pos + test_neg,
        "label": [1] * len(test_pos) + [0] * len(test_neg)
    })

    return train_df, test_df


def train_val_split(train_df, train_frac=0.9, random_state=42):
    """
    Shuffles train_df and splits it into training and validation subsets.

    Args:
        train_df (pd.DataFrame): Full training DataFrame.
        train_frac (float): Fraction of data to keep for training.
        random_state (int): Seed for reproducible shuffling.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): (train_data, val_data)
    """
    train_size = int(train_frac * len(train_df))
    shuffled_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_data = shuffled_df.iloc[:train_size]
    val_data = shuffled_df.iloc[train_size:]
    return train_data, val_data
