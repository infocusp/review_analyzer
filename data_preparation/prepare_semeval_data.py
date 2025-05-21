"""This file prepares SemEval datasets."""

import argparse
import collections
import os

import pandas as pd

from utils import analyzer_utils

Mylogger = analyzer_utils.Logger("Review Analyzer")
logger = Mylogger.get_logger()


def prepare_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare SemEval dataset.
    
    args:
        data_df (pd.DataFrame) : Dataframe containing raw data.
    returns:
        prepared_data_df (pd.DataFrame) : Dataframe containing prepared data.
    """
    grouped = collections.defaultdict(list)

    for _, row in data_df.iterrows():
        id = row['id']
        review = row['Review']
        aspect = row['aspect']
        polarity = row['polarity']

        key = (id, review)
        grouped[key].append({"aspect": aspect, "polarity": polarity})

    grouped_data = []
    for (id, review), aspect_list in grouped.items():
        grouped_data.append({
            'id': id,
            'Review': review,
            'Aspects': aspect_list
        })

    prepared_data_df = pd.DataFrame(grouped_data)
    return prepared_data_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SemEval dataset for ReviewAnalyzer")
    parser.add_argument("--file_path",
                        type=str,
                        required=True,
                        help="path to the csv dataset file")
    parser.add_argument("--save_path",
                        type=str,
                        required=True,
                        help="path to save prepared data")

    args = parser.parse_args()

    features_to_use = ["id", "Review", "aspect", "polarity", "from", "to"]
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(
            f"Unable to load data, file not found at {args.file_path}")

    data = pd.read_csv(args.file_path, skiprows=1, names=features_to_use)

    logger.info("Preparing data..")
    prepared_data = prepare_data(data)
    logger.info("Data preparation comleted succefully.")

    prepared_data.to_csv(args.save_path)
    logger.info(f"Prepared data saved to {args.save_path}")
