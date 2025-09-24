"""This file processes the SemEval-2014(ABSA) dataset by merging multiple aspect-polarity pairs per review into single entries, consolidating all aspects and their sentiments into one record per review."""

import argparse
import collections
import os

import pandas as pd

from utils import analyzer_utils

logger = analyzer_utils.Logger("Review Analyzer").get_logger()


def prepare_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare SemEval dataset.

    In the original format, each aspect-polarity pair for a review appears as a separate row.
    This function groups all entries by review and creates a new column `Aspects` that contains 
    a list of dictionaries with aspect-polarity pairs.
    
    Args:
        data_df (pd.DataFrame) : Dataframe containing raw data.
    Returns:
        prepared_data_df (pd.DataFrame) : Transformed DataFrame with one row per review and an aggregated `Aspects` column.
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


def main():
    """Parses command-line arguments, processes raw data, and saves the prepared output."""

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

    features_to_use = ["id", "Review", "aspect", "polarity"]
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(
            f"Unable to load data, file not found at {args.file_path}")

    data = pd.read_csv(args.file_path, skiprows=1, names=features_to_use)

    logger.info("Preparing data..")
    prepared_data = prepare_data(data)
    logger.info("Data preparation comleted succefully.")

    prepared_data.to_csv(args.save_path)
    logger.info(f"Prepared data saved to {args.save_path}")


if __name__ == "__main__":
    main()
