"""This file processes Amazon reviews for a specific category by grouping reviews sharing the same parent_asin and saves the top 20 groups as individual datasets."""

import argparse
import json
import os
from typing import List

import pandas as pd

from utils import analyzer_utils

Mylogger = analyzer_utils.Logger("Review Analyzer")
logger = Mylogger.get_logger()


def jsonl_to_dataframe(file_path: str, chunk_size: int = 10000):
    """Yields chunks from jsonl data.

    Args:
        file_path (str): path to jsonl file.
        chunk_size (int): size of a single chunk.
    
    Returns:
        (pd.DataFrame): dataframe consisting `chunk-size` of data.
    
    """
    data_chunk = []
    for i, line in enumerate(open(file_path, 'r')):
        data_chunk.append(json.loads(line))

        # Once chunk_size is hit, yield the chunk
        if (i + 1) % chunk_size == 0:
            yield pd.DataFrame(data_chunk)
            data_chunk = []

    # Yield any remaining lines
    if data_chunk:
        yield pd.DataFrame(data_chunk)


def load_data(file_path: str,
              features_to_use: List[str],
              chunk_size: int = 10000) -> pd.DataFrame:
    """Loads newline-delimited JSON file into a DataFrame by chunking the bulk data.
    
    Args:
        file_path (str): path to jsonl file.
        features_to_use (List[str]): List of names of particular columns that needs to be extracted.
        chunk_size (int): size of chunk.
    
    Returns:
        data_df (pd.DataFrame): Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Unable to load data, file not found at {file_path}")
    df_chunks = []
    for chunk in jsonl_to_dataframe(file_path, chunk_size):
        df_chunks.append(chunk)
        if len(df_chunks) >= 50:
            break

    # Combine all chunks
    data_df = pd.concat(df_chunks, ignore_index=True)
    return data_df[features_to_use]


def load_meta_data(file_path: str, features_to_use: List[str]) -> pd.DataFrame:
    """Load newline-delimited JSON file into a DataFrame
    
    Args:
        file_path (str): path to jsonl file.
        features_to_use (List[str]): List of names of particular columns that needs to be extracted.
        
    Returns:
        meta_df (pd.DataFrame): Loaded meta data.
    
    Raise:
        FiFileNotFoundError: if provided file path is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Unable to load meta data, file not found at {file_path}")
    meta_df = pd.read_json(file_path, lines=True)
    return meta_df[features_to_use]


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Prepare amzon-review dataset for ReviewAnalyzer")
    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--review_filename",
                        type=str,
                        required=True,
                        help="name of json file containing review data")
    parser.add_argument("--meta_filename",
                        type=str,
                        required=True,
                        help="name of json file containing meta data")

    args = parser.parse_args()
    data_dir = args.data_dir
    data_filename = args.data_filename
    metadata_filename = args.metadata_filename

    # load review data
    logger.info("loading data")
    data_df = load_data(
        file_path=os.path.join(data_dir, data_filename),
        features_to_use=['title', 'text', 'asin', 'parent_asin'])
    logger.info("loaded data")

    # load meta data
    logger.info("loading  metadata")
    meta_df = load_meta_data(
        file_path=os.path.join(data_dir, metadata_filename),
        features_to_use=['main_category', 'title', 'parent_asin'])
    logger.info("loaded metadata")

    # group reviews by `parent_asin`
    logger.info("grouping by parent asin")
    asin_groups = {
        asin: group for asin, group in data_df.groupby("parent_asin")
    }
    # sort the groups based on number of reviews
    sorted_asin_groups = dict(
        sorted(asin_groups.items(), key=lambda item: len(item[1]),
               reverse=True))
    logger.info("grouping completed succefully.")

    # Save the reviews for top 20 groups as individual datasets.
    logger.info(f"saving prepared data to {data_dir}")
    file = open(os.path.join(data_dir, "product_name.txt"), 'w')
    for idx, (parent_asin, group) in enumerate(sorted_asin_groups.items(),
                                               start=1):
        print(f"{'__'*30}\n")
        if idx == 20:
            break  # extracting reviews for top 20 products from a category
        product_name = meta_df.loc[meta_df["parent_asin"] == parent_asin,
                                   "title"].values[0]
        logger.info(
            f"\nPARENT ASIN = {parent_asin}\nproduct_name = {product_name}\nNumber of reviews = {len(group)}"
        )
        group = group.rename(columns={"text": "Review"})
        group.to_csv(f"{data_dir}/{idx}_{parent_asin}.csv")
        file.write(f"{idx}_{parent_asin} : {product_name} [{len(group)}]\n")
    file.close()


if __name__ == "__main__":
    main()
