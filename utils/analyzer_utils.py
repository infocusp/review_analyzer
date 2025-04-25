"""This file contains a logger class and utility functions."""

import json
import logging
import os
from typing import Dict, List

import colorlog
import pandas as pd

from utils import data_models


class Logger:
    """Logger with support for colored outputs."""

    def __init__(self, name: str = "app", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs

        # Define log colors
        log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }

        # Create a handler with color support
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                log_colors=log_colors,
                datefmt="%Y-%m-%d %H:%M:%S"))

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


logger = Logger("Review Analyzer").get_logger()


def load_csv(file_path: str,
             columns: List[str] = [],
             reviews_processed: int = -1) -> pd.DataFrame:
    """Loads data from a CSV file.
    
    Args:
        file_path (str): path to csv file.
        columns List[str]: List of names of particular columns that needs to be extracted. All will be used by default.
        reviews_processed (int, optional): Number of reviews proccessed. Loads all if no value is passed.
    
    Returns:
        df (pd.DataFrame): Dataframe containing all processed reviews
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not load csv, invalid path : {file_path}")

    kwargs: Dict[str, int | List[str]] = {}
    if reviews_processed > 0:
        kwargs['nrows'] = reviews_processed
    if columns:
        kwargs['usecols'] = columns

    df = pd.read_csv(file_path, **kwargs)
    return df


def analyze_coverage(data: pd.DataFrame,
                     report: data_models.AggregatedResults) -> Dict:
    """Extract coverage information from Analysis Report.

    args:
        data (pd.DataFrame): Dataframe containing all processed reviews
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            a dictionary containing sets of review IDS corresponnding to each sentiment.

    Returns:
        coverage_report (Dict):
            total_reviews (int): number of reviews processed.
            unattended_reviews (pd.DataFrame): Dataframe containing reviews for which no entity was assigned.
    """
    entity_mentions = set()
    for entity, sentiment_map in report.items():
        entity_mentions.update(sentiment_map["positive_review_ids"])
        entity_mentions.update(sentiment_map["negative_review_ids"])

    reviews = pd.DataFrame(data["Review"])

    all_review_ids = set([i for i in range(len(reviews))])
    reviews_ids_with_no_entities = list(
        all_review_ids.difference(entity_mentions))

    reviews_with_no_entities = reviews.iloc[reviews_ids_with_no_entities]
    reviews_with_no_entities.index = range(1, len(reviews_with_no_entities) + 1)

    coverage_report = {
        "total_reviews": len(all_review_ids),
        "unattended_reviews": reviews_with_no_entities
    }

    return coverage_report


def get_reviews_for_entity(data: pd.DataFrame,
                           report: data_models.AggregatedResults,
                           entity_name: str,
                           sentiment: str = "positive") -> pd.DataFrame:
    """Fecthes reviews assigned to a particular entity-sentiment group.

    Args:
        data (pd.DataFrame): Dataframe containing all processed reviews
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
             a dictionary containing sets of review IDS corresponnding to each sentiment.
        entity_name (str): Name of entity to be queried
        sentiment (str): Sentiment to be queried

    Returns:
        selected_reviews(pd.DataFrame): DataFrame containing only Reviews assigned to given entity-sentiment group.
    """
    review_ids = report[entity_name][f"{sentiment}_review_ids"]
    selected_reviews = data.iloc[sorted(review_ids)]["Review"]
    selected_reviews.index = range(1, len(selected_reviews) + 1)
    return selected_reviews


def read_json(file_path: str) -> Dict:
    """Loads json file to python dict.

    Args:
        file_path (str): path to json file.

    Resturns:
        data (Dict): python dict with loaded json data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
