import os
import json
import logging
import colorlog
import pandas as pd
from copy import deepcopy
from typing import List

class Logger:
    """Logger with support for colored outputs"""

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
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


logger = Logger("Review Analyzer").get_logger()

def load_csv(file_path: str, columns: List[str]=[], reviews_processed: int=None):
    """
    Loads data from a CSV file.
    
    Args:
        file_path (str): path to csv file.
        columns List[str]: List of names of particular columns that needs to be extracted. All will be used by default.
        reviews_processed (int, optional): Number of reviews proccessed. Loads all if no value is passed.
    
    Returns:
        df (pd.DataFrame): Dataframe containing all processed reviews
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not load csv, invalid path : {file_path}")
    
    if columns:
        df = pd.read_csv(file_path, usecols=columns)
    else:
        df = pd.read_csv(file_path)    

    if reviews_processed:
        df = df[:reviews_processed]
    
    return df

def load_analysis_report(file_path:str)->dict:
    """
    Loads the json report

    Args:
        file_path (str): path to analysis report (json file).

    Returns:
        report (dict): dict containing parsed json data     
    """

    try:
        with open(file_path, "r") as f:
            report = json.load(f)
            report = report[0]
    except FileNotFoundError:
        logger.warning(f"Invalid path for Analysis report: {file_path}")
        return {}

    # Convert lists to set inside each dictionary
    for entity_name, data in report.items():
        for key in ["positive_reviews", "negative_reviews"]:
            data[key]["ids"] = set(data[key]["ids"]) 
    return report          

def save_checkpoint(checkpoint, file_path):
    with open(file_path, "w") as f:
        json.dump(checkpoint, f, indent=4)

def load_checkpoint(file_path:str)->dict:
    """
    Loads the saved json checkpoint

    Args:
        file_path (str): path to chackpoint (json file).

    Returns:
        report (dict): dict containing parsed json data     
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Unable to find previous chackoint at {file_path}, starting from scratch.")
        return {"batch_size":None, "last_batch_idx": None, "existing_entities": []}
    
def save_analysis_report(aggregated_results:dict, file_path:str):
    """
    Saves results dictionary as report json file.

    Args:
       aggregated_results (dict): A dictionary where each key is an entity and the value is a dictionary with 
        "positive_reviews" and "negative_reviews" as keys mapping to lists of review IDs.
       file_path (str): path to save json report 
    """
    result = deepcopy(aggregated_results)
    # Convert sets to lists inside each dictionary
    for entity_name, data in result.items():
        for key in ["positive_reviews", "negative_reviews"]:
            data[key]["ids"] = list(data[key]["ids"])  # Convert sets to lists
    with open(file_path, "w") as f:
        json.dump([result], f, indent=4)

def analyze_coverage(data:pd.DataFrame, report:dict):
    """
    Extract coverage information from Analysis Report.

    args:
        data (pd.DataFrame): Dataframe containing all processed reviews
        report (dict): A dictionary where each key is an entity and the value is a dictionary with 
        "positive_reviews" and "negative_reviews" as keys mapping to lists of review IDs.

    Returns:
        coverage_report (dict):
            total_reviews (int): number of reviews processed.
            unattended_reviews (pd.DataFrame): Dataframe containing reviews for which no entity was assigned.
    """
    entity_mentions = set()
    for entity, details in report.items():
        entity_mentions.update(details["positive_reviews"]["ids"])
        entity_mentions.update(details["negative_reviews"]["ids"])
    
    reviews = pd.DataFrame(data["Review"])
    
    all_review_ids = set([i for i in range(len(reviews))])
    reviews_ids_with_no_entities = list(all_review_ids.difference(entity_mentions))
    
    reviews_with_no_entities = reviews.iloc[reviews_ids_with_no_entities]
    reviews_with_no_entities.index = range(1, len(reviews_with_no_entities)+1)
    
    coverage_report = {
        "total_reviews":len(all_review_ids),
        "unattended_reviews": reviews_with_no_entities
    }

    return coverage_report

def get_reviews_for_entity(data:pd.DataFrame, report:dict, entity_name:str, sentiment:str="positive"):
    """
    Fecthes reviews assigned to a particular entity-sentiment group

    Args:
        data (pd.DataFrame): Dataframe containing all processed reviews
        report (dict): A dictionary where each key is an entity and the value is a dictionary with 
        "positive_reviews" and "negative_reviews" as keys mapping to lists of review IDs.
        entity_name (str): Name of entity

    Returns:
        selected_reviews(pd.DataFrame): DataFrame containing only Reviews assigned to given entity-sentiment group.
    """
    sentiment += "_reviews"
    review_ids = report[entity_name][sentiment]["ids"]
    selected_reviews = data.iloc[sorted(review_ids)]["Review"]
    selected_reviews.index = range(1, len(selected_reviews) + 1)
    return selected_reviews