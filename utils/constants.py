"""This file contains constant variables."""

import os
from typing import List

data = {
    "spotify": "data/spotify_reviews.csv",
    "watch": "data/amazon_reviews/fashion/2_B09TXZHKLG.csv",
    "office_table": "data/amazon_reviews/office_products/2_B0C89HT5G5.csv",
    "dental_kit": "data/amazon_reviews/beauty/6_B08L5KN7X4.csv",
    "laptop": "data/SemEval/laptop_train.csv",
    "restaurant": "data/SemEval/restaurants_train.csv"
}
# data_config
dataset_name: str = "laptop"
data_csv_path: str = data[dataset_name]
features_to_use: List[str] = ["Review"]
result_dir: str = "results/"
experiment_name: str = "exp1"
result_subdir: str = os.path.join(result_dir,
                                  os.path.join(dataset_name, experiment_name))
debug_dir: str = os.path.join(result_subdir, "logs")

# analyzer_config
model: str = "gemini-2.0-flash"
batch_size: int = 50
aggregated_results_path: str = os.path.join(result_subdir,
                                            f"analysis_report.json")

# app_config
reviews_processed: int = -1  # set to -1 if all are processed
analysis_report_path: str = "app/static/analysis_report.json"
plot_dir: str = "app/static/plots"
review_level_analysis_img_path: str = "app/static/review-level-sentiment.jpg"
entity_level_analysis_img_path: str = "app/static/entity-level-sentiment.png"
hld_img_path: str = "app/static/high_level_design.png"
