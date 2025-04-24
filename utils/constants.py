"""This file contains constant variables."""

from typing import List

# data_config
data_csv_path: str = "data/spotify_reviews.csv"
features_to_use: List[str] = ["Time_submitted", "Review"]

# app_config
reviews_processed: int = 34050  # set to -1 if all are processed
analysis_report_path: str = "app/static/analysis_report.json"
plot_dir: str = "app/static/plots"
review_level_analysis_img_path: str = "app/static/review-level-sentiment.jpg"
entity_level_analysis_img_path: str = "app/static/entity-level-sentiment.png"
hld_img_path: str = "app/static/high_level_design.png"

# analyzer_config
model: str = "gemini-2.0-flash"
aggregated_results_path: str = "src/aggregated_results.json"