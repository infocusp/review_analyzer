"""This file contains utility functions for plotting."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import data_models


#  Entity Frequency (Top Entities)
def plot_entity_frequency(report: data_models.AggregatedResults,
                          top_k: int = 10,
                          save_path: str = "./entity_frequency.png") -> None:
    """Generates and saves a bar plot of the most frequently mentioned entities in the review analysis report.

    Args:
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            a dictionary containing sets of review IDS corresponnding to each sentiment.
        top_k (int, optional): The number of top entities to include in the plot. Defaults to 10.
        save_path (str, optional): File path where the plot image will be saved. Defaults to './entity_frequency.png'.
    
    Returns:
        None
    """
    entity_counts = {
        entity: len(sentiment_map["positive_review_ids"]) +
        len(sentiment_map["negative_review_ids"])
        for entity, sentiment_map in report.items()
    }
    sorted_entities = sorted(entity_counts.items(),
                             key=lambda x: x[1],
                             reverse=True)[:top_k]  # Top 20 entities

    x = [e[1] for e in sorted_entities]
    y = [e[0] for e in sorted_entities]

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x=x, y=y, palette="viridis")

    # Annotate bars with values
    for index, value in enumerate(x):
        ax.text(value + 1,
                index,
                str(value),
                va='center',
                fontsize=12,
                color='black')

    plt.xlabel("Total Mentions")
    plt.ylabel("Entity")
    plt.title(f"Top {top_k} Most Mentioned Entities")
    plt.savefig(save_path)
    plt.close()


# Review length vs Entities extracted
def plot_review_length_vs_entities_violin(
        reviews: List[str],
        report: data_models.AggregatedResults,
        save_path: str = "./review_length_vs_entities_violin.png") -> None:
    """Generates and saves a violin plot showcasing the relationship between review length and number of entities extracted.

    Args:
        reviews (List[str]): List of reviews.
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            a dictionary containing sets of review IDS corresponnding to each sentiment.
        save_path (str, optional): File path where the plot image will be saved. Defaults to './review_length_vs_entities_violin.png'.
    
    Returns:
        None
    """
    review_lengths = [len(review.split()) for review in reviews]
    entity_counts = []
    zero_count = 0
    for review_id in range(len(reviews)):
        count = sum(review_id in sentiment_map["positive_review_ids"] or
                    review_id in sentiment_map["negative_review_ids"]
                    for sentiment_map in report.values())
        entity_counts.append(count)
        if count == 0:
            zero_count += 1

    # Determine dynamic bins using percentiles
    num_bins = 8
    bin_edges = np.percentile(review_lengths, np.linspace(0, 100, num_bins + 1))
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        for i in range(len(bin_edges) - 1)
    ]

    # Assign reviews to bins
    bins = pd.cut(review_lengths,
                  bins=bin_edges,
                  labels=bin_labels,
                  include_lowest=True)
    df = pd.DataFrame({"Length Bin": bins, "Entity Count": entity_counts})

    plt.figure(figsize=(12, 8))
    sns.violinplot(x="Length Bin", y="Entity Count", data=df, inner="stick")
    plt.xlabel("Review Length (word bins)")
    plt.ylabel("Number of Entities Mentioned")
    plt.title("Review Length vs. Number of Entities (Violin Plot)")
    plt.savefig(save_path)
    plt.close()


# Sentiment Intensity Heatmap
def plot_sentiment_heatmap(report: data_models.AggregatedResults,
                           save_path: str = "./sentiment_heatmap.png") -> None:
    """Generates and saves a heatmap visualizing the distribution of positive and negative review counts for each extracted entity.

    Args:
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
           a dictionary containing sets of review IDS corresponnding to each sentiment.
            
        save_path (str, optional): File path where the heatmap image will be saved. Defaults to './sentiment_heatmap.png'.

    Returns:
        None
    """
    entity_names = []
    sentiment_scores = []

    for entity, sentiment_map in report.items():
        pos_count = len(sentiment_map["positive_review_ids"])
        neg_count = len(sentiment_map["negative_review_ids"])
        sentiment_intensity = (pos_count - neg_count) / max(
            (pos_count + neg_count), 1)  # Normalized score
        entity_names.append(entity)
        sentiment_scores.append(sentiment_intensity)

    df = pd.DataFrame({
        "Entity": entity_names,
        "Sentiment Score": sentiment_scores
    })
    df = df.pivot_table(index="Entity", values="Sentiment Score")
    df = df.sort_values(by="Sentiment Score", ascending=False)

    # Adjust figure size dynamically
    plt.figure(figsize=(12, max(8, len(df) * 0.4)))

    ax = sns.heatmap(df, annot=True, cmap="PiYG", linewidths=0.5, center=0)

    plt.title("Sentiment Intensity Heatmap")

    # Ensure text alignment
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # Add a separate legend in the top-right corner
    legend_labels = [
        ("+1 (Mostly Positive)", "green"),
        ("0 (Neutral)", "white"),
        ("-1 (Mostly Negative)", "purple"),
    ]

    legend_patches = [
        plt.Line2D([0], [0],
                   color=color,
                   marker='s',
                   markersize=12,
                   linestyle="") for _, color in legend_labels
    ]
    plt.legend(legend_patches, [label for label, _ in legend_labels],
               loc="upper right",
               bbox_to_anchor=(1.5, 1.0),
               title="Sentiment Score",
               fontsize=12,
               title_fontsize=12)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_sentiment_trend(entity_name: str,
                         data_df: pd.DataFrame,
                         report: data_models.AggregatedResults,
                         save_path: str = "./trend.png",
                         time_interval: str = "D") -> None:
    """Generates and saves a plots displaying the trend of sentiments over time for a given entity.

    Args:
        entity_name (str): The entity to visualize (e.g., "Notifications").
        report (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            a dictionary containing sets of review IDS corresponnding to each sentiment.
        data_df (pd.DataFrame): DataFrame containing "Time_submitted" and review IDs.
        save_path (str, optional): File path where the heatmap image will be saved. Defaults to './trend.png'.
        time_interval(str, optional): Time aggregation interval ("D" for daily, "W" for weekly).
    
    Returns:
        None
    """

    # Convert time column to datetime
    data_df["Time_submitted"] = pd.to_datetime(data_df["Time_submitted"])

    # Extract sentiment review IDs from JSON
    sentiment_map = report[entity_name]
    pos_review_ids = sentiment_map["positive_review_ids"]
    neg_review_ids = sentiment_map["negative_review_ids"]

    # Filter dataframe for matching review IDs
    df_pos = data_df[data_df.index.isin(pos_review_ids)].copy(
    ) if pos_review_ids else pd.DataFrame(columns=data_df.columns)
    df_neg = data_df[data_df.index.isin(neg_review_ids)].copy(
    ) if neg_review_ids else pd.DataFrame(columns=data_df.columns)

    # Group by time and count occurrences
    if not df_pos.empty:
        df_pos["sentiment"] = "Positive"
    if not df_neg.empty:
        df_neg["sentiment"] = "Negative"
    df_trend = pd.concat([df_pos, df_neg])
    df_trend = df_trend.groupby(
        [pd.Grouper(key="Time_submitted", freq=time_interval),
         "sentiment"]).size().unstack(fill_value=0)

    # Plot sentiment trends
    plt.figure(figsize=(12, 15))
    if not df_pos.empty:
        plt.plot(df_trend.index,
                 df_trend.get("Positive", 0),
                 label="Positive Sentiment",
                 color="green",
                 marker="o")
    if not df_neg.empty:
        plt.plot(df_trend.index,
                 df_trend.get("Negative", 0),
                 label="Negative Sentiment",
                 color="red",
                 marker="o")

    plt.xlabel("Time")
    plt.ylabel("Review Count")
    plt.title(f"Sentiment Trend for {entity_name}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()
