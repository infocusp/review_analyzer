"""This file represents the `insights` page of the streamlit application"""

import json
import os

import pandas as pd
import streamlit as st

from utils import analyzer_utils
from utils import constants
from utils import data_models
from utils import plotting_utils

# Set page title
st.title("📈 Insights & Reports")
st.write(
    "Explore entity trends, sentiment distributions, and key patterns in the review data."
)

plot_dir = constants.plot_dir
os.makedirs(plot_dir, exist_ok=True)

plots = {
    "Top Entities Mentioned": {
        "file_path":
            "entity_frequency.png",
        "description":
            "Identifies the most frequently mentioned entities in reviews."
    },
    "Sentiment Intensity Heatmap": {
        "file_path":
            "sentiment_heatmap.png",
        "description":
            "Visualizes the intensity of sentiments are across all reviews."
    },
    "Trend over time": {
        "file_path": "trend.png",
        "description": "Shows the distribution of sentiments over time"
    }
}

# Load analysis report
with open(constants.analysis_report_path, "r") as f:
    raw = json.load(f)
    report = data_models.AggregatedResults.model_validate(raw)

data = analyzer_utils.load_csv(file_path=constants.data_csv_path,
                               columns=constants.features_to_use,
                               reviews_processed=constants.reviews_processed)

if "Time_Submitted" in data.columns:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Report", "Entity Frequency", "Sentiment Distrubution",
        "Trend over time"
    ])
else:
    tab1, tab2, tab3 = st.tabs(
        ["Report", "Entity Frequency", "Sentiment Distrubution"])

# Tab 1: Report
with tab1:
    # Convert dict to DataFrame
    df = pd.DataFrame([{
        "Entity": entity,
        "Positive": len(sentiment_map["positive_review_ids"]),
        "Negative": len(sentiment_map["negative_review_ids"]),
    } for entity, sentiment_map in report.items()])
    df["Total"] = df["Positive"] + df["Negative"]
    # Sort entities by total reviews assigned
    df = df.sort_values(by="Total", ascending=False)
    df.index = range(1, len(df) + 1)
    # Display Table
    st.header("Entity Sentiment Summary")
    st.dataframe(df, use_container_width=True)

# Tab 2: Entity Frequency
with tab2:
    selected_plot = "Top Entities Mentioned"
    plot_path = os.path.join(plot_dir, plots[selected_plot]["file_path"])

    plotting_utils.plot_entity_frequency(
        report=report,
        top_k=20,
        save_path=plot_path,
    )

    st.image(plot_path,
             caption=plots[selected_plot]["description"],
             use_container_width=True)

    st.markdown("""
        # Key Observations

        - **Spotify App** is the most frequently mentioned entity, with **15,612** mentions, significantly higher than other entities.
        - **Music Selection (4,903 mentions)** and **Ads (4,566 mentions)** are the next most discussed topics, indicating strong user interest in content availability and ad experiences.
        - **Playlist (3,548 mentions)** also has high mentions, suggesting that playlist management is a crucial aspect for users.
        - **Podcast (1,563 mentions)** and **Subscription Cost (1,557 mentions)** are frequently discussed, likely reflecting user opinions on content variety and pricing concerns.
        - **Player Controls (1,246 mentions)** and **Shuffle Feature (1,185 mentions)** are also major topics, highlighting user interest in playback functionality.
        - **UI (1,147 mentions)** and **Audio Quality (1,104 mentions)** indicate that user experience and sound quality are important factors.
        - **Lower-mentioned entities** such as **Storage (108 mentions), Videoke Feature (19 mentions), and Dark Mode (16 mentions)** suggest these features are either niche or not widely discussed.

        ---

        ## Overall Insights:
        - The most mentioned entities generally revolve around **core functionality (Spotify App, Playlists, Player Controls, UI, Audio Quality)** and **monetization factors (Ads, Subscription Cost).**
        - Features with lower mentions might indicate **lesser user engagement** or **lack of awareness**.
        """)

# Tab 3 : Sentiment Distrubution
with tab3:
    selected_plot = "Sentiment Intensity Heatmap"
    plot_path = os.path.join(plot_dir, plots[selected_plot]["file_path"])

    plotting_utils.plot_sentiment_heatmap(report=report, save_path=plot_path)

    st.image(plot_path,
             caption=plots[selected_plot]["description"],
             use_container_width=True)

    with st.container(border=True):
        st.markdown("**How is the sentiment score calulated?**")
        st.latex(r"""
        \frac{\text{positive count} - \text{negative count}}{\text{max(positive count + negative count,1)}} \times 100
        """)
        st.markdown("""
            **Why is this effective?**
            - Simple, informative and easy to understand.
            - Normalizes the score between -1 (fully negative) to +1 (fully positive).
            - Works well for imbalanced sentiment distributions (e.g., when positive or negative reviews dominate).
            """)

if "Time_Submitted" in data.columns:
    # Tab 4 : Trend over time
    with tab4:
        selected_plot = "Trend over time"
        plot_path = os.path.join(plot_dir, plots[selected_plot]["file_path"])

        # Create two columns
        col1, col2 = st.columns([2, 1])  # Adjust width ratios if needed
        # Place widgets in respective columns
        with col1:
            selected_entity = st.selectbox("🔍 Select an Entity:",
                                           sorted(report.keys()))
        with col2:
            selected_interval = st.radio("⏳ Time Interval:",
                                         ["Daily", "Weekly"],
                                         horizontal=True)

        # Generate plot
        plotting_utils.plot_sentiment_trend(
            entity_name=selected_entity,
            data_df=data,
            report=report,
            save_path=plot_path,
            time_interval="D" if selected_interval == "Daily" else "W")

        st.image(plot_path,
                 caption=plots[selected_plot]["description"],
                 use_container_width=True)
