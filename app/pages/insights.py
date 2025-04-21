import json
import os

import pandas as pd
import streamlit as st

from utils import analyzer_utils
from utils import constants
from utils import plotting_utils
from utils import pydantic_models

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
    report = pydantic_models.AggregatedResults.parse_obj(raw)

data = analyzer_utils.load_csv(file_path=constants.data_csv_path,
                               columns=constants.features_to_use,
                               reviews_processed=constants.reviews_processed)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Report", "Entity Frequency", "Sentiment Distrubution", "Trend over time"])

# Tab 1: Report
with tab1:
    # Convert dict to DataFrame
    df = pd.DataFrame([{
        "Entity": entity,
        "Positive": sentiment_map.positive_reviews.count,
        "Negative": sentiment_map.negative_reviews.count,
    } for entity, sentiment_map in report.items()])
    # Sort by entity name (alphabetical order)
    df = df.sort_values(by="Entity")
    df.index = range(1, len(df) + 1)
    # Display Table
    st.header("Entity Sentiment Summary")
    st.dataframe(df, use_container_width=True)

# Tab 2: Entity Frequency
with tab2:
    selected_plot = "Top Entities Mentioned"
    plot_path = os.path.join(plot_dir, plots[selected_plot]["file_path"])

    if not os.path.exists(plot_path):
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

    if not os.path.exists(plot_path):
        plotting_utils.plot_sentiment_heatmap(report=report,
                                              save_path=plot_path)

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

    st.markdown("""
        # Key Observations
        #### 🟩 Positive Sentiment:
        - **Music Selection (+0.63)** has the highest **positive sentiment**, indicating strong user appreciation.
        - **Audio Quality (+0.48)** is also viewed **positively**, showing user satisfaction with sound performance.
        - **Spotify App overall sentiment (+0.074)** is slightly **positive**, suggesting a neutral to positive perception.

        #### 🟨 Mildly Negative Sentiment:
        - **Podcast (-0.02), UI (-0.12), and Playlist (-0.2)** have **mild negative sentiment**, indicating some dissatisfaction but not extreme.
        - **Subscription Cost (-0.36), Dark Mode (-0.38), and Background Playback (-0.22)** show moderate **negative sentiment**, suggesting concerns around pricing and UI.

        #### 🟥 Strong Negative Sentiment:
        - **Ads (-0.75)** and **Customer Support (-0.61)** receive **strong negative sentiment**, showing user frustration.
        - **Player Controls (-0.91)** and **Shuffle Feature (-0.92)** are **highly criticized**, indicating a poor user experience.
        - **Light Theme (-1.0), Account Login (-0.99), and Bluetooth (-0.94)** have **the worst sentiment scores**, implying major issues or strong user dissatisfaction.

        ---

        ## Overall Insights:
        - **Users appreciate Music Selection and Audio Quality.**
        - **Subscription Cost, Ads, Player Controls, and Shuffle Feature are major pain points.**
        - **Login issues and UI customization (themes) have extremely negative sentiment.**
        
        """)

# Tab 3 : Trend over time
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
        selected_interval = st.radio("⏳ Time Interval:", ["Daily", "Weekly"],
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
