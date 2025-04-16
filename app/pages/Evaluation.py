import os
import sys
import time
import streamlit as st

sys.path.append("..")
import Utils.plotting_utils as plotter
from Utils.analyzer_utils import (
    load_csv,
    load_analysis_report,
    get_reviews_for_entity,
    analyze_coverage)

# Page Title
st.title("📊 Evaluation & Quality Assessment")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Coverage Analysis", "📊 Review Length vs. Entities Extracted", "🧑🏾‍💻 Manual Verification"])

plot_dir = "./static/plots"  # All plots needs to placed here
os.makedirs(plot_dir, exist_ok=True)

plot_files = {
    "Review Length vs. Number of Entities": "review_length_vs_entities_violin.png",
    "Coverage Analysis": "coverage_analysis.png",
}

# Load Data and report
json_report_path = "./static/Analysis_report.json"
report = load_analysis_report(json_report_path)

review_csv_path = "../data/spotify_reviews.csv"
data = load_csv(review_csv_path, columns=["Time_submitted", "Review"], reviews_processed = 34050)

reviews = data["Review"].dropna().to_list()

# Tab 1: Coverage Analysis
with tab1:
    st.header("📈 Coverage Analysis")
    st.write(
        "Understanding how well our system assigns reviews to entities is crucial for evaluating its effectiveness. "
        "Here, we measure the **proportion of reviews for which at least one entity is extracted** to ensure comprehensive coverage."
    )
    
    coverage_report = analyze_coverage(data,report)
    # Display the Metric 
    total_reviews = coverage_report["total_reviews"]  
    reviews_without_entities = coverage_report["unattended_reviews"]  
    coverage_percentage = ((total_reviews - len(reviews_without_entities)) / total_reviews) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("📝 Total Reviews", total_reviews)
    zero_entity_reviews = col2.empty()
    
    # Create placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()  # Placeholder for progress text
    
    current_reviews_without_entities = 0
    step = int(len(reviews_without_entities)//(coverage_percentage+1))
    # Simulate progress update
    for i, percent in enumerate(range(int(coverage_percentage) + 1)):
        time.sleep(0.03)  # Simulate processing delay
        
        # Update progress bar
        progress_bar.progress(percent / 100)

        # Update progress status text
        status_text.write(f"**Coverage: {percent}%** of reviews have assigned entities.")

        # update count of unattended reviews
        if current_reviews_without_entities < len(reviews_without_entities):
            if i==int(coverage_percentage):
                current_reviews_without_entities = len(reviews_without_entities)
            else:    
                current_reviews_without_entities += (step + 1)

        # Update metric dynamically
        zero_entity_reviews.metric("🔴 Reviews Without Entities", current_reviews_without_entities)
        
    status_text.write(f"**Final Coverage: {coverage_percentage:.2f}%** ✅")

    st.divider()

    # Display the reviews for which no entities were assigned
    st.subheader("Reviews Without Assigned Entities")
    st.dataframe(reviews_without_entities)

# Tab 2: Review Length vs. Entities Extracted
with tab2:
    st.header("📊 Review Length vs. Entities Extracted")
    st.write(
        "This analysis helps us understand if longer reviews contain more extracted entities, "
        "or if entity extraction is independent of review length."
    )

    #  Display Selected Plot
    plot_path = os.path.join(plot_dir,"review_length_vs_entities_violin.png")
    if not os.path.exists(plot_path):
        # generate plot
        plotter.plot_review_length_vs_entities_violin(
            reviews=data["Review"].dropna().to_list(),
            report=report,
            save_path=plot_path
        )
    st.image(plot_path, use_container_width=True)
    
    st.markdown(
        """
        ### Performance Evaluation of LLM-Based Extraction
        - Some reviews across all bins have zero entities, which could indicate:

            - Cases where relevant entities were missed.

            - Reviews that are vague or lack extractable information.

        - If long reviews consistently yield a low number of entities, this might signal under-extraction issues.

        - Conversely, short reviews producing multiple entities could indicate over-extraction or noise in the LLM pipeline.
        """
    )

    st.markdown(
        """
        ### Entity Extraction Trends
        - Across different review length bins, the median number of entities extracted remains relatively stable.

        - Shorter reviews (less than 30 words) typically mention only 1-2 entities, which aligns with expectations since they contain fewer concepts.

        - Longer reviews exhibit higher variance, sometimes mentioning up to 5-6 entities, indicating more detailed feedback.
        """
    )

    st.markdown(
        """
        ### Future Anomaly Detection & Continuous Improvement
        - This plot can help track LLM performance over time to detect extraction inconsistencies.

        - Any sudden deviation in entity distribution (e.g., too few or too many entities in a given length range) could indicate shifts in review patterns or extraction inefficiencies.

        - Regular monitoring of this visualization can help fine-tune the entity extraction system and identify areas for optimization.
        """
    )

# Tab 3: Manual Verification
with tab3:
    st.write("Manually verify the reviews tagged under respective sentiment for a particular entity.")
    # Create two columns
    col1, col2 = st.columns([2, 1])  # Adjust width ratios if needed
    # Place widgets in respective columns
    with col1:
        selected_entity = st.selectbox("🔍 Select an Entity:", sorted(report.keys()))  
    with col2:
        selected_sentiment = st.radio("⏳ Sentiment:", ["positive", "negative"], horizontal=True)  

    reviews = get_reviews_for_entity(data=data, report=report, entity_name=selected_entity, sentiment=selected_sentiment)
    st.dataframe(reviews)