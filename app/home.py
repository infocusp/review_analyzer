"""This file represents the homepage of the streamlit application."""

import streamlit as st

from utils import constants

# set page config
st.set_page_config(page_title="REVIEW ANALYZER", layout="wide")

# Heading and description
st.header(" REVIEW ANALYZER -Turning Feedback into Fuel")
st.markdown("""
    User reviews are one of the richest sources of insight - but also one of the hardest to tap into. Burried in thousands of comments are real opinions about various aspects of a product like features, services, updates, overall experience etc.

    In a world where users freely share their thought acorss platforms, reviews and feedbacks become a direct line to customer sentiment.
    They reveal what's working, what's broken, and what the users really care about. 

    🎯 ***With the right analysis, feedbacks can be turned into fuel to drive impactful businesses with data-backed decisions rooted in user expierence.*** 
    """)
st.subheader("Tradional review analysis tools")
st.markdown("""
        Traditional review analysis tools treat each review as a single sentiment - Positive, Negative or Neutral. 
        
        But real users are more nuanced. They might love the new feature, but hate the UI. Praise the artist selection, but complain about loading speed. 
        With traditional systems, such insight rich feedbacks are often burried beneath noise, contraditctions, and vague sentiments as they oversimplify the feedback, reducing entire reviews to a single label. 

        """)
st.image(constants.review_level_analysis_img_path)
st.markdown("*The Result?* Uncovered nuances and missed opportunities.")

st.divider()

st.subheader("Our Edge: Entity-Level  Sentiment Analysis")
st.markdown("""
    Our system dives deeper. It doesn't just label ***how users feel*** - but also uncovers ***what they're feeling about***.

    Leveraging the power of LLMs, it break down reviews to indentify specific entities that are being discussed *(fuatures, artists, experiences etc)*, and analyze the sentiment around each one. 
    """)
st.image(constants.entity_level_analysis_img_path)
st.markdown("*The Result?* Clear, focused and more informative insights.")
st.markdown("""
    #### Key Features
    🔍 **Extract Key Entities** -- Identify important themes, features, and topics in reviews.  
    😊 **Analyze Sentiment** -- Determine whether customers are praising or criticizing each entity.  
    🧠 **Memory-Enhanced AI** -- The system remembers entities across reviews for better consistency.  
    📊 **Visual Reports & Trends** -- Dynamic charts and graphs make it easy to spot patterns.  
    """)
st.divider()

# page links
st.markdown(f"""
    ### **Stop guessing—start understanding!**

    ### 🛠️ [Know How](/acedemia)  👉 How It works & What powers it.

    ### 🧐 [Evaluation Portal](/evaluation)  👉 Assessment and Validatation. 

    ### 📈 [The Final Takeaways](/insights) 👉 Key Insights, Trends & Visual Reports. 

    """,
            unsafe_allow_html=True)
