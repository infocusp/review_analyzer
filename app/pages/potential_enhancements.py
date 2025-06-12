"""This file represents the `potential enhancement` page of the streamlit application"""

import streamlit as st

st.header("Potential Enhancements")

st.subheader("1. Improve Coverage")
st.write(
    "Currently, around 10% of reviews getting are missed, i.e, no entity is extraced.Improving coverage means ensuring that more reviews are assigned meaningful entities. Here are some strategies to achieve this:  "
)
st.markdown("""
    **Enhance the prompts**
    - Use better few-shot examples to guide the LLM on more diverse review-styles.
    - Explicitly instruct the model to avoid skipping relevant information.

    **Adding sentiment categories**
    - Include neutral reviews.
    - If a review contains any suggestion, instead of skipping it, assign it to an "Improvement Suggestions" category.

    **Additional LLM call**
    - Re-run Unassigned Reviews with Adjusted Prompts.
    """)

st.subheader("2. Improve Accuracy")
st.markdown("""
    **Injecting Product/Domain Knowledge**
    - Provide the LLM with context about the product's features to map vague reviews correctly.

    **Post-Processing Refinement**
    - After the final report is generated, a LLM call can be made to merge two entities if they are very similar.
    - This will act as a correction step wherein mistakes commited while assigning entities can be corrected.
    - For example: "Bugs", "Glitches" --> "Bugs"
    """)

st.subheader("3. Advanced Features")
st.markdown("""
    **Sentiment Tagging**

    - Highliting the exact phrase in the review due to which either sentiment was assigned to a particular entity.

    - Example: 
        - **Review** : love most of the features in the app but one thing is <span style='color:red'>a bit expensive at each month subcription for $9.90</span>
        - **Assigned Entity** : Subscription Cost, **Sentiment** : Negative

    """,
            unsafe_allow_html=True)

st.subheader("4. Making the system more scalable")
st.markdown("""
    **Cloud Integration & API Deployment**

    - Deploy as an API service to enable real-time review analysis.

    **Multi-Language Support**
    - Enable multi-language support to process reviews from global users.

    """)
