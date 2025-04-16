import sys

import streamlit as st

sys.path.append("..")
from Utils.analyzer_utils import load_csv

# Page Title
st.title("🔍 Understanding Our AI-Powered Review Analysis Solution")

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 High-Level Design", "📊 Data Preview", "💬 Prompting",
    "📂 Output Structure", "🧠 Entity Memory"
])

# Tab 1: High-Level Design
with tab1:
    st.header("High-Level Design Overview")
    st.write(
        "The following diagram illustrates how our review analysis system works, from data ingestion to final insights."
    )
    design_diagram_path = "static/high_level_design.png"
    st.image(design_diagram_path,
             caption="High-Level System Architecture",
             use_container_width=True)

# Tab 2: Data Preview
with tab2:
    st.header("Sample Data Preview - Spotify Reviews")
    st.write("""
             Spotify is one of the largest music streaming service providers, with over 422 million monthly active users, including 182 million paying subscribers, as of March 2022. Some of them don't hesitate to share their experience using this application, expressing how satisfied or dissatisfied they are with the Application.
        """)
    st.write("A glance at the customer reviews dataset before processing.")

    # Load Sample CSV and Display Preview
    sample_csv_path = "../data/spotify_reviews.csv"
    data = load_csv(sample_csv_path, columns=["Time_submitted", "Review"])
    data.index = range(1, len(data) + 1)
    st.dataframe(data.head(20))

# Tab 3: Prompting
with tab3:
    st.header("System Prompt")
    st.write(
        "The **system prompt** serves as the core instruction set for the LLM, guiding it to extract structured insights from reviews efficiently."
    )
    st.code("""
    You are an AI assistant specializing in **extracting structured insights from Spotify user reviews**.
    Your goal is to **identify key entities, classify sentiment, and avoid redundant entity creation**.

    ### **Key Responsibilities**
    - **Extract entities**: Identify relevant aspects in reviews.
    - **Standardize names**: Group similar entities to avoid duplication.
    - **Assign sentiment**: Classify as `Positive` or `Negative`, ignoring neutral statements.
    - **Track occurrences**: Store review IDs under respective sentiment categories.

    ### **Existing Entities Context**
    Below are the entities identified so far across previous reviews.Reuse these entities 
    whenever possible and do not create redundant entries.

    {existing_entities}

    ### **Response Format**
    Return structured **JSON only**, without explanations or markdown.
    """,
            language="markdown")

    st.header("Few-Shot Examples")
    st.write(
        "To ensure accurate and structured responses, carefully crafted few-shot examples are used that demonstrate exactly how the LLM should analyze and respond to reviews. These examples represent a diverse range of review styles -- positive, negative, mixed, vague, and detailed -- to help the model generalize across different tones and topics. By providing clear demonstration of entity extraction and sentiment labeling in a consistent format, we guide the model to produce outputs that are both reliable and easily feedable into downstream analysis"
    )

    # Display a few-shot example
    st.code("""
        #1. Mixed cases (positive, negative, synonyms, same entity across reviews)
        Human: 
        Extract entities and sentiment from these reviews:
        review-101: The sound quality is fantastic! Love how crisp it is.
        review-102: The shuffle feature is completely useless.
        review-103: The audio is crystal clear, amazing clarity in music.
        review-104: The sound system is top-notch, really enjoying it.
        
        AI:
        {
            "Audio Quality": {{"positive_reviews": [101, 103, 104], "negative_reviews": []}},
            "Shuffle Feature": {{"positive_reviews": [], "negative_reviews": [102]}}
        }

        #2. Standardization of synonyms + implicit sentiment
        Human: 
        Extract entities and sentiment from these reviews:
        review-201: The app experience is smooth and intuitive.
        review-202: Navigating through the UI is frustrating, too many unnecessary steps.Worst app ever.
        review-203: The interface is clean and easy to use.
        review-204: The design and UX are just what I needed.
        
        AI:
        {
            "Spotify App": {{"positive_reviews": [201], "negative_reviews": [202]}},
            "UI": {{"positive_reviews": [203,204], "negative_reviews": [202]}}
        }

        #3. Mixed sentiment on the same entity
        Human: 
        Extract entities and sentiment from these reviews:
        review-301: The music selection is fantastic, but the ads are too frequent.
        review-302: Love the app, but way too many ads.
        review-303: The ads are ruining my experience.
        review-304: They added new genres, which I really appreciate!
        
        AI:
        {
            "Music Selection": {{"positive_reviews": [301, 304], "negative_reviews": []}},
            "Ads": {{"positive_reviews": [], "negative_reviews": [301, 302, 303]}}
        }

        #4. Handling ambiguous sentiment and comparisons
        Human: 
        Extract entities and sentiment from these reviews:
        review-401: The app is slightly better now, but the shuffle feature is still useless.
        review-402: Not bad, but I still expected more.
        review-403: The latest update is much better than before!
        review-404: The last version was way smoother than this update.
        
        AI:
        {
            "Shuffle Feature": {{"positive_reviews": [], "negative_reviews": [401]}},
            "Spotify App": {{"positive_reviews": [402, 403], "negative_reviews": [404]}}
        }

        """,
            language="markdown")

# Tab 4: Output Structure
with tab4:
    st.header("📄 Structured Output Format")
    st.write(
        "The extracted insights are stored in a structured JSON format, ensuring easy analysis and integration."
    )

    st.code("""
        {
            "Entity_1": {
                "positive_reviews": [review_id_1, review_id_2, ...],
                "negative_reviews": [review_id_3, review_id_4, ...]
            },
            "Entity_2": {
                "positive_reviews": [review_id_5, review_id_6, ...],
                "negative_reviews": [review_id_7, review_id_8, ...]
            }
        }
        """,
            language="json")

    st.success(
        "This structured output enables deeper analysis, visualization, and actionable insights!"
    )

# Tab 5: Memory Management
with tab5:
    st.header("🧠 How Entity Memory Works")

    st.write(
        "To ensure consistency, our system maintains memory of previously identified entities. "
        "This memory is injected into the system prompt for each batch, preventing duplication "
        "and improving entity recognition.")

    st.subheader("Why is it needed?")
    st.markdown("""
        - **To Prevent Duplication** -- If a particular entity has already been extracted in previous batches (e.g., Battery Life), it should not   be extracted again under a different variation (e.g., Battery Performance), ensuring consistency and avoiding duplication.
        - **To Improve Entity Recognition** -- Previously identified entities are used as context, guiding the model to be more precise.
        - **To Enhance Relevance** -- Helps in refining entity extraction, ensuring the model does not miss important mentions.
        """)

    st.subheader("How It Works?")

    st.markdown("""
        **1️⃣ Extracting Entities in Batches**  
        - We process reviews in **small batches** and extract key entities dynamically.
        - For Example, **Batch 1** identifies `"Battery Life"` and `"Sound Quality"` as entities.    

       **2️⃣ Storing in Memory**  
        - The extracted entities are stored in **Memory**.  
        """)

    st.code(
        """
          Memory = {"Entities" : ["Battery Life","Sound Quality"]}
        """,
        language="json",
    )

    st.markdown("""
        **3️⃣ Injecting Memory into the System Prompt**  
        - Before processing the next batch, we **dynamically update the system prompt** to include previously identified entities.
        - The system prompt now explicitly tells the model to **maintain consistency**:

        """)

    st.code(
        """
        **Existing Entities Context**
        Below are the entities identified so far across previous reviews.Reuse these entities 
        whenever possible and do not create redundant entries.

        ["Battery Life","Sound Quality"]
        """,
        language="text",
    )

    st.markdown("""
        **4️⃣ Updating Memory**
        - For each **new batch**, the memory is updated with any additional entities while ensuring previously extracted entities remain referenced.  
        - This prevents duplication and ensures consistency across batches.
        - For Example, **Batch 2** identifies `"Battery Life"` and `"Ads"` as entities.
        """)
    st.code(
        """
          # Updated Memory
          Memory = {"Entities" : ["Battery Life","Sound Quality", "Ads"]}
        """,
        language="json",
    )

    st.markdown("""
        ** 5️⃣ Continuos generation and mapping**  
        - **Entity Alignment with Memory** -- The LLM references existing entities stored in memory. If a review mentions a previously identified concept (e.g., "Battery Life"), the model assigns it to the same entity instead of creating a duplicate (e.g., avoiding "Battery Performance" as a separate entity).

        - **Handling New Topics** -- If a review introduces a completely new topic that isn't in memory, the model creates a new entity dynamically, ensuring that all key themes are captured.

        - **Automatic Categorization** -- The process is fully automated, as the LLM inherently understands context and aligns reviews with relevant entities without requiring additional logic.
        
        """)

    st.subheader("Why This Matters?")
    st.markdown("""
        ✅ **Ensures Accuracy** -- The model learns from previous extractions and maintains consistent entity names.  
        ✅ **Prevents Redundancy** -- Avoids extracting different variations of the same feature.  
        ✅ **Improves Reporting** -- Users get a **clean and structured** report with meaningful insights.  
        """)
