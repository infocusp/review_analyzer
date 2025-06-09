"""This file represents the `academia` page of the streamlit application"""

import streamlit as st

from utils import analyzer_utils
from utils import constants

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
    st.image(image=constants.hld_img_path,
             caption="High-Level System Architecture",
             use_container_width=True)

# Tab 2: Data Preview
with tab2:
    st.header("Sample Data Preview")
    st.write("A glance at the customer reviews dataset before processing.")

    # Load Sample CSV and Display Preview
    data = analyzer_utils.load_csv(file_path=constants.data_csv_path,
                                   columns=constants.features_to_use)
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

    ### **Important Instructions**
    - Focus on **meaning and implication** of the review sentence, not just keywords.
    """,
            language="markdown")

    st.header("User Prompt")
    st.write(
        "The **user prompt** is the actual query made to the LLM. Here we provide existing entities and the reviews to be processed."
    )
    st.code("""
    The following entities have been identified from previous reviews.
    Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
    {existing_entities}
       
    You are tasked with extracting entities/themes/topics and their corresponding sentiment from the new set of reviews:
    {formatted_reviews}
    """,
            language="markdown")

    st.header("Few-Shot Examples")
    st.write(
        "To ensure accurate and structured responses, carefully crafted few-shot examples are used that demonstrate exactly how the LLM should analyze and respond to reviews. These examples represent a diverse range of review styles -- positive, negative, mixed, vague, and detailed -- to help the model generalize across different tones and topics. By providing clear demonstration of entity extraction and sentiment labeling in a consistent format, we guide the model to produce outputs that are both reliable and easily feedable into downstream analysis."
    )

    # Display a few-shot example
    st.code("""
        #1. Mixed cases (positive, negative, synonyms, same entity across reviews)
        Human: 
        The following entities have been identified from previous reviews.
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [Seat Comfort, Baggage Handling, Food Quality]

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-301: The airport lounge smelled like bleach and old cheese, wasted a credit card voucher.
        review-302: Our 9-hour flight became 15 because of a missed connection. Not a single announcement, just mass confusion.
        review-303: I'll never forget the gate agent who sprinted across the terminal to return my passport. Heroes wear hi-vis vests too.
        review-304: Won't recommend this airline at all, only book if it's the only option left.
        review-305: The business class seat reclined so far I thought I'd need a chiropractor — but honestly, I haven't slept that well in weeks.
        review-306: The baggage claim at JFK is like a reverse lottery. You wait, pray, and still go home empty-handed. Mine came 2 days later smelling like diesel.
        review-307: I loved the mobile app! Boarding passes, gate info, luggage tracking — everything worked without needing to talk to a single human. That's a win for me.
        review-308: I travel often for work, and usually have my routine down to a science. But this time? From the start, things felt... off. The check-in process dragged—not because of a line, but because no one seemed to know how to handle a passport that wouldn't scan. I ended up being bounced between counters like a pinball. The flight was uneventful, which I normally appreciate, but somehow I left the plane feeling more drained than usual. Maybe it was the constant buzzing from the overhead bin or the stale air. What stuck with me most though was post-landing — standing alone at the carousel long after everyone else had left, realizing my bag wasn't coming. No apology from the staff, Just a form and a shrug.
        
        AI:
        {
            "entity_sentiment_map" : {
                "Lounge": {"positive_review_ids": [], "negative_review_ids": [301]},
                "Delay handling": {"positive_review_ids": [], "negative_review_ids": [302]},
                "Hospitality": {"positive_review_ids": [303], "negative_review_ids": [302, 308]},
                "Seat Comfort": {"positive_review_ids": [305], "negative_review_ids": []},
                "Baggage Handling": {"positive_review_ids": [], "negative_review_ids": [306, 308]},
                "Mobile Application": {"positive_review_ids": [307], "negative_review_ids": []},
                "Transparency & Communication": {"positive_review_ids": [], "negative_review_ids": [302]},
                "General Statisfaction": {"positive_review_ids": [], "negative_review_ids": [304,308]},
                "Check-in Process": {"positive_review_ids": [], "negative_review_ids": [302, 308]}
            }
        }

        #2. Standardization of synonyms + implicit sentiment
        Human: 
        The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        []

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-901: Binge-watched the whole thing in two nights. Not because I loved it, but because I needed to know how that mess would end.
        review-902: The lead actor carries the show on their back. Half the script is just them reacting silently and somehow it works.
        review-903: Beautiful cinematography but someone explain to me how four characters survived that explosion completely unscathed?
        review-904: Season one had magic. Season two has…flashbacks. So many flashbacks. I spent half the runtime trying to remember what happened in season one.
        review-905: The plot got lost somewhere around episode four.
        review-906: I cried. I laughed. I tweeted angrily at the writers. Isn't that what TV is supposed to do?
        review-907: Could've ended two episodes sooner. Dragged.
        
        AI:
        {
            "entity_sentiment_map" : {
            "Plot": {"positive_review_ids": [], "negative_review_ids": [901,903,905,907]},
            "Lead Performance": {"positive_review_ids": [902], "negative_review_ids": []},
            "Cinematography": {"positive_review_ids": [903], "negative_review_ids": []}
            }
        }

        #3. Mixed sentiment on the same entity
        Human: 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        []

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-301: The music selection is fantastic, but the ads are too frequent.
        review-302: Love the app, but way too many ads.
        review-303: The ads are ruining my experience.
        review-304: They added new genres, which I really appreciate!
        
        AI:
        {
            "entity_sentiment_map" : {
                "Music Selection": {"positive_review_ids": [301, 304], "negative_review_ids": []},
                "Ads": {"positive_review_ids": [], "negative_review_ids": [301, 302, 303]}
            }
        }

        #4. Handling ambiguous sentiment and comparisons
        Human: 
        The following entities have been identified from previous reviews.
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [Food Quality, Ambiance, Hygiene Standards, Value for Money]

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-401: The 45-minute wait for a table would've been tolerable if the famous melt-in-your-mouth short ribs weren't tougher than my work deadlines.
        review-402: Our waiter disappeared after taking orders - had to flag down 3 different staff to get our check 90 minutes later.
        review-403: Michelin-star presentation with McDonald's-level flavor. The $28 cocktail was the only memorable part.
        review-404: Found a hair in my pasta.
        review-405: Generous portions with authentic flavors! Noise levels were brutal though - had to shout across the table.
        
        AI:
        {
            "entity_sentiment_map" : {
                "Food Quality": {"positive_review_ids": [405], "negative_review_ids": [401,403]},
                "Service Speed": {"positive_review_ids": [], "negative_review_ids": [401,402]},
                "Ambiance": {"positive_review_ids": [], "negative_review_ids": [405]},
                "Portion Size": {"positive_review_ids": [405], "negative_review_ids": []},
                "Hygiene Standards": {"positive_review_ids": [], "negative_review_ids": [404]},
                "Value for Money": {"positive_review_ids": [], "negative_review_ids": [403]}
            }
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
                "positive_review_ids": [review_id_1, review_id_2, ...],
                "negative_review_ids": [review_id_3, review_id_4, ...]
            },
            "Entity_2": {
                "positive_review_ids": [review_id_5, review_id_6, ...],
                "negative_review_ids": [review_id_7, review_id_8, ...]
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
