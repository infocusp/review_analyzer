"""This file contains few-shot examples."""

from langchain_core import prompts

# Few-shot examples - specific to spotify app
spotify_examples = [
    [
        # Batch 1: Mixed cases (positive, negative, synonyms, same entity across reviews)
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-101: The sound quality is fantastic! Love how crisp it is.
        review-102: The shuffle feature is completely useless.
        review-103: The audio is crystal clear, amazing clarity in music.
        review-104: The sound system is top-notch, really enjoying it.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Audio Quality": {{"positive_review_ids": [101, 103, 104], "negative_review_ids": []}},
            "Shuffle Feature": {{"positive_review_ids": [], "negative_review_ids": [102]}}
             }}
         }}"""),
    ],
    [
        # Batch 2: Standardization of synonyms + implicit sentiment
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-201: The app experience is smooth and intuitive.
        review-202: Navigating through the UI is frustrating, too many unnecessary steps.Worst app ever.
        review-203: The interface is clean and easy to use.
        review-204: The design and UX are just what I needed.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Spotify App": {{"positive_review_ids": [201], "negative_review_ids": [202]}},
            "UI": {{"positive_review_ids": [203,204], "negative_review_ids": [202]}}
            }}
        }}"""),
    ],
    [
        # Batch 3: Mixed sentiment on the same entity
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-301: The music selection is fantastic, but the ads are too frequent.
        review-302: Love the app, but way too many ads.
        review-303: The ads are ruining my experience.
        review-304: They added new genres, which I really appreciate!
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Music Selection": {{"positive_review_ids": [301, 304], "negative_review_ids": []}},
            "Ads": {{"positive_review_ids": [], "negative_review_ids": [301, 302, 303]}}
            }}
        }}"""),
    ],
    [
        # Batch 4: Handling ambiguous sentiment and comparisons
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-401: The app is slightly better now, but the shuffle feature is still useless.
        review-402: Not bad, but I still expected more.
        review-403: The latest update is much better than before!
        review-404: The last version was way smoother than this update.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Shuffle Feature": {{"positive_review_ids": [], "negative_review_ids": [401]}},
            "Spotify App": {{"positive_review_ids": [402, 403], "negative_review_ids": [404]}}
            }}
        }}"""),
    ],
    [
        # Batch 5: Feature requests and implicit criticism
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-501: Would be great if we had a dark mode option.
        review-502: Why is there still no offline lyrics support? Annoying!
        review-503: Offline mode is so helpful when traveling.
        review-504: Missing a key feature—background playback on free accounts.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Dark Mode": {{"positive_review_ids": [], "negative_review_ids": [501]}},
            "Lyrics Support": {{"positive_review_ids": [], "negative_review_ids": [502]}},
            "Offline Mode": {{"positive_review_ids": [503], "negative_review_ids": []}},
            "Background Playback": {{"positive_review_ids": [], "negative_review_ids": [504]}}
            }}
        }}"""),
    ],
]

generalized_examples = [
    [
        # Example 1: Tech Product Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-101: The battery life is incredible! Lasts all day without issues.
        review-102: The camera quality is disappointing, expected much better.
        review-103: Love the sleek design and display clarity of the phone.
        review-104: The phone gets too hot while gaming.
        review-105: The battery drains too fast when using high-performance apps.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Battery Life": {{"positive_review_ids": [101], "negative_review_ids": [105] }},
            "Camera Quality": {{"positive_review_ids": [], "negative_review_ids": [102] }},
            "Design": {{"positive_review_ids": [103], "negative_review_ids": [] }},
            "Performance": {{"positive_review_ids": [], "negative_review_ids": [104] }}
            }}
        }}"""),
    ],
    [
        # Example 2: Streaming Service Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-201: The video quality is amazing, even on slow internet.
        review-202: So many ads! It ruins the experience.
        review-203: The content selection is diverse and engaging.
        review-204: The subtitles are always out of sync.
        review-205: The ads are annoying, but at least they are skippable.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Video Quality": {{"positive_review_ids": [201], "negative_review_ids": [] }},
            "Ads": {{"positive_review_ids": [], "negative_review_ids": [202, 205] }},
            "Content Selection": {{"positive_review_ids": [203], "negative_review_ids": [] }},
            "Subtitles": {{"positive_review_ids": [], "negative_review_ids": [204] }}
            }}
        }}"""),
    ],
    [
        # Example 3: E-Commerce Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-301: The delivery was super fast, received it in one day!
        review-302: The packaging was terrible, the product arrived damaged.
        review-303: Excellent customer support, resolved my issue immediately.
        review-304: The return policy is frustrating, takes too long.
        review-305: The delivery was late, took over a week!
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Delivery Speed": {{"positive_review_ids": [301], "negative_review_ids": [305] }},
            "Packaging": {{"positive_review_ids": [], "negative_review_ids": [302] }},
            "Customer Support": {{"positive_review_ids": [303], "negative_review_ids": [] }},
            "Return Policy": {{"positive_review_ids": [], "negative_review_ids": [304] }}
            }}
        }}"""),
    ],
    [
        # Example 4: Food Delivery App Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-401: The app is easy to use, very intuitive.
        review-402: My order arrived 30 minutes late, really frustrating!
        review-403: The variety of food options is impressive but so many extra charges.
        review-404: The delivery charges are too high!
        review-405: The app interface is slow, takes forever to load.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "App": {{"positive_review_ids": [401], "negative_review_ids": [405] }},
            "Delivery Time": {{"positive_review_ids": [], "negative_review_ids": [402] }},
            "Food Options": {{"positive_review_ids": [403], "negative_review_ids": [] }},
            "Pricing": {{"positive_review_ids": [], "negative_review_ids": [403,404] }}
            }}
        }}"""),
    ],
    [
        # Example 5: SaaS Software Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        ["performance", "bugs", "UI"] 
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-501: The new update is really efficient, saves a lot of time. User Interface changes are on pint.
        review-502: The UI is confusing, takes too long to find basic features.
        review-503: Love the integration with third-party tools, very useful.The UI is awesome.
        review-504: Frequent bugs make the software unreliable.
        review-505: The update introduced some bugs, but performance is slightly better.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Software Update": {{"positive_review_ids": [501], "negative_review_ids": [505] }},
            "UI": {{"positive_review_ids": [501,503], "negative_review_ids": [502] }},
            "Third Party Integrations": {{"positive_review_ids": [503], "negative_review_ids": [] }},
            "bugs": {{"positive_review_ids": [504, 505], "negative_review_ids": []}},
            "Reliability": {{"positive_review_ids": [], "negative_review_ids": [504] }}
            }}
        }}"""),
    ]
]
