from langchain_core import prompts

# Few-shot examples - specific to spotify app
spotify_examples = [
    [
        # Batch 1: Mixed cases (positive, negative, synonyms, same entity across reviews)
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-101: The sound quality is fantastic! Love how crisp it is.
        review-102: The shuffle feature is completely useless.
        review-103: The audio is crystal clear, amazing clarity in music.
        review-104: The sound system is top-notch, really enjoying it.
        """)),
        ("ai", """{{
            "Audio Quality": {{"positive_reviews": [101, 103, 104], "negative_reviews": []}},
            "Shuffle Feature": {{"positive_reviews": [], "negative_reviews": [102]}}
        }}"""),
    ],
    [
        # Batch 2: Standardization of synonyms + implicit sentiment
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-201: The app experience is smooth and intuitive.
        review-202: Navigating through the UI is frustrating, too many unnecessary steps.Worst app ever.
        review-203: The interface is clean and easy to use.
        review-204: The design and UX are just what I needed.
        """)),
        ("ai", """{{
            "Spotify App": {{"positive_reviews": [201], "negative_reviews": [202]}},
            "UI": {{"positive_reviews": [203,204], "negative_reviews": [202]}}
        }}"""),
    ],
    [
        # Batch 3: Mixed sentiment on the same entity
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-301: The music selection is fantastic, but the ads are too frequent.
        review-302: Love the app, but way too many ads.
        review-303: The ads are ruining my experience.
        review-304: They added new genres, which I really appreciate!
        """)),
        ("ai", """{{
            "Music Selection": {{"positive_reviews": [301, 304], "negative_reviews": []}},
            "Ads": {{"positive_reviews": [], "negative_reviews": [301, 302, 303]}}
        }}"""),
    ],
    [
        # Batch 4: Handling ambiguous sentiment and comparisons
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-401: The app is slightly better now, but the shuffle feature is still useless.
        review-402: Not bad, but I still expected more.
        review-403: The latest update is much better than before!
        review-404: The last version was way smoother than this update.
        """)),
        ("ai", """{{
            "Shuffle Feature": {{"positive_reviews": [], "negative_reviews": [401]}},
            "Spotify App": {{"positive_reviews": [402, 403], "negative_reviews": [404]}}
        }}"""),
    ],
    [
        # Batch 5: Feature requests and implicit criticism
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-501: Would be great if we had a dark mode option.
        review-502: Why is there still no offline lyrics support? Annoying!
        review-503: Offline mode is so helpful when traveling.
        review-504: Missing a key feature—background playback on free accounts.
        """)),
        ("ai", """{{
            "Dark Mode": {{"positive_reviews": [], "negative_reviews": [501]}},
            "Lyrics Support": {{"positive_reviews": [], "negative_reviews": [502]}},
            "Offline Mode": {{"positive_reviews": [503], "negative_reviews": []}},
            "Background Playback": {{"positive_reviews": [], "negative_reviews": [504]}}
        }}"""),
    ],
]

generalized_examples = [
    [
        # Example 1: Tech Product Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-101: The battery life is incredible! Lasts all day without issues.
        review-102: The camera quality is disappointing, expected much better.
        review-103: Love the sleek design and display clarity of the phone.
        review-104: The phone gets too hot while gaming.
        review-105: The battery drains too fast when using high-performance apps.
        """)),
        ("ai", """{{
            "Battery Life": {{"positive_reviews": [101], "negative_reviews": [105] }},
            "Camera Quality": {{"positive_reviews": [], "negative_reviews": [102] }},
            "Design": {{"positive_reviews": [103], "negative_reviews": [] }},
            "Performance": {{"positive_reviews": [], "negative_reviews": [104] }}
        }}"""),
    ],
    [
        # Example 2: Streaming Service Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-201: The video quality is amazing, even on slow internet.
        review-202: So many ads! It ruins the experience.
        review-203: The content selection is diverse and engaging.
        review-204: The subtitles are always out of sync.
        review-205: The ads are annoying, but at least they are skippable.
        """)),
        ("ai", """{{
            "Video Quality": {{"positive_reviews": [201], "negative_reviews": [] }},
            "Ads": {{"positive_reviews": [], "negative_reviews": [202, 205] }},
            "Content Selection": {{"positive_reviews": [203], "negative_reviews": [] }},
            "Subtitles": {{"positive_reviews": [], "negative_reviews": [204] }}
        }}"""),
    ],
    [
        # Example 3: E-Commerce Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-301: The delivery was super fast, received it in one day!
        review-302: The packaging was terrible, the product arrived damaged.
        review-303: Excellent customer support, resolved my issue immediately.
        review-304: The return policy is frustrating, takes too long.
        review-305: The delivery was late, took over a week!
        """)),
        ("ai", """{{
            "Delivery Speed": {{"positive_reviews": [301], "negative_reviews": [305] }},
            "Packaging": {{"positive_reviews": [], "negative_reviews": [302] }},
            "Customer Support": {{"positive_reviews": [303], "negative_reviews": [] }},
            "Return Policy": {{"positive_reviews": [], "negative_reviews": [304] }}
        }}"""),
    ],
    [
        # Example 4: Food Delivery App Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-401: The app is easy to use, very intuitive.
        review-402: My order arrived 30 minutes late, really frustrating!
        review-403: The variety of food options is impressive but so many extra charges.
        review-404: The delivery charges are too high!
        review-405: The app interface is slow, takes forever to load.
        """)),
        ("ai", """{{
            "App": {{"positive_reviews": [401], "negative_reviews": [405] }},
            "Delivery Time": {{"positive_reviews": [], "negative_reviews": [402] }},
            "Food Options": {{"positive_reviews": [403], "negative_reviews": [] }},
            "Pricing": {{"positive_reviews": [], "negative_reviews": [403,404] }}
        }}"""),
    ],
    [
        # Example 5: SaaS Software Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """Extract entities and sentiment from these reviews:
        review-501: The new update is really efficient, saves a lot of time. UI changes are on pint.
        review-502: The UI is confusing, takes too long to find basic features.
        review-503: Love the integration with third-party tools, very useful.The UI is awesome.
        review-504: Frequent bugs make the software unreliable.
        review-505: The update introduced some bugs, but performance is slightly better.
        """)),
        ("ai", """{{
            "Software Update": {{"positive_reviews": [501], "negative_reviews": [505] }},
            "User Interface": {{"positive_reviews": [501,503], "negative_reviews": [502] }},
            "Integrations": {{"positive_reviews": [503], "negative_reviews": [] }},
            "Reliability": {{"positive_reviews": [], "negative_reviews": [504] }}
        }}"""),
    ]
]
