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

# Few-shot examples - cross domain
generalized_examples = [
    [
        # Batch 1: Smartphones
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [Battery Life, Display Quality, Software Performance, Build Durability]
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-101: The adaptive 120Hz display is buttery smooth and perfect for gaming, though the battery drains 30% faster when enabled. Wish the aluminum frame was more scratch-resistant.
        review-102: Camera system is phenomenal - night mode shots look like daytime! However, the phone overheats after 10 minutes of 4K recording.
        review-103: My third unit with the same issue - green tint on gray backgrounds. I'm honestly questioning their QC standards at this point.
        review-104: Clever idea hiding the camera under the display, but I now look like I'm in a sauna in every selfie.
        review-105: Survived being run over by a car (!!) but the curved edges make accidental touches infuriating.
        review-106: 5G connectivity cuts out randomly in crowded areas. Basic feature shouldn't be this buggy.
        review-107: After the recent update, the phone takes forever to open basic apps. The fingerprint sensor stopped recognizing me too.
        review-108: 65W charging is witchcraft - 0 to 100% in 28 minutes flat!
        review-109: I've been using this device for over 3 months now, and it's been a rollercoaster. The display is absolutely stunning — whether I'm watching HDR content or just reading, the colors pop and brightness is top-notch. Battery life started off strong, but after a couple of updates, it's been a bit unpredictable — some days I end with 30%, other days I'm reaching for the charger by dinner. Software-wise, it's mostly snappy, but there are occasional hangs when switching between heavy apps. What annoys me is the heating while playing games, back gets hot within 15 minutes and performance starts throttling soon after.  Overall, I like the device, but it feels like it's 90% there — just not quite flagship perfection."""
         )),
        ("ai", """{{"entity_sentiment_map": {{
            "Display Quality": {{"positive_review_ids": [101,109], "negative_review_ids": [103]}},
            "Design": {{"positive_review_ids": [], "negative_review_ids": [105]}}
            "Battery Life": {{"positive_review_ids": [], "negative_review_ids": [101,109]}},
            "Thermal Performance": {{"positive_review_ids": [], "negative_review_ids": [102, 109]}},
            "Quality Control": {{"positive_review_ids": [], "negative_review_ids": [103]}},
            "Software Performance": {{"positive_review_ids": [109], "negative_review_ids": []}},
            "Build Durability": {{"positive_review_ids": [105,109], "negative_review_ids": []}},
            "Fingerprint Sensor": {{"positive_review_ids": [], "negative_review_ids": [107]}},
            "Network Connectivity": {{"positive_review_ids": [], "negative_review_ids": [106]}},
            "Camera System": {{"positive_review_ids": [102], "negative_review_ids": [104]}},
            "Charging Speed": {{"positive_review_ids": [108], "negative_review_ids": []}},
            "Overall Experience": {{"positive_review_ids": [109], "negative_review_ids": []}}
            }}
        }}""")
    ],
    [
        # Batch 2: Hotel Stays
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        []
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-201: The luxury suite had visible mold in the shower grout. Front desk shrugged and offered 10% off our next stay - how generous.
        review-202: Perfect location - we walked to every major landmark without needing transport!
        review-203: Swimming pool was booked out the entire weekend. What's the point of premium charges if amenities aren't usable?
        review-204: AC rattled like a diesel engine. Maintenance 'fixed' it by turning it off completely during a heatwave. When I escalated the issue to the management, they told we'll fix it before high-tea and its dinner time already, no lookout.
        review-205: Great experience
        review-206: Super happy to spend $25/night for a luxurious resort where  shady towels and lumpy bathrobes are provided, the pool is drained and gym equipment broken.
        review-207: There's a tiny cafe tucked into the lobby — not advertised anywhere, but a fantastic surprise.
        review-208: I've stayed in dozens of hotels in this city — none struck the balance of silence and centrality like this one.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Room Cleanliness": {{"positive_review_ids": [], "negative_review_ids": [201,205]}},
            "Hospitality": {{"positive_review_ids": [205], "negative_review_ids": [201, 204]}},
            "Location Convenience": {{"positive_review_ids": [202,208], "negative_review_ids": []}},
            "Swimming Pool": {{"positive_review_ids": [203], "negative_review_ids": [203,206]}},
            "gym": {{"positive_review_ids": [203], "negative_review_ids": [206]}},
            "Room Maintenance": {{"positive_review_ids": [], "negative_review_ids": [204]}},
            "Value for Money": {{"positive_review_ids": [], "negative_review_ids": [203,206]}},
            "General Satisfaction": {{"positive_review_ids": [205], "negative_review_ids": []}}
            }}
        }}""")
    ],
    [
        # Batch 3: Airline Experiences
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
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
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Lounge": {{"positive_review_ids": [], "negative_review_ids": [301]}},
            "Delay handling": {{"positive_review_ids": [], "negative_review_ids": [302]}},
            "Hospitality": {{"positive_review_ids": [303], "negative_review_ids": [302, 308]}},
            "Seat Comfort": {{"positive_review_ids": [305], "negative_review_ids": []}},
            "Baggage Handling": {{"positive_review_ids": [], "negative_review_ids": [306, 308]}},
            "Mobile Application": {{"positive_review_ids": [307], "negative_review_ids": []}},
            "Transparency & Communication": {{"positive_review_ids": [], "negative_review_ids": [302]}},
            "General Statisfaction": {{"positive_review_ids": [], "negative_review_ids": [304,308]}},
            "Check-in Process": {{"positive_review_ids": [], "negative_review_ids": [302, 308]}}
        }}""")
    ],
    [
        # Batch 4: Restaurant Dining
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [Food Quality, Ambiance, Hygiene Standards, Value for Money]
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-401: The 45-minute wait for a table would've been tolerable if the famous melt-in-your-mouth short ribs weren't tougher than my work deadlines.
        review-402: Our waiter disappeared after taking orders - had to flag down 3 different staff to get our check 90 minutes later.
        review-403: Michelin-star presentation with McDonald's-level flavor. The $28 cocktail was the only memorable part.
        review-404: Found a hair in my pasta.
        review-405: Generous portions with authentic flavors! Noise levels were brutal though - had to shout across the table."""
         )),
        ("ai", """{{"entity_sentiment_map": {{
            "Food Quality": {{"positive_review_ids": [405], "negative_review_ids": [401,403]}},
            "Service Speed": {{"positive_review_ids": [], "negative_review_ids": [401,402]}},
            "Ambiance": {{"positive_review_ids": [], "negative_review_ids": [405]}},
            "Portion Size": {{"positive_review_ids": [405], "negative_review_ids": []}},
            "Hygiene Standards": {{"positive_review_ids": [], "negative_review_ids": [404]}},
            "Value for Money": {{"positive_review_ids": [], "negative_review_ids": [403]}}
            }}
        }}""")
    ],
    [
        # Batch 5: E-Commerce
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        []
        
        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-501: Ordered 'medium' but received XXL - now I have to navigate their Byzantine return portal.
        review-502: Package arrived looking like it played football with the postal service. Miraculously, contents were intact!
        review-503: Chat support escalate' my issue 4 times before ghosting me. Never resolved.
        review-504: 2-day delivery took 2 weeks with zero communication. Support people tells that the Seller delayed, Seller blames supply chain issues, nobody takes the respnsibilty.
        review-505: Perfect match to the product photos! Even better quality than expected.
        review-506: Return label never arrived despite 3 requests. Had to eat the $120 cost.
        review-507: Item description said 'waterproof' - it died during light drizzle. When I raised a return request, Seller refused.
        review-508: Packaging was eco-friendly and damage-proof. More companies should do this!
        review-509: Ordered 3 items, received 2 with no explanation. Customer service copy-pasted the same unhelpful response 5 times.
        review-510: Lightning-fast shipping, but the product had clearly been used and repackaged."""
         )),
        ("ai", """{{"entity_sentiment_map": {{
            "Delivery Speed": {{"positive_review_ids": [501,510], "negative_review_ids": [504]}},
            "Item-as-Described": {{"positive_review_ids": [505], "negative_review_ids": [507]}},
            "Packaging Quality": {{"positive_review_ids": [502,508], "negative_review_ids": []}},
            "Customer Support": {{"positive_review_ids": [], "negative_review_ids": [503,504,506,509]}},
            "Return Process": {{"positive_review_ids": [], "negative_review_ids": [501,506]}},
            "Product Authenticity": {{"positive_review_ids": [], "negative_review_ids": [510]}}
            "Seller": {{"positive_review_ids": [], "negative_review_ids": [504, 507]}}
            }}
        }}""")
    ],
    [
        # Example 6: Company Culture at FinTech Startup
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        ["Work-Life Balance", "Management"]

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
        review-601: The work is exciting, sure, but I clock 12-hour days regularly and weekends are optional in theory only. I don't mind the grind now and then, but it's not sustainable.
        review-602: Burnout central.
        review-603: They say it's a flat structure, but try disagreeing in a meeting — you'll get sidelined for weeks.
        review-604: Great place to work.
        review-605: I left because of poor middle management. Top-down communication is broken.
        review-606: It's my first job out of college and honestly, the learning curve has been wild. Long hours, yes, but I've never felt like I was alone, my mentor has always been there to support and guide me through difficulties.
        review-607: I used to lead a team here, and I've watched talented folks leave because their voices weren't heard. We celebrate transparency but don't always practice it.
        review-608: There are plenty of mentoring opportunities if you're proactive. But don't expect spoon-feeding.
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Growth Opportunities": {{"positive_review_ids": [601,609], "negative_review_ids": [] }},
            "Work-Life Balance": {{"positive_review_ids": [], "negative_review_ids": [601,602] }},
            "Transparency": {{"positive_review_ids": [], "negative_review_ids": [607] }},
            "Overall Experience": {{"positive_review_ids": [604], "negative_review_ids": [] }},
            "Management": {{"positive_review_ids": [], "negative_review_ids": [603, 605, 607] }},
            "Mentorship": {{"positive_review_ids": [606,608], "negative_review_ids": [] }}
        }}""")
    ],
    [
        # Example 7: Mouse
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        [Buttons, Scroll Wheel]

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:                                             
        review-701: Eliminated my carpal tunnel pain within days! Though the thumb buttons require gorilla-level pressure to click.
        review-702: Driver software crashed my MacBook Pro twice. The mouse itself is comfortable but not worth the system instability..
        review-703: Doesn't work properly. Won't recommend.
        review-704: Scroll wheel stopped working after 4 months of light use. Warranty process demanded video proof of the defect.
        review-705: Not worth the price.
        review-706: Perfect for my small hands, but the ambidextrous design is a lie - lefties will hate this.
        review-707: Top notch product, just what i needed.                                            
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Wrist Comfort": {{"positive_review_ids": [701, 702, 706], "negative_review_ids": [] }},
            "Driver Stability": {{"positive_review_ids": [], "negative_review_ids": [702] }},
            "Scroll Wheel": {{"positive_review_ids": [], "negative_review_ids": [704] }},
            "Buttons": {{"positive_review_ids": [], "negative_review_ids": [701] }},
            "Design": {{"positive_review_ids": [], "negative_review_ids": [706] }},
            "General Satisfaction": {{"positive_review_ids": [707], "negative_review_ids": [703, 705]}},
            }}
        }}""")
    ],
    [
        # Example 8: general
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
        Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
        []

        You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:                                             
        review-801: Loved it!
        review-802: Amazing quality.
        review-803: Works like a charm.
        review-804: Super happy with this.
        review-805: Disappointed..
        review-806: Terrible experience.
        review-807: Very satisfied.
        review-808: Exceeded expectations.
        review-809: Totally useless.
        review-810: Highly recommend.
        review-811: Wouldn't recommend.
        review-812: Impressed!
        review-813: Worth every penny.
        review-814: Very cheap feeling.
        review-815: Don't waste your money, it's useless.                       
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "General Satisfaction": {{"positive_review_ids": [801,802,803,804,807,808,810,812,813], "negative_review_ids": [805,806,809,811,814,815]}}
            }}
        }}""")
    ],
    [
        # Example 9: Streaming TV Series Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
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
        """)),
        ("ai", """{{"entity_sentiment_map": {{
            "Plot": {{"positive_review_ids": [], "negative_review_ids": [901,903,905,907]}},
            "Lead Performance": {{"positive_review_ids": [902], "negative_review_ids": []}},
            "Cinematography": {{"positive_review_ids": [903], "negative_review_ids": []}}
        }}""")
    ],
    [
        # Example 10: Cosmetic Product Reviews
        ("human",
         prompts.PromptTemplate.from_template(
             """The following entities have been identified from previous reviews. 
    Please refer to and reuse these entities wherever applicable to avoid creating duplicates:
    ["Value for Money", "Texture",]

    You are tasked with extracting entities and their corresponding sentiment from the new set of reviews:
    review-1001: Broke me out in three places, and not the cute kind of breakout.
    review-1002: Feels satisfying on the skin— like a whipped cloud.
    review-1003: My skin drinks it up in the morning, but by afternoon I'm back to flaky patches. Not sure if it's me or the product.
    review-1004: Scent is like melted plastic and regret.
    review-1005: Surprisingly lightweight for how hydrating it feels. I didn't expect to love it, but here we are.
    review-1006: Okay okay.
    review-1007: I bought it for the packaging. I'd probably buy it again just to keep it on my shelf. No regrets.
    review-1008: I used the same moisturizer for six years and it never gave me trouble. At first, I was skeptical. It smelled too strong, and it felt sticky going on. But after two weeks, my skin looked brighter and less irritated, even around the chin where I usually break out. Now I notice when I skip it. The fragrance is still not my favorite, but I'm sold on what it does for my skin.
    """)),
        ("ai", """{{"entity_sentiment_map": {{
        "Effectiveness": {{"positive_review_ids": [1005,1008], "negative_review_ids": [1001,1003]}},
        "Texture": {{"positive_review_ids": [1002,1005], "negative_review_ids": []}},
        "Fragrance": {{"positive_review_ids": [], "negative_review_ids": [1004,1008]}},
        "Packaging": {{"positive_review_ids": [1007], "negative_review_ids": []}}
    }}""")
    ]
]
