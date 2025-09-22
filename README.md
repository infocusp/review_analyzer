# Review Analyzer: LLM-Powered Feedback Insights

> Analyze user reviews using LLMs to extract entity-level sentiment, uncover key themes, and explore insights via a web interface.

![Streamlit App Demo](assets/webapp_demo.gif)


# 🔍 Overview

This project uses **large language models (LLMs)** to analyze user reviews by extracting key **entities** (such as product features or services) and identifying the associated **sentiments** (positive or negative). It combines **named entity recognition (NER)** with **sentiment analysis** to generate structured, interpretable insights from raw textual feedback.

The system includes two main components:

- ## 🧠 Review Analyzer  
  A backend module that processes batches of user reviews and generates a structured **JSON report**. Each entry captures the detected entities, their sentiment, and the associated review id.

- ## 💻 Web Application  
  An interactive frontend that loads and visualizes the JSON report. It provides:
  - A static explanation of the methodology, such as a flowchart describing how the **LLM-based analysis** works
  - Aggregate statistics and visual summaries across all reviews  
  - Detailed insights into each review, including extracted entities and sentiment labels  

This setup is ideal for teams that want to analyze user feedback at scale with **LLM-powered precision** and explore the results through a clean, insightful interface.

# Key components
The system is primarily driven by few-shot prompting, batch-processing and context chaining for better accuracy and efficiency.

- **LLM-Based Review Understanding:** Uses Gemini-2.0-Flash for natural language understanding.
- **Few-shot Prompting:** Guides the LLM to extract meaningful entities and understand the underlying sentiment using domain specific examples.
- **Entity-Level Sentiment Analysis:** Unlike traditional systems that assign a single sentiment per review, this system evaluates and assigns sentiment to each entity mentioned in the review, providing a more granular and accurate understanding of the user's opinion.
- **Batch Processing Mechanism:** Reviews are processed in batches. By any chance, if the process stops abruptly (for eg: LLM daily quota is exhausted), a checkpoint including all details and results retrieved till then will be saved. The saved checkpoint can be used to resume the analysis from next batch onwards.
- **Context Memory:** All the unique entities extracted are stored and used by the LLM as reference before generating new entities for the subsequent batch. It helps in generalizing over similar entities and avoid duplications.

# Setup
Setp 1: Create a Virtual Environment (recommended)
```bash
python3.10 -m venv <env_name>
source <env_name>/bin/activate
```
Step 2: Install dependencies:
```bash
pip install -r requirements.txt
```
Step 3: Configure API Key for LLM:
- create a `.env` file in the project root directory
- write to file: GOOGLE_API_KEY = "your api key"

# Dataset

We use the following for review analysis:

- **ABSA (Aspect-Based Sentiment Analysis) Dataset from SemEval-2014:**  
   Contains fine-grained sentiment annotations for multiple aspects per review, covering Laptops and Restaurants.  
   It’s particularly useful for evaluating the system’s accuracy in aspect-level sentiment detection.

 <details>

  <summary><b>Additional Datasets (click to expand)</b></summary>

- **Amazon Reviews Dataset:**  
   Provides large-scale, real-world customer reviews across various product categories.  
   It allows us to analyze sentiment and extract key aspects from diverse and noisy review data, making it ideal for general-purpose entity-level sentiment analysis.

- **Spotify Review Data:**  
   Contains user reviews for the Spotify app collected from the Google Play Store.
</details>

---

# Data Preparation

**Input format:**  
Each dataset should be a CSV file with a column named `"Review"`.  
Optionally, a column named `"Time_submitted"` can be included to store review timestamps (date or date-time).

---

## SemEval Data
Follow the steps below to prepare SemEval-2014 data for use in this project.

### step 1. Download the Data

- Download the data from provide link:  
  👉 [SemEval 2014 Task 4: AspectBasedSentimentAnalysis](https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis)

### step 2. Organize the Files

Extract the zip downloaded in step 1 and place the files in the following directory structure:
```
data/
└── semeval/
  ├── Laptop_Train_v2.csv
  └── Restaurants_Train_v2.csv
  └── ....
```   

### step 3. Run the Preparation Script

Use the following command to process the data:

```bash
python -m data_preparation.prepare_semeval_data \
--file_path <path to the csv dataset file> \
--save_path <path to save prepared data>
```
**Example**
```bash
python -m data_preparation.prepare_semeval_data \
--file_path data/semeval/Laptop_Train_v2.csv \
--save_path data/semeval/prepared_laptop_train.csv
```
In the original dataset, each review is split across multiple rows, with each row representing a different aspect and its sentiment. During data preparation, these rows are merged into a single row per review to consolidate all aspect-sentiment pairs together. The prepared csv will include a new column `Aspects`, which contains a list of dictionaries like:

```json
[
  {"aspect": "battery life", "polarity": "positive"},
  {"aspect": "price", "polarity": "negative"}
]
```

<details>
<summary><b>Additional Datasets (click to expand)</b></summary>

 ## Spotify Review Data

 ### step 1. Download the Data

- Download the data from provide link:  
  👉 [spotify-app-reviews-2022](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022)

 ### step 2. Organize the Files
-  Place it in the following directory:
```
data/
└── spotify_reviews.csv
```

## Amazon-Reviews Data 

Follow the steps below to prepare Amazon review data for use in this project.

### step 1. Download the Data

- Visit the dataset page:  
  👉 [Amazon Reviews 2023 – McAuley Lab | grouped-by-category](https://amazon-reviews-2023.github.io/#grouped-by-category)

- For your desired category (e.g., *Office_products*, *Electronics*), download both:
  - review
  - meta

---

### step 2. Organize the Files
After downloading the desired category-wise dataset as per step 1, you will get two files(zip) for the selected category
(For eg: Fashion):

- A **review file** (e.g., `Fashion.jsonl`)
- A **metadata file** (e.g., `meta_Fashion.jsonl`)

Extract and place both files in the following directory structure:
```
data/
└── amazon_reviews/
  └── fashion/                    <----   <category_name>
    ├── Fashion.jsonl
    └── meta_Fashion.jsonl
```    

### step 3. Run the Preparation Script

Once organized, use the following command to process the data:

```bash
python -m data_preparation.prepare_amazon_data \
--data_dir <path_to_dataset_dir> \
--review_filename <review_file_name> \
--meta_filename <meta_file_name>
```
**Example**
```bash
python -m data_preparation.prepare_amazon_data \
--data_dir data/amazon_reviews/fashion \
--review_filename Amazon_Fashion.jsonl \
--meta_filename meta_Amazon_Fashion.jsonl
```
This will process the dataset by grouping reviews with the same parent_asin and save the top 20 product groups (based on number of reviews) as separate datasets.

</details>

---

# Run the Analyzer

### Step 1: Configure Dataset & Experiment Name
In `constants.py`, set the following:

- `dataset_name`: Choose one of the predefined keys from the `data` dictionary or register your dataset's path and use that.  
- `experiment_name`: A short identifier for your run.

### step 2: Execute the review analysis:

Run the following command from project root:
```bash
python -m src.analyzer
```

This will:
- Load the dataset specified in constants.py
- Process reviews in batches using LLM
- Log inputs and outputs
- Save final structured entity-sentiment map

**Results and logs will be saved under:** `results/<dataset_name>/<experiment_name>/`



**Auto-Resume Support:** If the analysis is interrupted midway, simply rerun the command.
The analyzer will resume from the last successfully processed batch using the saved logs.

# Launch Web-App
It transforms raw customer reviews into structured insights. Beyond visual reports, it includes sections for evaluation and the underlying academic design of the solution.

### Step 1: Configure the app
In `constants.py`, set the following:

- `dataset_name`: Same as used while running the analyzer.
- `reviews_processed`: If ran for specific number of batches, mention the number of reviews processed (batch_size * num_batches processed), -1 if ran the analyzer over entire dataset.

### step 2: Copy `analysis_report.jon` to `app/static`
- Copy the generated `analysis_report.json` from the output dir to `app/static`.

### step 3: Run streamlit app
Run the following command from project root:
```bash
PYTHONPATH=. streamlit run app/home.py
```
# Customization
The current system is **dataset-agnostic** and can be applied to review data from any domain (apps, products, services, locations, etc.).

To tailor the system to a specific use case:

- Modify the few-shot examples in `src/few_shot_examples` and the system prompt in `prompts.py`.
- Make sure the data format alligns with the current one, for eg: column names in csv file.

# Debugging
If you'd like to debug the LLM output for a specific batch, you can use the debugging script to re-run the LLM on the same input and compare it with the previously saved output.

#### Usage

Run the script as follows:

```bash
python -m utils.debug_batch_output --log_path results/<dataset_name>/<experiment_name>/logs/<batch_name>.json
```

**Example**
```bash
python -m utils.debug_batch_output --log_path results/laptop/exp1/logs/batch_11.json
```

# Results & Observations
The system has been **qualitatively evaluated** across a range of datasets spanning different domains—products, services, and user experiences. The observed results have been highly encouraging as the entity extraction and sentiment tagging outputs have been consistently accurate and context-aware across domains.

# Citation

If you use this project in your work, please cite it as:

```bibtex
@misc{infocusp2025reviewanalyzer,
  author       = {Falak Shah and Tushar Gadhiya and Milind Padalkar and Yash Bohra},
  title        = {Review Analyzer: LLM-Powered Feedback Insights},
  year         = {2025},
  howpublished = {\url{https://github.com/infocusp/review_analyzer}},
  note         = {Open-source project}
}