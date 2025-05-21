# Review Analyzer - Extract Key themes/topics along with their sentiments

Analyzes user reviews using LLM to identify the key entities being discussed along with their sentiment (Positive or Negative) and generates meaningful insights. So basically it a combination of Named Entity Recognition and Sentiment Anlysis, performed at real time using the power of LLM.


Includes a web application which includes detailed explanation of how and why the solution works. Also, you can explore various interactive insights and analyze the user feedbacks efficiently and effectively.

# Key components
The system is primarily driven by few-shot prompting, batch-processing and context chaining for better accuracy and efficiency.

- **LLM-Based Review Understanding:** Uses Gemini-2.0-Flash for natural language understanding.
- **Few-shot Prompting:** Guides the LLM to extract meaningful entities and understand the underlying sentiment using domain specific examples.
- **Entity-Level Sentiment Analysis:** Unlike traditional systems that assign a single sentiment per review, this system evaluates and assigns sentiment to each entity mentioned in the review, providing a more granular and accurate understanding of the user's opinion.
- **Batch Processing Mechanism:** Reviews are processed in batches. By any chance, if the process stops adruptly (for eg: LLM daily quota is exhausted), a checkpoint including all details and results retrieved till then will be saved. The saved checkpoint can be used to resume the analysis from next batch onwards.
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

# Data Preparation
## Amazon-Reviews Data
Follow the steps below to prepare Amazon review data for use in this project.

### step 1. Download the Data

- Visit the dataset page:  
  👉 [Amazon Reviews 2023 – McAuley Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

- Scroll to the **"Grouped by Category"** section.

- For your desired category (e.g., *Office_products*, *Electronics*), download both:
  - review
  - meta

---

### step 2. Organize the Files

Place the downloaded files in the following directory: `data/amazon_reviews/<category_name>`

### step 3. Run the Preparation Script

Use the following command to process the data:

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

## SemEval Data
### step 1. Download the Data

- Download the data from provide link:  
  👉 [SemEval 2014 Task 4: AspectBasedSentimentAnalysis](https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis)

### step 2. Organize the Files

Place the downloaded files in the following directory: `data/semeval/`

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
--file_path data/semeval/Restaurants_Train_v2.csv \
--save_path data/semeval/prepared_restaurants_train.csv
```

# Run Analysis

```bash
python -m src.analyzer
```
**NOTE:** Logs and analysis report will be saved under `results\<dataset_name>\<experiment_name>`, as configured in `constants.py`.

# Launch Web-App
It transforms raw customer reviews into structured insights.Beyond visual reports, it includes sections for evaluation and the underlying academic design of the solution.
```bash
PYTHONPATH=. streamlit run app/home.py
```
# Customization
The current system is optimized for **Spotify user reviews**. If you want to analyse reviews for some other product or service, you need to:

- Modify the few-shot examples in `src/few_shot_examples` and the system prompt in `prompts.py`.
- Make sure the data format alligns with the current one, for eg: column names in csv file.\

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