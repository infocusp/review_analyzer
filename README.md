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


# Run Analysis

```bash
python -m src.analyzer
```

# Launch Web-App
It transforms raw customer reviews into structured insights.Beyond visual reports, it includes sections for evaluation and the underlying academic design of the solution.
```bash
PYTHONPATH=. streamlit run app/home.py
```
# Customization
The current system is optimized for **Spotify user reviews**. If you want to analyse reviews for some other product or service, you need to:

- Modify the few-shot examples in `src/few_shot_examples` and the system prompt in `prompts.py`.
- Make sure the data format alligns with the current one, for eg: column names in csv file.\
