"""This file contains code to debug a particular batch output."""

import argparse
import json
import os
import pprint

from dotenv import load_dotenv
import langchain_google_genai

from utils import analyzer_utils
from utils import constants

load_dotenv()

#Initialize logger
logger = analyzer_utils.Logger("Review Analyzer").get_logger()


def debug_batch(file_path: str,
                llm: langchain_google_genai.ChatGoogleGenerativeAI) -> None:
    """Debug a batch by loading input from saved log and the regenerating output.

    Args:
        file_path (str): path to saved batch log (json file)
    Returns:
        None
    """
    with open(file_path, "r") as f:
        debug_data = json.load(f)

    logger.info(f"\nFetching saved input and output from {file_path}...")
    input_prompt = debug_data["query"]
    saved_output = debug_data.get("response")

    saved_output = json.loads(saved_output)
    logger.info("\n📄 Saved Output:")
    pprint.pprint(saved_output['entity_sentiment_map'])
    logger.info(
        f"Entities in Saved output: {list(saved_output['entity_sentiment_map'].keys())}"
    )

    logger.info("\nRunning LLM on saved input...")
    regenerated_output = llm.invoke(input_prompt)
    try:
        regenerated_output = json.loads(
            regenerated_output.content.strip("```json"))
    except Exception as e:
        logger.error(f"Error parsing generated output : {e}")
        raise
    logger.info("\n🆕 Regenerated Output:")
    pprint.pprint(regenerated_output['entity_sentiment_map'])
    logger.info(
        f"Entities in regenerated output: {list(regenerated_output['entity_sentiment_map'].keys())}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Debug LLM output using a saved batch JSON file.")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help=
        "Path to the JSON file containing logged data for a particular batch.")
    args = parser.parse_args()

    file_path = args.log_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    llm = langchain_google_genai.ChatGoogleGenerativeAI(model=constants.model)

    debug_batch(file_path, llm)


if __name__ == "__main__":
    main()
