"""This file contains a class for analyzing reviews."""

import json
import os
import time
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from langchain import output_parsers
import langchain_google_genai
import pandas as pd
import tqdm

from src import prompts
from utils import analyzer_utils
from utils import constants
from utils import data_models

#Initialize logger
Mylogger = analyzer_utils.Logger("Review Analyzer")
logger = Mylogger.get_logger()
# load env variables
load_dotenv()


class ReviewAnalyzer:
    """Class to analyze user reviews using LLM."""

    def __init__(self, report_path: str = "analysis_report.json"):
        """ReviewAnalyzer parameters initialization.

        Args:
            report_path (str) : path to save/load the aggregated results(json report).
        """

        # Initialize Gemini model
        llm = langchain_google_genai.ChatGoogleGenerativeAI(
            model=constants.model)
        parser = output_parsers.PydanticOutputParser(
            pydantic_object=data_models.AggregatedResults)
        self.structured_llm = llm | parser
        self.result_path = report_path

        # Load previously aggregated results
        if os.path.exists(self.result_path):
            previous_state = analyzer_utils.read_json(self.result_path)
            self.aggregated_results = data_models.AggregatedResults.model_validate(
                previous_state)
        else:
            logger.info(
                f"Could not find previous state for aggregated results at provided path : {self.result_path}, creating new report."
            )
            self.aggregated_results = data_models.AggregatedResults(
                entity_sentiment_map={})

    def format_reviews(self, reviews: List[Tuple[int, str]]) -> str:
        """Formats the batch reviews in a single string.

        Args:
            reviews (List[Tuple[int, str]]) : List of reviews (current batch)
            batch_start_idx (int) : start index of the batch

        Returns:
            formatted_reviews (str) : Reviews formatted as a string
        """

        formatted_reviews = "\n".join(
            [f"review-{id} : {review}" for id, review in reviews])
        return formatted_reviews

    def process_reviews_in_batches(
            self,
            data: pd.DataFrame,
            batch_size: int = 50) -> data_models.AggregatedResults:
        """Processes user reviews in batches, extracting entities and sentiment from each batch.

        Args:
            data (pd.DataFrame): Dataframe containing all processed reviews
            batch_size (int, optional): The number of reviews to process in a single batch. Default is 50.

        Returns:
            aggregated_results (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            a dictionary containing sets of review IDS corresponnding to each sentiment.

        Functionality:
            - skips processed batches using previous state.
            - Splits the list of reviews into smaller batches of size `batch_size`.
            - Generates structured prompts for the model using predefined templates.
            - Calls the LLM model to extract entities and sentiments for each batch.
            - Aggregates extracted entities.
            - Save the checkpoint details and results after processing each batch.
        """
        os.makedirs(constants.result_subdir, exist_ok=True)
        os.makedirs(constants.debug_dir, exist_ok=True)
        if self.aggregated_results.batch_size is None:
            self.aggregated_results.batch_size = batch_size
        else:
            assert self.aggregated_results.batch_size == batch_size, f"batch size Mismatch, Checkpoint: {self.aggregated_results.batch_size}, Current: {batch_size}"

        reviews = list(data["Review"].items())
        logger.info(
            f"Processing {len(reviews)} reviews in batches of {batch_size}...")
        print("=" * 100)

        # extract reviews and process batch
        for batch_start_idx in tqdm.tqdm(range(0, len(reviews), batch_size)):
            print("- -" * 60)

            # Skip already completed batches
            if self.aggregated_results.last_batch_idx:
                if batch_start_idx <= self.aggregated_results.last_batch_idx:
                    logger.info(
                        f"Skipping batch {batch_start_idx//batch_size}[reviews {batch_start_idx} - {batch_start_idx+batch_size}], already processed."
                    )
                    continue

            # Load batch and format input
            logger.info(
                f"Loading batch {batch_start_idx // batch_size + 1}, Reviews {batch_start_idx}-{batch_start_idx+batch_size}\n"
            )

            # Slice reviews for current batch
            batch_reviews = reviews[batch_start_idx:batch_start_idx +
                                    batch_size]

            # Format batch reviews in a string
            formatted_reviews = self.format_reviews(reviews=batch_reviews)

            existing_entities = self.aggregated_results.existing_entities
            # format the ChatPromptTemplate with system, user prompt
            formatted_prompt = prompts.chat_prompt_template.format(
                system_prompt=prompts.get_system_propmt(),
                user_prompt=prompts.get_user_prompt(
                    existing_entities=existing_entities,
                    formatted_reviews=formatted_reviews))

            if batch_start_idx == 0:
                print("=" * 100)
                print(formatted_prompt)
                print("=" * 100)

            try:
                logger.info("Invoking LLM ..")
                t1 = time.perf_counter()
                # LLM call
                response = self.structured_llm.invoke(formatted_prompt)
                t2 = time.perf_counter()
                logger.info(
                    f"time taken to process the batch: {(t2-t1)*1000} ms")
                try:
                    validated_response = data_models.AggregatedResults.model_validate(
                        response)
                except Exception as e:
                    logger.error(f"Validation Error: {e}")

                analyzer_utils.dump_batch_log(
                    batch_idx=(batch_start_idx // batch_size) + 1,
                    llm_input=formatted_prompt,
                    llm_output=response.model_dump_json())
                logger.info(
                    f"ENTITIES EXTRACTED IN CURRENT BATCH : {list(validated_response.keys())}\n"
                )

                # Update memory and aggregate results
                logger.info("Updating Memory and Aggregating Results")
                self.aggregated_results.update(validated_response,
                                               batch_start_idx)

                if len(existing_entities) != len(
                        self.aggregated_results.existing_entities):
                    new_entities = [
                        entity_name for entity_name in
                        self.aggregated_results.existing_entities
                        if entity_name not in existing_entities
                    ]
                    logger.info(
                        f"added new entities to memory : {new_entities}")
                else:
                    logger.info(
                        "Did not encounter any new entity, skipped memory update."
                    )

                logger.info("Results aggregated successfully.\n")
                logger.info(
                    f"[MEMORY | EXISTING ENTITIES]:\n{self.aggregated_results.existing_entities}\n"
                )
            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_start_idx // batch_size + 1}: {e}"
                )
                logger.info(
                    f"{self.aggregated_results.last_batch_idx//batch_size +1 } batches,i.e,, {self.aggregated_results.last_batch_idx+batch_size} reviews proceced, saving details to {self.result_path}"
                )
                break

            # Save aggregated results after every batch
            with open(self.result_path, "w") as f:
                f.write(self.aggregated_results.model_dump_json(indent=4))

            time.sleep(2)  # To Prevent rate limit issues
        else:
            logger.info(
                f"All batches proceced, results saved to {self.result_path}")
        return self.aggregated_results


def main(csv_file):

    data = analyzer_utils.load_csv(
        file_path=csv_file, reviews_processed=constants.reviews_processed)
    analyzer = ReviewAnalyzer(report_path=constants.aggregated_results_path)

    analysis_report = analyzer.process_reviews_in_batches(
        data, batch_size=constants.batch_size)


if __name__ == "__main__":
    main(constants.data_csv_path)
