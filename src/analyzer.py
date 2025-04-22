import json
import os
import time
from typing import Dict, List, Set

from dotenv import load_dotenv
from langchain import memory
from langchain_core import messages
import langchain_google_genai
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

    def __init__(self,
                 checkpoint_path: str = "Checkpoint.json",
                 report_path: str = "Analysis_report.json"):
        """ReviewAnalyzer parameters initialization.

        Args:
            checkpoint_path (str) : path to save/load checkpoint.
            report_path (str) : path to save/load the aggregated results(json report).
        """

        # Initialize Gemini model
        self.llm = langchain_google_genai.ChatGoogleGenerativeAI(
            model=constants.model)
        self.ckpt_path = checkpoint_path
        self.result_path = report_path

        # Load previous state
        if os.path.exists(self.ckpt_path):
            with open(self.ckpt_path, "r") as f:
                raw = json.load(f)
                self.previous_state = data_models.Checkpoint.model_validate(raw)
        else:
            logger.warning(
                f"Unable to find previous chackoint at {self.ckpt_path}, starting from scratch."
            )
            self.previous_state = data_models.Checkpoint()

        # Load entities from previous state to memory
        if self.previous_state.existing_entities:
            logger.info(
                "Loading previously extracted entities from checkpoint to memory"
            )
            self.update_entity_memory(self.previous_state.existing_entities)

        # Load previously aggregated results
        if os.path.exists(self.result_path):
            with open(self.result_path, "r") as f:
                raw = json.load(f)
                self.aggregated_results = data_models.AggregatedResults.model_validate(
                    raw)
        else:
            logger.warning(
                f"Could not find previous state for aggregated results at provided path : {self.result_path}, creating new report."
            )
            self.aggregated_results = data_models.AggregatedResults(data={})

    def format_reviews(self, reviews: List[str], batch_start_idx: int) -> str:
        """Formats the batch reviews in a single string.

        Args:
            reviews (List[str]) : List of reviews (current batch)
            batch_start_idx (int) : start index of the batch

        Returns:
            formatted_reviews (str) : Reviews formatted as a string
        """

        formatted_reviews = "\n".join([
            f"review-{batch_start_idx+id} : {review}"
            for id, review in enumerate(reviews)
        ])
        return formatted_reviews

    def process_batch_output(self, model_response: Dict) -> Set[str]:
        """Processes the model's response and updates the aggregated result.
        Args:
            model_response (Dict): The structured response from the model.

        Returns:
            batch_entities (Set[str]): entities recognized in current batch.

        """
        # entities for current batch.
        batch_entities = set()

        for entity_name, reviews in model_response.items():
            # setting defaults, safeguarding LLM's output
            for sentiment in ["positive_reviews", "negative_reviews"]:
                model_response[entity_name].setdefault(sentiment, [])

            # Convert to set for uniqueness
            positive_review_ids = set(reviews["positive_reviews"])
            negative_review_ids = set(reviews["negative_reviews"])

            # Ensuring entity exists in aggregated_result
            if entity_name not in self.aggregated_results:
                self.aggregated_results[
                    entity_name] = data_models.EntitySentimentMap(
                        positive_review_ids=set(), negative_review_ids=set())

            # Update values with unique IDs
            self.aggregated_results[entity_name].positive_review_ids.update(
                positive_review_ids)
            self.aggregated_results[entity_name].negative_review_ids.update(
                negative_review_ids)

            # Append to current batch entities
            batch_entities.add(entity_name)

        return batch_entities

    def update_entity_memory(self, batch_entities: Set[str]) -> None:
        """Updates entity memory with new entities, if any.

        Args:
            batch_entities (Set[str]): entities recognized in current batch.

        returns:
            None
        """

        # Fetch existing entities from memory
        existing_entities = self.previous_state.existing_entities
        entities_added = False

        # Add new entities (avoid duplication)
        for entity in batch_entities:
            if entity not in existing_entities:
                logger.info(f"adding new entity to memory : {entity}")
                existing_entities.append(entity)
                entities_added = True

        if entities_added:
            logger.info("Memory updated".upper())
        else:
            logger.info(
                "Did not encounter any new entity, skipping memory update.")
        logger.info(f"[MEMORY | EXISTING ENTITIES]:\n{existing_entities}\n")
        return

    def process_reviews_in_batches(
            self,
            reviews: List[str],
            batch_size: int = 50) -> data_models.AggregatedResults:
        """Processes user reviews in batches, extracting entities and sentiment from each batch.

        Args:
            reviews (List[str]): A list of user reviews to be analyzed.
            batch_size (int, optional): The number of reviews to process in a single batch. Default is 50.

        Returns:
            aggregated_results (AggregatedResults): A Pydantic object where each key is an entity, and the value is
            an EntitySentimentMap containing "positive_reviews" and "negative_reviews" with sets of review IDS and  count.

        Functionality:
            - skips processed batches using previous state.
            - Splits the list of reviews into smaller batches of size `batch_size`.
            - Generates structured prompts for the model using predefined templates.
            - Calls the LLM model to extract entities and sentiments for each batch.
            - Aggregates extracted entities.
            - Updates memory with newly identified entities.
            - Update the checkpoint after processing each batch.
        """

        checkpoint = self.previous_state

        if checkpoint.batch_size is None:
            checkpoint.batch_size = batch_size
        else:
            assert checkpoint.batch_size == batch_size, f"batch size Mismatch, Checkpoint: {checkpoint.batch_size}, Current: {batch_size}"

        logger.info(
            f"Processing {len(reviews)} reviews in batches of {batch_size}...")
        print("=" * 100)

        # extract reviews and process batch
        for batch_start_idx in tqdm.tqdm(range(0, len(reviews), batch_size)):
            print("- -" * 60)

            # Skip already completed batches
            if checkpoint.last_batch_idx:
                if batch_start_idx <= checkpoint.last_batch_idx:
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
            formatted_reviews = self.format_reviews(
                reviews=batch_reviews, batch_start_idx=batch_start_idx)

            # format the ChatPromptTemplate with system, user prompt
            formatted_prompt = prompts.chat_prompt_template.format(
                system_prompt=prompts.get_system_propmt(
                    existing_entities=checkpoint.existing_entities),
                user_prompt=prompts.get_user_prompt(
                    formatted_reviews=formatted_reviews))

            if batch_start_idx == 0:
                print("=" * 100)
                print(formatted_prompt)
                print("=" * 100)

            try:
                logger.info("Invoking LLM ..")
                t1 = time.perf_counter()
                # LLM call
                response = self.llm.invoke(formatted_prompt)
                t2 = time.perf_counter()
                print(f"time taken to process the batch: {(t2-t1)*1000} ms")

                # Load json string to python dictionary
                response_json = json.loads(response.content.strip("```json"))

                # Process batch output
                batch_entities = self.process_batch_output(response_json)
                logger.info(
                    f"ENTITIES EXTRACTED IN CURRENT BATCH : {batch_entities}\n")
                logger.info("Updating Memory")

                # Update memory
                self.update_entity_memory(batch_entities)

            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_start_idx // batch_size + 1}: {e}"
                )
                logger.info(
                    f"{checkpoint.last_batch_idx//batch_size +1 } batches,i.e,, {checkpoint.last_batch_idx+batch_size} reviews proceced, saving details to {self.result_path}"
                )
                break

            # Save checkpoint and aggregated results after every batch
            checkpoint.last_batch_idx = batch_start_idx
            with open(self.ckpt_path, "w") as f:
                f.write(checkpoint.model_dump_json(indent=4))
            with open(self.result_path, "w") as f:
                f.write(self.aggregated_results.model_dump_json(indent=4))

            time.sleep(2)  # To Prevent rate limit issues
        else:
            logger.info(
                f"All batches proceced, results saved to {self.result_path}")
        return self.aggregated_results


def main(csv_file):

    data = analyzer_utils.load_csv(file_path=csv_file)
    reviews = data["Review"].dropna().tolist()

    analyzer = ReviewAnalyzer(checkpoint_path=constants.ckpt_path,
                              report_path=constants.aggregated_results_path)

    analysis_report = analyzer.process_reviews_in_batches(reviews,
                                                          batch_size=50)


if __name__ == "__main__":
    main("data/spotify_reviews.csv")
