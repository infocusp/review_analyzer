import json
import sys
import time
from typing import Dict, List, Set, TypedDict

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import tqdm

import prompts

sys.path.append("..")
from utils import analyzer_utils

Mylogger = analyzer_utils.Logger("Review Analyzer")
logger = Mylogger.get_logger()


class EntityStats(TypedDict):
    """
    Represents review statistics for an entity across sentiment categories (positive or negative).

    Attributes:
        count: Total number of reviews in this category.
        ids: A set of review IDs associated with this category.
    """
    count: int
    ids: Set[int]


class EntitySentimentMap(TypedDict):
    """
    Represents categorized review data for a single entity.

    Attributes:
        positive_reviews: Statistics and IDs for positive reviews.
        negative_reviews: Statistics and IDs for negative reviews.
    """
    positive_reviews: EntityStats
    negative_reviews: EntityStats


class ReviewAnalyzer:
    """
    Class to analyze user reviews using LLM.
    """

    def __init__(self,
                 checkpoint_path: str = "Checkpoint.json",
                 report_path: str = "Analysis_report.json"):
        """
        ReviewAnalyzer parameters initialization

        Args:
            checkpoint_path (str) : path to save/load checkpoint.
            report_path (str) : path to save/load the aggregated results(json report).
        """
        # load the API Key from env variable
        load_dotenv()
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        # Memory to track previously extracted entities
        self.memory = ConversationBufferMemory(memory_key="entities",
                                               return_messages=True)
        self.ckpt_path = checkpoint_path
        self.result_path = report_path

        self.aggregated_results: Dict[str, EntitySentimentMap] = {}

    def get_existing_entities(self, key: str) -> List[str]:
        """
        Extract stored entities from memory

        Args:
            key (str) : key under which the desired data is stored in memory.

        Returns:
            existing_entities (List[str]) : Stored entities
        """

        existing_entities = []
        # Load existing memory
        current_memory = self.memory.load_memory_variables({})
        stored_messages = current_memory.get(key, [])

        for msg in stored_messages:
            if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                # Extract stored entities from AIMessage
                existing_entities = msg.content

        return existing_entities

    def process_batch_output(self, model_response: Dict) -> Set[str]:
        """
        Processes the model's response, updates the aggregated results, and maintains entity memory.

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
            positive_reviews = set(reviews["positive_reviews"])
            negative_reviews = set(reviews["negative_reviews"])

            # Ensuring entity exists in aggregated_result
            if entity_name not in self.aggregated_results:
                self.aggregated_results[entity_name] = {
                    "positive_reviews": {
                        "count": 0,
                        "ids": set()
                    },
                    "negative_reviews": {
                        "count": 0,
                        "ids": set()
                    }
                }

            # Update values with unique IDs
            self.aggregated_results[entity_name]["positive_reviews"][
                "ids"].update(positive_reviews)
            self.aggregated_results[entity_name]["negative_reviews"][
                "ids"].update(negative_reviews)

            # Update counts based on unique IDs
            self.aggregated_results[entity_name]["positive_reviews"][
                "count"] = len(self.aggregated_results[entity_name]
                               ["positive_reviews"]["ids"])
            self.aggregated_results[entity_name]["negative_reviews"][
                "count"] = len(self.aggregated_results[entity_name]
                               ["negative_reviews"]["ids"])

            # Append to current batch entities
            batch_entities.add(entity_name)

        return batch_entities

    def update_entity_memory(self, batch_entities: Set[str]) -> None:
        """
        Updates entity memory with new entities, if any.

        Args:
            batch_entities (Set[str]): entities recognized in current batch.

        returns:
            None
        """

        # Fetch existing entities from memory
        existing_entities = self.get_existing_entities(key="entities")
        entities_added = False

        # Add new entities (avoid duplication)
        for entity in batch_entities:
            if entity not in existing_entities:
                logger.info(f"adding new entity to memory : {entity}")
                existing_entities.append(entity)
                entities_added = True

        if entities_added:
            # Save updated memory
            self.memory.clear()
            self.memory.save_context({"input": "Updating entity memory"},
                                     {"entities": existing_entities})
            logger.info("Memory updated".upper())
        else:
            logger.info(
                "Did not encounter any new entity, skipping memory update.")
        logger.info(f"[MEMORY | EXISTING ENTITIES]:\n{existing_entities}\n")
        return

    def process_reviews_in_batches(self,
                                   reviews: List[str],
                                   batch_size: int = 50) -> Dict:
        """
        Processes user reviews in batches, extracting entities and sentiment from each batch.

        Args:
            reviews (List[str]): A list of user reviews to be analyzed.
            batch_size (int, optional): The number of reviews to process in a single batch. Default is 50.

        Returns:
            analysis_report (Dict): A dictionary where each key is an entity and the value is a dictionary with 
                "positive_reviews" and "negative_reviews" as keys mapping to lists of review IDs.

        Functionality:
            - Loads previous checkpoint if available, skips processed batches.
            - Splits the list of reviews into smaller batches of size `batch_size`.
            - Generates structured prompts for the model using predefined templates.
            - Calls the LLM model to extract entities and sentiments for each batch.
            - Aggregates extracted entities.
            - Updates memory with newly identified entities.
            - Update the checkpoint after processing each batch.
        """

        checkpoint = analyzer_utils.load_checkpoint(self.ckpt_path)

        if checkpoint["batch_size"] is None:
            checkpoint["batch_size"] = batch_size
        else:
            assert checkpoint[
                "batch_size"] == batch_size, f"batch size Mismatch, Checkpoint: {checkpoint['batch_size']}, Current: {batch_size}"

        logger.info(
            f"Processing {len(reviews)} reviews in batches of {batch_size}...")
        print("=" * 100)

        if bool(checkpoint["existing_entities"]):
            logger.info(
                "Loading previously extracted entities from checkpoint to memory"
            )
            self.update_entity_memory(checkpoint["existing_entities"])
            self.aggregated_results = analyzer_utils.load_analysis_report(
                self.result_path)

        # extract reviews and process batch
        for batch_start_idx in tqdm.tqdm(range(0, len(reviews), batch_size)):
            print("- -" * 60)

            # Skip already completed batches
            if checkpoint["last_batch_idx"]:
                if batch_start_idx <= checkpoint["last_batch_idx"]:
                    logger.info(
                        f"Skipping batch {batch_start_idx//batch_size}[reviews {batch_start_idx} - {batch_start_idx+batch_size}], already processed."
                    )
                    continue

            # Load batch and format input
            logger.info(
                f"Loading batch {batch_start_idx // batch_size + 1}, Reviews {batch_start_idx}-{batch_start_idx+batch_size}\n"
            )
            batch = reviews[batch_start_idx:batch_start_idx + batch_size]
            formatted_reviews = "\n".join([
                f"review-{batch_start_idx+id} : {review}"
                for id, review in enumerate(batch)
            ])

            # fetch existing entities from memory
            existing_entities = self.get_existing_entities(key="entities")

            formatted_prompt = prompts.chat_prompt_template.format(
                system_prompt=prompts.get_system_propmt(existing_entities),
                user_prompt=prompts.get_user_prompt(formatted_reviews))

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
                    f"{checkpoint['last_batch_idx']//batch_size +1 } batches,i.e,, {checkpoint['last_batch_idx']+batch_size} reviews proceced, saving details to {self.result_path}"
                )
                break

            # Save checkpoint after every batch
            checkpoint['last_batch_idx'] = batch_start_idx
            checkpoint["existing_entities"] = self.get_existing_entities(
                key="entities")

            analyzer_utils.save_checkpoint(checkpoint, file_path=self.ckpt_path)
            analyzer_utils.save_analysis_report(self.aggregated_results,
                                                file_path=self.result_path)
            time.sleep(2)  # To Prevent rate limit issues
        else:
            logger.info(
                f"All batches proceced, results saved to {self.result_path}")
        return self.aggregated_results


def main(csv_file):

    data = analyzer_utils.load_csv(csv_file)
    reviews = data["Review"].dropna().tolist()

    checkpoint_path = "./Checkpoint.json"
    report_path = "Analysis_report.json"

    analyzer = ReviewAnalyzer(checkpoint_path, report_path)
    analysis_report = analyzer.process_reviews_in_batches(reviews,
                                                          batch_size=50)


if __name__ == "__main__":
    main("../data/spotify_reviews.csv")
