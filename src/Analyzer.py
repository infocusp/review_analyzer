import json
import sys
import time
from typing import List, Set

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

import prompts

sys.path.append("..")
from Utils.analyzer_utils import load_analysis_report
from Utils.analyzer_utils import load_checkpoint
from Utils.analyzer_utils import load_csv
from Utils.analyzer_utils import Logger
from Utils.analyzer_utils import save_analysis_report
from Utils.analyzer_utils import save_checkpoint

Mylogger = Logger("Review Analyzer")
logger = Mylogger.get_logger()


class ReviewAnalyzer:

    def __init__(self,
                 save_checkpoint_path="Checkpoint.json",
                 save_result_path="Analysis_report.json"):

        # load the API Key from env variable
        load_dotenv()
        # Initialize Gemini model (LangChain Wrapper)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        # Memory to track previously extracted entities
        self.memory = ConversationBufferMemory(memory_key="entities",
                                               return_messages=True)
        self.ckpt_path = save_checkpoint_path
        self.result_path = save_result_path

        self.aggregated_results = {}

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

    def process_batch_output(self, model_response: dict) -> Set[str]:
        """
        Processes the model's JSON response, updates the aggregated results, andmaintains entity memory.

        Args:
            model_response (dict): The structured response from the model.

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

            # Ensuring entity exists in all_result
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

            batch_entities.add(entity_name)

        return batch_entities

    def update_entity_memory(self, batch_entities: Set[str]) -> None:
        """
        Updates entity memory with new entities if any.

        Args:
            batch_entities (Set[str]): entities recognized in current batch.

        returns:
            None
        """

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
            # print(f"entity list after merging new entities: \n{entity_list}")
            logger.info("Memory updated".upper())
        else:
            logger.info(
                "Did not encounter any new entity, skipping memory update.")
        logger.info(f"[MEMORY | EXISTING ENTITIES]:\n{existing_entities}\n")
        return

    def process_reviews_in_batches(self,
                                   reviews: List[str],
                                   batch_size: int = 50) -> dict:
        """
        Processes user reviews in batches, extracting entities and sentiment from each batch.

        Args:
            reviews (List[str]): A list of user reviews to be analyzed.
            batch_size (int, optional): The number of reviews to process in a single batch. Default is 50.

        Returns:
            analysis_report (dict): A dictionary where each key is an entity and the value is a dictionary with 
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

        checkpoint = load_checkpoint(self.ckpt_path)

        if checkpoint["batch_size"] is None:  # Register batch-size in checkpoint
            checkpoint["batch_size"] = batch_size
        else:  # Ensure the batch-size is same as used in previous checkpoint
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
            self.aggregated_results = load_analysis_report(self.result_path)

        # extract reviews and process batch
        for batch_start_idx in tqdm(range(0, len(reviews), batch_size)):
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

            # print(f"Input :\n{formatted_reviews}\n")

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
                response = self.llm.invoke(formatted_prompt)
                t2 = time.perf_counter()
                print(f"time taken : {(t2-t1)*1000} ms")
                response_json = json.loads(response.content.strip("```json"))
                batch_entities = self.process_batch_output(response_json)
                logger.info(
                    f"ENTITIES EXTRACTED IN CURRENT BATCH : {batch_entities}\n")
                logger.info("Updating Memory")
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

            save_checkpoint(checkpoint, file_path=self.ckpt_path)
            save_analysis_report(self.aggregated_results,
                                 file_path=self.result_path)
            time.sleep(2)  # To Prevent rate limit issues
        else:
            logger.info(
                f"All batches proceced, results saved to {self.result_path}")
        return self.aggregated_results


def main(csv_file):

    data = load_csv(csv_file)
    reviews = data["Review"].dropna().tolist()

    save_checkpoint_path = "./Checkpoint.json"
    save_result_path = "Analysis_report.json"

    analyzer = ReviewAnalyzer(save_checkpoint_path, save_result_path)
    analysis_report = analyzer.process_reviews_in_batches(reviews,
                                                          batch_size=50)


if __name__ == "__main__":
    main("../data/spotify_reviews.csv")
