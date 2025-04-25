"""This file contains pydantic data models."""

from typing import Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel
from pydantic import Field


class AggregatedResults(BaseModel):
    """Container for mapping entities to their sentiment statistics.

    Attributes:
        entity_sentiment_map (Dict[str, Dict[str, Set[int]]]): A nested dictionary where:
        - The outer key is the entity name.
        - The inner dictionary contains:
            - "positive_review_ids": a set of review IDs expressing positive sentiment.
            - "negative_review_ids": a set of review IDs expressing negative sentiment.
    """
    entity_sentiment_map: Dict[str, Dict[str,
                                         Set[int]]] = Field(description=("""
            Structured output format mapping each entity name (string) to its sentiment-based review IDs.
            Each entity maps to a dictionary with two keys: 'positive_review_ids' and 'negative_review_ids',
            each containing a set of integers representing associated review IDs.
            """))
    batch_size: Optional[int] = None
    last_batch_idx: Optional[int] = None

    @property
    def existing_entities(self) -> List[str]:
        return list(self.entity_sentiment_map.keys())

    def update(self, model_response: "AggregatedResults",
               batch_idx: int) -> None:
        """Merges the contents of a validated model response into the current AggregatedResults instance.

        Args:
            model_response (AggregatedResults): The validated model output.
            batch_idx (int): index of the processed batch.

        Returns:
            None
        """
        self.last_batch_idx = batch_idx
        for entity_name, sentiment_map in model_response.items():
            if entity_name not in self.entity_sentiment_map:
                self.entity_sentiment_map[entity_name] = sentiment_map
            else:
                if "positive_review_ids" in sentiment_map:
                    self.entity_sentiment_map[entity_name][
                        "positive_review_ids"].update(
                            sentiment_map["positive_review_ids"])
                if "negative_review_ids" in sentiment_map:
                    self.entity_sentiment_map[entity_name][
                        "negative_review_ids"].update(
                            sentiment_map["negative_review_ids"])
        return

    def __getitem__(self, key: str) -> Dict[str, Set[int]]:
        return self.entity_sentiment_map[key]

    def __setitem__(self, key: str, value: Dict[str, Set[int]]) -> None:
        self.entity_sentiment_map[key] = value

    def __delitem__(self, key: str) -> None:
        del self.entity_sentiment_map[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entity_sentiment_map)

    def __len__(self) -> int:
        return len(self.entity_sentiment_map)

    def keys(self) -> Iterator[str]:
        return iter(self.entity_sentiment_map.keys())

    def values(self) -> Iterator[Dict[str, Set[int]]]:
        return iter(self.entity_sentiment_map.values())

    def items(self) -> Iterator[Tuple[str, Dict[str, Set[int]]]]:
        return iter(self.entity_sentiment_map.items())
