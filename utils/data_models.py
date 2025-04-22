import json
from typing import Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel


class Checkpoint(BaseModel):
    """Represents the checkpoint state used to resume processing.

    Attributes:
        batch_size (Optional[int]): Number of reviews processed per batch (None if unknown).
        last_batch_idx (Optional[int]): Index of the last processed batch (None if starting fresh).
        existing_entities (List[str]): List of entity names already encountered.
    """
    batch_size: Optional[int] = None
    last_batch_idx: Optional[int] = None
    existing_entities: List[str] = []


class EntitySentimentMap(BaseModel):
    """Holds sentiment-wise review statistics for a single entity.

    Attributes:
        positive_review_ids (Set[int]): Set of review IDs for positive sentiment.
        negative_review_ids (Set[int]): Set of review IDs for negative sentiment.

    Properties:
        positive_count (int): The total number of positive reviews.
        negative_count (int): The total number of negative reviews.
    """

    positive_review_ids: Set[int]
    negative_review_ids: Set[int]

    @property
    def positive_count(self) -> int:
        return len(self.positive_review_ids)

    @property
    def negative_count(self) -> int:
        return len(self.negative_review_ids)

    class Config:
        json_encoders = {set: list}


class AggregatedResults(BaseModel):
    """Container for mapping entities to their sentiment statistics.

    Attributes:
        data (Dict[str, EntitySentimentMap]): Mapping from entity name to sentiment breakdown.
    """
    data: Dict[str, EntitySentimentMap]

    def __getitem__(self, key: str) -> EntitySentimentMap:
        return self.data[key]

    def __setitem__(self, key: str, value: EntitySentimentMap) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Iterator[str]:
        return iter(self.data.keys())

    def values(self) -> Iterator[EntitySentimentMap]:
        return iter(self.data.values())

    def items(self) -> Iterator[Tuple[str, EntitySentimentMap]]:
        return iter(self.data.items())
