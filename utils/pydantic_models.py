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


class EntityStats(BaseModel):
    """Represents statistics for a sentiment category of an entity.

    Attributes:
        count (int): Total number of unique reviews in this sentiment category.
        ids (Set[int]): Set of review IDs corresponding to this sentiment.
    """
    count: int
    ids: Set[int]

    class Config:
        json_encoders = {
            set: list  # Automatically convert sets to lists when saving
        }


class EntitySentimentMap(BaseModel):
    """Holds sentiment-wise review statistics for a single entity.

    Attributes:
        positive_reviews (EntityStats): Statistics for positive sentiment.
        negative_reviews (EntityStats): Statistics for negative sentiment.
    """

    positive_reviews: EntityStats
    negative_reviews: EntityStats

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
