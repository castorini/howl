from dataclasses import dataclass
from typing import Dict, List, Tuple

__all__ = ["FrameLabelData"]


@dataclass
class FrameLabelData:
    # Map of timestamp of which the word finishes and word label
    timestamp_label_map: Dict[float, int]
    # Array of (word label, timestamp of which the word starts)
    start_timestamp: List[Tuple[int, float]]
    # Array of (word label, character index of the word)
    char_indices: List[Tuple[int, List[int]]]
