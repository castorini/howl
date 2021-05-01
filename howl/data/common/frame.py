from dataclasses import dataclass
from typing import List, Mapping, Tuple

__all__ = ["FrameLabelData"]


@dataclass
class FrameLabelData:
    timestamp_label_map: Mapping[float, int]
    start_timestamp: List[Tuple[int, float]]
    char_indices: List[Tuple[int, List[int]]]
