import os
import json
from .models import RagDataset
from .models import Chunk, MinimalSource

def dataset_loader(filename: str) -> RagDataset:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset file '{filename}' was not found"
        )
    with open(filename, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in '{filename}': {e}"
            )
    return RagDataset(**data)


def chunk_to_source(chunk: Chunk) -> MinimalSource:
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index
    )
