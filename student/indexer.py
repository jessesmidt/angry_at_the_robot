# chunks vLLM repo, saves index to data/processed

import bm25s
import os, json, ast
from .models import Chunk

def find_char_index(content: str, line_number: int) -> int:
    """
    Finds the index for given function / class definition

    Args: 
        content: the full data within file in str
        line_number: pointer within content in int

    Returns:
        integer for position 
    """
    lines = content.split('\n')
    return sum(len(lines[i]) + 1 for i in range(line_number - 1))


def walk_repo(repo_path: str) -> list[str]:
    """
    Find all the filepaths within the repository.

    Args:
        repo_path: Path to data

    Returns:
        A list of directory paths
    """
    filepaths = []
    for root, _, files in os.walk(repo_path):
        if 'test' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.md') or \
                file.endswith('.txt') or file.endswith('.jinja'):
                filepaths.append(os.path.join(root, file))
    return filepaths


def split_by_size(
        chunk_text: str, start_char: int, max_chunk_size: int, filepath: str
        ) -> list[Chunk]:
    """
    Splits chunk into chunks smaller then max_chunk_size.
    Optimalizations for newlines.

    Args:
        chunk_text: The whole chunk as a string
        start_char: Starting pos for chunk
        max_chunk_size: Max chars per chunk
        filepath: path to data as str

    Returns:
        A list of chunks that fit withing max_chunk_size
    """
    chunks: list[Chunk] = []
    offset: int = 0

    while offset < len(chunk_text):
        end = offset + max_chunk_size

        if end < len(chunk_text):
            newline = chunk_text.rfind('\n', offset, end)
            if newline != -1:
                end = newline + 1

        piece = chunk_text[offset:end]
        chunks.append(Chunk(
            file_path=filepath,
            first_character_index=start_char + offset,
            last_character_index=start_char + offset + len(piece),
            content=piece,
            chunk_id=0
        ))
        offset = end
    return chunks


def chunk_python_file(filepath: str, max_chunk_size: int) -> list[Chunk]:
    """
    Args:
        filepath: path to data as str
        max_chunk_size: Max chars per chunk
        
    Returns:
        A list of chunks that fit withing max_chunk_size
    """
    with open(filepath, 'r') as f:
        content = f.read()

    tree = ast.parse(content)

    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.end_lineno:
            start_char = find_char_index(content, node.lineno)
            end_char = find_char_index(content, node.end_lineno + 1)
            chunk_text = content[start_char:end_char]

            if len(chunk_text) > max_chunk_size:
                sub_chunks = split_by_size(
                    chunk_text, start_char, max_chunk_size, filepath
                    )
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    file_path=filepath,
                    first_character_index=start_char,
                    last_character_index=end_char,
                    content=chunk_text,
                    chunk_id=len(chunks)
                ))
    return chunks


# def group_tables(content: str) -> str:
#     """
#         Replaces table blocks with single paragraph units
#         so they don't get split mid-table.
#     """
#     lines = content.split('\n')
#     result = []
#     in_table = False
#     table_buffer = []
#     for line in lines:
#         if line.strip().startswith('|'):
#             in_table = True
#             table_buffer.append(line)
#         else:
#             if in_table:
#                 result.append('\n'.join(table_buffer))
#                 table_buffer = []
#                 in_table = False
#             result.append(line)
#     if table_buffer:
#         result.append('\n'.join(table_buffer))

#     return '\n'.join(result)

def chunk_text_file(filepath: str, max_chunk_size: int) -> list[Chunk]:
    """
    Args:
        filepath: path to data as str
        max_chunk_size: Max chars per chunk
        
    Returns:
        A list of chunks that fit withing max_chunk_size
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # content = group_tables(content)
    paragraphs: list[str] = content.split('\n\n')

    chunks: list[Chunk] = []
    current_chunk: str = ""
    current_start: int = 0

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(Chunk(
                    file_path=filepath,
                    first_character_index=current_start,
                    last_character_index=current_start + len(current_chunk),
                    content=current_chunk,
                    chunk_id=len(chunks)
                ))
                current_start += len(current_chunk)
                current_chunk = ""

            if len(paragraph) > max_chunk_size:
                sub_chunks = split_by_size(paragraph, current_start, max_chunk_size, filepath)
                chunks.extend(sub_chunks)
                current_start += len(paragraph)
            else:
                current_chunk = paragraph
        else:
            current_chunk += paragraph + '\n\n'

    if current_chunk:
        chunks.append(Chunk(
            file_path=filepath,
            first_character_index=current_start,
            last_character_index=current_start + len(current_chunk),
            content=current_chunk,
            chunk_id=len(chunks)
        ))
    return chunks


def save_chunks(chunks: list[Chunk], path: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "chunks.json"), 'w') as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)


def index_repository(repo_path: str, max_chunk_size: int = 2000) -> None:
    all_chunks: list[Chunk] = []
    chunks: list[Chunk] = []

    for filepath in walk_repo(repo_path):
        print(filepath)
        if filepath.endswith('.py'):
            chunks = chunk_python_file(filepath, max_chunk_size)
        elif filepath.endswith('.md'):
            chunks = chunk_text_file(filepath, max_chunk_size)
        all_chunks.extend(chunks)
        
    corpus = [chunk.content for chunk in all_chunks]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))

    retriever.save("data/processed/bm_25_index")
    save_chunks(all_chunks, "data/processed/chunks")

def load_chunks(path: str) -> list[Chunk]:
    with open(os.path.join(path, "chunks.json"), 'r') as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]
