import os
import sys
import fire
from pydantic import BaseModel
from .dataset import dataset_loader
from .generator import Generator
from .indexer import index_repository
from .retriever import Retriever

from .models import MinimalSource, MinimalSearchResults, StudentSearchResults

# 1. Ingest the vLLM repository (provided as attachment) and create a searchable
# knowledge base
# 2. Search this knowledge base to find relevant code snippets and documentation for
# given questions
# 3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
# 4. Evaluate your retrieval system’s quality using recall@k metrics

class RAG(BaseModel):
    def __init__(self) -> None:
        pass

    def index(self, max_chunk_size: int = 2000) -> None:
        index_repository("data/raw/vllm-0.10.1", max_chunk_size)
        # print(f"Succesfully indexed data into max chunk size {max_chunk_size}")

    def search(self, query: str, k: int = 5) -> None:
        retriever = Retriever()
        results = retriever.retrieve(query, k)
        for chunk in results:
            print(f"File: {chunk.file_path}")
            print(f"Chars: {chunk.first_character_index} -> {chunk.last_character_index}")
            print(f"Content: {chunk.content[:100]}...")
            print("---")
    
    def search_dataset(
            self,
            dataset_path: str,
            k: int = 5,
            save_dir: str = "data/output/search_results"
            ) -> None:
        retriever = Retriever()
        try:
            dataset = dataset_loader(dataset_path)
        except FileNotFoundError as e:
            print(f"Dataset Error: {e}")
            sys.exit(1)

        search_results = []
        for question in dataset.rag_questions:
            chunks = retriever.retrieve(question.question, k)
            sources = [MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            ) for chunk in chunks]

            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=sources
            ))

        output = StudentSearchResults(search_results=search_results, k=k)
        os.makedirs(save_dir, exist_ok= True)
        filename = os.path.basename(dataset_path)
        with open(os.path.join(save_dir, filename), 'w') as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Succesfully written search results to: {save_dir}")

    def answer(self, query: str, k: int = 5) -> None:
        retriever = Retriever()
        generator = Generator()
        results = retriever.retrieve(query, k)
        print(generator.generate(query, results))


    def answer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

# reminder: implement testing for vLLM presence.

if __name__ == "__main__":
    fire.Fire(RAG)