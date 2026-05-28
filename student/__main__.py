# pyright: reportMissingImports=false

import os
import sys
import fire
import json
import time
from tqdm import tqdm
from pydantic import BaseModel
from .dataset import dataset_loader
from .generator import Generator
from .indexer import index_repository
from .retriever import Retriever



from .models import StudentSearchResultsAndAnswer, MinimalAnswer, MinimalSource, MinimalSearchResults, StudentSearchResults, Chunk

# 1. Ingest the vLLM repository (provided as attachment) and create a searchable
# knowledge base
# 2. Search this knowledge base to find relevant code snippets and documentation for
# given questions
# 3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
# 4. Evaluate your retrieval system’s quality using recall@k metrics

class RAG:
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
        self.generator = Generator()
        results = retriever.retrieve(query, k)
        print("Answer:\n")
        self.generator.generate(query, results, True)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        
        """
        start_time = time.time()
        with open(student_search_results_path, 'r') as f:
            search_results = StudentSearchResults(**json.load(f))

        answers: list[MinimalAnswer] = []
    
        for result in tqdm(search_results.search_results, leave=False):
            chunks: list[Chunk] = []
            for source in result.retrieved_sources:
                with open(source.file_path, 'r') as f:
                    content = f.read()
                chunk_text = content[source.first_character_index:source.last_character_index]
                chunks.append(Chunk(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    content=chunk_text,
                    chunk_id=0
                ))

            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question_str=result.question_str,
                retrieved_sources=result.retrieved_sources,
                answer=self.generator.generate(result.question_str, chunks, False)
            ))

        output = StudentSearchResultsAndAnswer(search_results=answers, k=search_results.k)
        os.makedirs(os.path.dirname(f"{save_directory}.json"), exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        with open(f"{save_directory}.json", 'w') as f:
            f.write(output.model_dump_json(indent=2))

        print(
            f"Completed answering questions in {(time.time() - start_time):.0f} seconds"
            )

    def evaluate(self, student_answer_path: str, dataset_path: str, max_context_length: int = 2000) -> None:
        from .eval import recall_at_k
        for k_val in [1, 3, 5, 10]:
            score = recall_at_k(student_answer_path, dataset_path)
            print(f"Recall@{k_val}: {score:.3f} ({score*100:.1f}%)")







    # reminder: implement testing for vLLM presence.

    

if __name__ == "__main__":
    fire.Fire(RAG)