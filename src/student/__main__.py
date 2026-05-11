import os
import sys
from retriever import Retriever
from .dataset import dataset_loader()

# 1. Ingest the vLLM repository (provided as attachment) and create a searchable
# knowledge base
# 2. Search this knowledge base to find relevant code snippets and documentation for
# given questions
# 3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
# 4. Evaluate your retrieval system’s quality using recall@k metrics

def main() -> None:
    try:
        from data.vllm import LLM
    except ImportError:
        print(
            "Error: vllm not found. Please copy the vllm folder "
            "into the project's 'data' folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = Retriever()
    try:
        dataset = dataset_loader("data/datasets/UnansweredQuestions/dataset_docs_public.json")
    except FileNotFoundError as e:
        print(f"Dataset Error: {e}")
        sys.exit(1)

    for question in dataset.rag_questions:
        results = retriever.retrieve(question.question, k=5)


if __name__ == "__main__":
    main()