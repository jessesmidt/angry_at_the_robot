https://pypi.org/project/bm25s/
https://huggingface.co/docs/transformers/main_classes/tokenizer
https://docs.python.org/3/library/ast.html
https://huggingface.co/Qwen/Qwen3-0.6B
https://github.com/xhluca/bm25s
https://github.com/google/python-fire/blob/master/docs/guide.md
https://huggingface.co/docs/transformers/model_doc/auto
https://tqdm.github.io/


 <!-- test generation -->

  uv run python -m student answer "How does vLLM handle batching?" --k 5 

 <!-- docs moulinette recall@k check -->

cd /home/jsmidt/Desktop/CC/C4/rag/data/moulinette/moulinette_pkg
chmod +x moulinette-ubuntu
./moulinette-ubuntu evaluate_student_search_results \
    ../../output/search_results/dataset_docs_public.json \
    ../../datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
    --k 5 \
    --max_context_length 2000 \
    --threshold 0.80

./moulinette-ubuntu list_valid_questions \
    ../../output/search_results/dataset_docs_public.json \
    ../../datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
    --k 5

<!-- code moulinette recall@k check -->

./moulinette-ubuntu evaluate_student_search_results \
    ../../output/search_results/dataset_code_public.json \
    ../../datasets_public/public/AnsweredQuestions/dataset_code_public.json \
    --k 5 \
    --max_context_length 2000 \
    --threshold 0.50

<!-- check valid questions -->

./moulinette-ubuntu list_valid_questions \
    ../../output/search_results/dataset_code_public.json \
    ../../datasets_public/public/AnsweredQuestions/dataset_code_public.json \
    --k 5

../../..

