import json
from tqdm import tqdm 
from .models import RagDataset, StudentSearchResults

def calculate_iou(source_a, source_b) -> float:
    intersection: int = max(0, min(source_a.last_character_index, source_b.last_character_index)
        - max(source_a.first_character_index, source_b.first_character_index))
    union: int = (max(source_a.last_character_index, source_b.last_character_index)
                  - min(source_a.first_character_index, source_b.first_character_index))
    iou: float = intersection / union
                    
    return iou
    
def recall_at_k(student_results, dataset_truth) -> float:
    with open(dataset_truth) as f:
        truth = RagDataset(**json.load(f))
    with open(student_results) as f:
        student_answers = StudentSearchResults(**json.load(f))
    
    questions_dict = {q.question_id: q for q in truth.rag_questions}
    tally: float = 0 
    for result in student_answers.search_results:
        ground = questions_dict[result.question_id]
        sources_found = 0

        #IoU
        for source in ground.sources:
            found = False
            for student_src in result.retrieved_sources:
                if source.file_path == student_src.file_path:
                    if calculate_iou(source, student_src) >= 0.05:
                        found = True
                        break
            if found:
                sources_found += 1

        tally += sources_found / len(ground.sources)

    return tally / len(student_answers.search_results)