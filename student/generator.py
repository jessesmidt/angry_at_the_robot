# Augmenting: Once the AI has retrieved the information, it can combine it with
# what it already “knows.” However, in most practical applications, we try to rely as
# much as possible on the retrieved data rather than the model’s internal knowledge
# — since mixing both may lead to outdated or hallucinatory answers. Starting from
# the retrieved results, you can clean and filter them to remove irrelevant snippets
# (to avoid potential noise), insert them into the context window.

# Generating: Now that you have retrieved the information and augmented it,
# the AI can finally generate an answer! Whether it’s writing text, explaining a
# concept, or producing code snippets, this is the visible outcome of RAG. To do
# so, the AI reads the context window, understands the task at hand, blends the
# knowledge, and generates the output. Modern RAG systems often refine while
# writing, adjusting phrasing on the fly to maintain coherence and match the tone
# requested in the query.

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from .models import Chunk

class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32
        )

    # def _format_prompt(self, question: str, chunks: list[Chunk]) -> str:
    #     """
    #     Catenates all the found chunks and creates the prompt
    #     for the LLM

    #     Args: 
    #         question = the input prompt
    #         chunks = A list of chunks in which the required
    #             context is given

    #     Returns:
    #         The prompt ready to send to the LLM
    #     """
    #     context = "\n\n".join([chunk.content for chunk in chunks])
    #     return f"""You're a RAG agent. Here is your context:

    #     Context: {context}
        

    #     Answer the question below based ONLY on the given context.
    #     Do not use any external knowledge or URLs not present in the context.
    #     ONLY cite the used source file's filepath. 
    #     Keep your answer to 1-2 sentences maximum. do NOT repeat 

    #     Question: {question}

    #     Answer:"""

    def generate(self, question: str, chunks: list[Chunk], stream: bool) -> str:
        """
        Generates an answer using the LLM. Sends a question and
        all context to the class' model

        Args: 
            question = the input prompt
            chunks = A list of chunks in which the required
                context is given

        Returns:
            The LLM's answer to our question in a string.
        """
        context = "\n\n".join([chunk.content for chunk in chunks])
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the provided context. Keep answers to 2-3 sentences. Do not hallucinate."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
