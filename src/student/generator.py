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