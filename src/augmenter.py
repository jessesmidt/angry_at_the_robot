# Augmenting: Once the AI has retrieved the information, it can combine it with
# what it already “knows.” However, in most practical applications, we try to rely as
# much as possible on the retrieved data rather than the model’s internal knowledge
# — since mixing both may lead to outdated or hallucinatory answers. Starting from
# the retrieved results, you can clean and filter them to remove irrelevant snippets
# (to avoid potential noise), insert them into the context window.