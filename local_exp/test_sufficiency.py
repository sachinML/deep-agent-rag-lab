
from retrieval_evaluation import evaluate_sufficiency

question = "Compare BM25 and dense retrieval."

# Both concepts are explained.
sufficient_context = [
    (
        "[DOC-002] BM25 uses lexical matching, term frequency, "
        "inverse document frequency, and length normalization. "
        "It works well for exact terminology."
    ),
    (
        "[DOC-003] Dense retrieval uses embeddings and vector "
        "similarity. It can retrieve semantically related documents "
        "without identical terms. Its performance depends on "
        "the embedding model and domain."
    ),
]

# Only BM25 is explained.
insufficient_context = [
    (
        "[DOC-002] BM25 is a lexical retrieval algorithm "
        "that uses term frequency and inverse document frequency."
    )
]

print("\n=== SUFFICIENT CONTEXT ===")

result_1 = evaluate_sufficiency(
    question=question,
    retrieval_context=sufficient_context,
)

print(result_1)

print("\n=== INSUFFICIENT CONTEXT ===")

result_2 = evaluate_sufficiency(
    question=question,
    retrieval_context=insufficient_context,
)

print(result_2)
