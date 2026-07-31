# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a technique for grounding a language model's
answers in an external collection of documents. Instead of relying only on the
knowledge baked into the model's weights, a RAG system first retrieves the most
relevant passages from a document store and then asks the model to answer using
only those passages.

The main benefit of RAG is that answers can cite their sources, which makes them
verifiable. It also lets the system stay up to date: adding a new document to the
store is enough to make its contents answerable, with no model retraining.

A typical RAG pipeline has four stages: ingestion (loading and chunking
documents), embedding (turning text into vectors), retrieval (finding the
nearest chunks to a query), and generation (writing a grounded answer). When no
retrieved passage is relevant, a well-behaved RAG system should refuse to answer
rather than inventing a response.
