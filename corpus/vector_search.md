# Vector Search and Embeddings

An embedding is a list of numbers that represents the meaning of a piece of
text. Texts with similar meaning have embeddings that point in similar
directions, so their similarity can be measured with the cosine of the angle
between the two vectors.

Vector search stores an embedding for every chunk and, given a query embedding,
returns the chunks whose vectors are closest. A similarity threshold can be
applied so that only sufficiently relevant chunks are returned; if nothing
clears the threshold, the query is treated as out of scope.

Chroma is a local vector database that persists embeddings to disk and requires
no external server, which makes it convenient for small projects. Larger systems
often move to a database such as pgvector once scale or operational concerns
demand it.
