# Chunking Documents

Chunking is the process of splitting a document into smaller pieces before
embedding them. Chunk size is a trade-off: chunks that are too large dilute the
embedding and retrieve irrelevant text, while chunks that are too small lose the
surrounding context needed to answer a question.

A common starting point is a few hundred to around a thousand characters per
chunk, with a small overlap between consecutive chunks. The overlap ensures that
a fact sitting on a chunk boundary is still captured whole in at least one chunk.

Chunking should be word-aware so that tokens are not split in the middle. Each
chunk carries metadata about its source document and its position, so that
retrieved text can be cited back to a precise location.
