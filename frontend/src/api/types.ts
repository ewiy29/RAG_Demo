// TypeScript mirrors of the FastAPI response models (see src/rag/api.py).
// Kept in one place so components consume typed data instead of loose JSON.

export interface Citation {
  marker: number;
  source: string;
  chunk_index: number;
  score: number;
  quote: string;
}

export interface Retrieved {
  id: string;
  source: string;
  chunk_index: number;
  score: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatResponse {
  query: string;
  answer: string;
  grounded: boolean;
  citations: Citation[];
  retrieved: Retrieved[];
  usage: Usage;
  latency_ms: number;
  user_id: string;
  conversation_id: string;
}

export interface FileError {
  source: string;
  code: string;
  context: Record<string, unknown>;
}

export interface IngestResponse {
  documents: number;
  chunks: number;
  files: string[];
  failures: FileError[];
  user_id: string;
}

export interface DocumentInfo {
  source: string;
  chunks: number;
}

export interface DocumentListResponse {
  user_id: string;
  documents: DocumentInfo[];
}

export interface DeleteDocumentResponse {
  user_id: string;
  source: string;
  status: string;
}

// The structured, prose-free error envelope the API returns for typed failures.
export interface ErrorEnvelope {
  error: {
    domain: string;
    code: string;
    context: Record<string, unknown>;
  };
}
