// Thin fetch wrapper around the RAG API.
//
// Responsibilities kept here (not in components/hooks):
//   * Persist the minted tenant id (X-User-Id) in localStorage and send it on
//     every request; adopt the id the server echoes back so a fresh client
//     keeps a stable identity across reloads.
//   * Thread the conversation id (X-Conversation-Id) returned by /chat so
//     multi-turn history continues on later turns.
//   * Translate the backend's typed error envelope into a thrown ApiError the
//     UI can render off a stable `code` rather than parsing prose.

import type {
  ChatResponse,
  DeleteDocumentResponse,
  DocumentListResponse,
  ErrorEnvelope,
  IngestResponse,
} from "./types";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

const USER_ID_KEY = "rag.userId";
const CONVERSATION_ID_KEY = "rag.conversationId";
const USER_ID_HEADER = "X-User-Id";
const CONVERSATION_ID_HEADER = "X-Conversation-Id";

/** A typed API failure carrying the backend's machine-readable code + context. */
export class ApiError extends Error {
  readonly status: number;
  readonly domain: string;
  readonly code: string;
  readonly context: Record<string, unknown>;

  constructor(
    status: number,
    domain: string,
    code: string,
    context: Record<string, unknown> = {},
  ) {
    super(`${domain}:${code}`);
    this.name = "ApiError";
    this.status = status;
    this.domain = domain;
    this.code = code;
    this.context = context;
  }
}

function readStored(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function writeStored(key: string, value: string): void {
  try {
    if (value) {
      window.localStorage.setItem(key, value);
    }
  } catch {
    // Ignore storage failures (private mode, disabled storage): the app still
    // works for the current session, identity just won't persist.
  }
}

export function getUserId(): string {
  return readStored(USER_ID_KEY);
}

export function getConversationId(): string {
  return readStored(CONVERSATION_ID_KEY);
}

/** Forget the current thread so the next chat message starts a new conversation. */
export function resetConversation(): void {
  try {
    window.localStorage.removeItem(CONVERSATION_ID_KEY);
  } catch {
    // no-op
  }
}

function withIdentityHeaders(init: RequestInit): RequestInit {
  const headers = new Headers(init.headers);
  const userId = getUserId();
  if (userId) {
    headers.set(USER_ID_HEADER, userId);
  }
  return { ...init, headers };
}

function captureEchoedIds(res: Response): void {
  const userId = res.headers.get(USER_ID_HEADER);
  if (userId) {
    writeStored(USER_ID_KEY, userId);
  }
  const conversationId = res.headers.get(CONVERSATION_ID_HEADER);
  if (conversationId) {
    writeStored(CONVERSATION_ID_KEY, conversationId);
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let domain = "internal";
  let code = "INTERNAL";
  let context: Record<string, unknown> = {};
  try {
    const body = (await res.json()) as Partial<ErrorEnvelope>;
    if (body?.error) {
      domain = body.error.domain ?? domain;
      code = body.error.code ?? code;
      context = body.error.context ?? {};
    }
  } catch {
    // Non-JSON error body (e.g. a proxy 502): fall back to the generic code.
  }
  return new ApiError(res.status, domain, code, context);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, withIdentityHeaders(init));
  } catch (cause) {
    // Network/connection failure: no envelope to parse.
    throw new ApiError(0, "network", "UNREACHABLE", {
      message: cause instanceof Error ? cause.message : String(cause),
    });
  }

  captureEchoedIds(res);

  if (!res.ok) {
    throw await toApiError(res);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export const api = {
  listDocuments(): Promise<DocumentListResponse> {
    return request<DocumentListResponse>("/documents");
  },

  uploadDocuments(files: File[]): Promise<IngestResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append("files", file, file.name);
    }
    return request<IngestResponse>("/ingest", { method: "POST", body: form });
  },

  deleteDocument(source: string): Promise<DeleteDocumentResponse> {
    return request<DeleteDocumentResponse>(
      `/documents/${encodeURIComponent(source)}`,
      { method: "DELETE" },
    );
  },

  chat(query: string): Promise<ChatResponse> {
    const headers = new Headers({ "Content-Type": "application/json" });
    const conversationId = getConversationId();
    if (conversationId) {
      headers.set(CONVERSATION_ID_HEADER, conversationId);
    }
    return request<ChatResponse>("/chat", {
      method: "POST",
      headers,
      body: JSON.stringify({ query }),
    });
  },
};
