// Thin, stateless fetch wrapper around the RAG API.
//
// Identity is NOT owned here: callers pass the active tenant id (and, for chat,
// the current conversation id) so this module never touches storage or global
// state. See `context/UserContext.tsx` for where identity lives.
//
// Responsibilities kept here (not in components/hooks):
//   * Attach the X-User-Id / X-Conversation-Id headers from the supplied
//     identity, and surface the conversation id the server echoes on /chat so
//     the caller can thread multi-turn history.
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

const USER_ID_HEADER = "X-User-Id";
const CONVERSATION_ID_HEADER = "X-Conversation-Id";

/** The identity a request is made on behalf of. */
export interface RequestIdentity {
  userId?: string;
  conversationId?: string;
}

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

function withUserId(init: RequestInit, userId?: string): RequestInit {
  const headers = new Headers(init.headers);
  if (userId) {
    headers.set(USER_ID_HEADER, userId);
  }
  return { ...init, headers };
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

async function request<T>(
  path: string,
  init: RequestInit,
): Promise<{ data: T; res: Response }> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, init);
  } catch (cause) {
    // Network/connection failure: no envelope to parse.
    throw new ApiError(0, "network", "UNREACHABLE", {
      message: cause instanceof Error ? cause.message : String(cause),
    });
  }

  if (!res.ok) {
    throw await toApiError(res);
  }
  if (res.status === 204) {
    return { data: undefined as T, res };
  }
  return { data: (await res.json()) as T, res };
}

/** The parsed chat body plus the conversation id the server echoed for it. */
export interface ChatResult {
  data: ChatResponse;
  conversationId: string | null;
}

export const api = {
  async listDocuments(
    identity: RequestIdentity = {},
  ): Promise<DocumentListResponse> {
    const { data } = await request<DocumentListResponse>(
      "/documents",
      withUserId({}, identity.userId),
    );
    return data;
  },

  async uploadDocuments(
    files: File[],
    identity: RequestIdentity = {},
  ): Promise<IngestResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append("files", file, file.name);
    }
    const { data } = await request<IngestResponse>(
      "/ingest",
      withUserId({ method: "POST", body: form }, identity.userId),
    );
    return data;
  },

  async deleteDocument(
    source: string,
    identity: RequestIdentity = {},
  ): Promise<DeleteDocumentResponse> {
    const { data } = await request<DeleteDocumentResponse>(
      `/documents/${encodeURIComponent(source)}`,
      withUserId({ method: "DELETE" }, identity.userId),
    );
    return data;
  },

  async chat(query: string, identity: RequestIdentity = {}): Promise<ChatResult> {
    const headers = new Headers({ "Content-Type": "application/json" });
    if (identity.conversationId) {
      headers.set(CONVERSATION_ID_HEADER, identity.conversationId);
    }
    const { data, res } = await request<ChatResponse>(
      "/chat",
      withUserId(
        { method: "POST", headers, body: JSON.stringify({ query }) },
        identity.userId,
      ),
    );
    return { data, conversationId: res.headers.get(CONVERSATION_ID_HEADER) };
  },
};
