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
const USER_ROSTER_KEY = "rag.userRoster";
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

// --- Demo tenancy roster --------------------------------------------------
// A "switch user" affordance for the demo: several tenant GUIDs remembered
// locally so uploading as one and switching to another shows per-user
// document isolation. The backend accepts any client-supplied X-User-Id
// (it's a tenant/correlation key, not auth), so we can mint ids client-side.

/** Generate a fresh tenant GUID, preferring the platform crypto UUID. */
function mintUserId(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {
    // fall through to the manual hex id below
  }
  return Array.from({ length: 32 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");
}

function readRoster(): string[] {
  try {
    const raw = window.localStorage.getItem(USER_ROSTER_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    if (Array.isArray(parsed)) {
      return parsed.filter((id): id is string => typeof id === "string" && !!id);
    }
  } catch {
    // Corrupt/blocked storage: fall back to an empty roster.
  }
  return [];
}

function writeRoster(ids: string[]): void {
  try {
    window.localStorage.setItem(USER_ROSTER_KEY, JSON.stringify(ids));
  } catch {
    // Ignore storage failures; the roster just won't persist this session.
  }
}

/**
 * Return the remembered roster of tenant ids, always including the active id.
 * Seeds a fresh id when there is no identity yet so callers get a non-empty
 * roster to render.
 */
export function getUserRoster(): string[] {
  const active = ensureUserId();
  const roster = readRoster();
  if (!roster.includes(active)) {
    roster.unshift(active);
    writeRoster(roster);
  }
  return roster;
}

/**
 * Return the active tenant id, minting and persisting one (and seeding the
 * roster) when the client has no identity yet. Makes identity deterministic
 * on first load instead of waiting for the server to echo a minted id.
 */
export function ensureUserId(): string {
  const existing = getUserId();
  if (existing) {
    return existing;
  }
  const minted = mintUserId();
  writeStored(USER_ID_KEY, minted);
  writeRoster([minted, ...readRoster().filter((id) => id !== minted)]);
  return minted;
}

/** Make an existing roster id the active tenant, starting a fresh thread. */
export function switchUser(id: string): void {
  writeStored(USER_ID_KEY, id);
  const roster = readRoster();
  if (!roster.includes(id)) {
    writeRoster([...roster, id]);
  }
  resetConversation();
}

/** Mint a brand-new tenant, add it to the roster, and make it active. */
export function addUser(): string {
  const id = mintUserId();
  writeRoster([...readRoster(), id]);
  writeStored(USER_ID_KEY, id);
  resetConversation();
  return id;
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
