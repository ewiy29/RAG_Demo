import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./client";

// Minimal stand-in for the parts of Response the client touches: headers.get,
// ok/status, and json(). jsonThrows simulates a non-JSON error body.
function makeResponse(options: {
  status?: number;
  ok?: boolean;
  headers?: Record<string, string>;
  json?: unknown;
  jsonThrows?: boolean;
}): Response {
  const {
    status = 200,
    ok = status >= 200 && status < 300,
    headers = {},
    json,
    jsonThrows = false,
  } = options;
  const lower = new Map(
    Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]),
  );
  return {
    ok,
    status,
    headers: { get: (name: string) => lower.get(name.toLowerCase()) ?? null },
    json: async () => {
      if (jsonThrows) {
        throw new SyntaxError("not json");
      }
      return json;
    },
  } as unknown as Response;
}

const fetchMock = vi.fn();

beforeEach(() => {
  window.localStorage.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Read the RequestInit headers passed to fetch on a given call. */
function headersOf(callIndex = 0): Headers {
  const init = fetchMock.mock.calls[callIndex][1] as RequestInit;
  return new Headers(init.headers);
}

describe("identity headers", () => {
  it("attaches the stored X-User-Id on requests", async () => {
    window.localStorage.setItem("rag.userId", "user-123");
    fetchMock.mockResolvedValue(makeResponse({ json: { documents: [] } }));

    await api.listDocuments();

    expect(headersOf().get("X-User-Id")).toBe("user-123");
  });

  it("omits X-User-Id when nothing is stored", async () => {
    fetchMock.mockResolvedValue(makeResponse({ json: { documents: [] } }));

    await api.listDocuments();

    expect(headersOf().get("X-User-Id")).toBeNull();
  });

  it("captures echoed user and conversation ids into storage", async () => {
    fetchMock.mockResolvedValue(
      makeResponse({
        headers: {
          "X-User-Id": "server-user",
          "X-Conversation-Id": "conv-9",
        },
        json: { answer: "hi" },
      }),
    );

    await api.chat("hello");

    expect(window.localStorage.getItem("rag.userId")).toBe("server-user");
    expect(window.localStorage.getItem("rag.conversationId")).toBe("conv-9");
  });
});

describe("chat", () => {
  it("sends the stored conversation id, JSON content type and body", async () => {
    window.localStorage.setItem("rag.conversationId", "conv-42");
    fetchMock.mockResolvedValue(makeResponse({ json: { answer: "ok" } }));

    await api.chat("what is up");

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/chat");
    expect(init.method).toBe("POST");
    const headers = new Headers(init.headers);
    expect(headers.get("X-Conversation-Id")).toBe("conv-42");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ query: "what is up" }));
  });
});

describe("error handling", () => {
  it("throws a typed ApiError carrying the envelope domain/code/context", async () => {
    fetchMock.mockResolvedValue(
      makeResponse({
        status: 413,
        json: {
          error: { domain: "ingest", code: "TOO_LARGE", context: { max: 10 } },
        },
      }),
    );

    const err = await api.uploadDocuments([]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(413);
    expect(err.domain).toBe("ingest");
    expect(err.code).toBe("TOO_LARGE");
    expect(err.context).toEqual({ max: 10 });
  });

  it("falls back to internal:INTERNAL for a non-JSON error body", async () => {
    fetchMock.mockResolvedValue(makeResponse({ status: 502, jsonThrows: true }));

    const err = await api.listDocuments().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.domain).toBe("internal");
    expect(err.code).toBe("INTERNAL");
  });

  it("wraps a network failure as network:UNREACHABLE with status 0", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    const err = await api.listDocuments().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.domain).toBe("network");
    expect(err.code).toBe("UNREACHABLE");
  });
});

describe("responses", () => {
  it("resolves 204 to undefined without parsing a body", async () => {
    fetchMock.mockResolvedValue(makeResponse({ status: 204 }));

    await expect(api.deleteDocument("a.md")).resolves.toBeUndefined();
  });

  it("URL-encodes the document source in the delete path", async () => {
    fetchMock.mockResolvedValue(makeResponse({ status: 204 }));

    await api.deleteDocument("my notes/åäö.md");

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/documents/my%20notes%2F%C3%A5%C3%A4%C3%B6.md",
    );
  });
});
