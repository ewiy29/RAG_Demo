import { ApiError } from "../api/client";

// The UI owns wording: the backend only sends a machine-readable code. Map the
// codes we expect to friendly sentences, with a sensible generic fallback.
const MESSAGES: Record<string, string> = {
  UNREACHABLE: "Can't reach the server. Is the API running?",
  INTERNAL: "Something went wrong on the server. Please try again.",
  UNAVAILABLE: "The service is temporarily unavailable. Please try again.",
  WRITE_FAILED: "Saving failed on the server. Please try again.",
  QUERY_FAILED: "The search backend failed. Please try again.",
  RATE_LIMITED: "Rate limited by the model provider. Please wait a moment.",
  TIMEOUT: "The request timed out. Please try again.",
  AUTH: "The server could not authenticate with the model provider.",
  UNSUPPORTED_TYPE: "That file type isn't supported (use .md, .txt or .pdf).",
  TOO_LARGE: "That file is too large to upload.",
  DECODE_FAILED: "That file could not be read as text.",
  EMPTY_CONTENT: "That file appears to be empty.",
  EXTRACTION_FAILED: "Text could not be extracted from that file.",
};

/** Map a bare backend error code to a user-facing message. */
export function codeMessage(code: string): string {
  return MESSAGES[code] ?? `Request failed (${code}).`;
}

/** Turn any thrown error into a user-facing message. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return codeMessage(error.code);
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "An unexpected error occurred.";
}
