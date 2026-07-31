import { describe, expect, it } from "vitest";

import { ApiError } from "../api/client";
import { codeMessage, errorMessage } from "./errorMessage";

describe("codeMessage", () => {
  it("maps a known code to its friendly message", () => {
    expect(codeMessage("TOO_LARGE")).toBe("That file is too large to upload.");
  });

  it("falls back to a generic message that echoes the unknown code", () => {
    expect(codeMessage("NOPE")).toBe("Request failed (NOPE).");
  });
});

describe("errorMessage", () => {
  it("maps an ApiError via its code", () => {
    const err = new ApiError(0, "network", "UNREACHABLE");
    expect(errorMessage(err)).toBe("Can't reach the server. Is the API running?");
  });

  it("uses the message of a plain Error", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });

  it("returns the default sentence for non-Error values", () => {
    expect(errorMessage("nope")).toBe("An unexpected error occurred.");
  });
});
