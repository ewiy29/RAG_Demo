import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "./test/utils";
import type { DocumentListResponse } from "./api/types";

// Mock only the network boundary; keep the identity helpers (ensureUserId,
// addUser, switchUser, getUserId, getUserRoster) real so the switch wiring is
// genuinely exercised.
vi.mock("./api/client", async (importActual) => {
  const actual = await importActual<typeof import("./api/client")>();
  return {
    ...actual,
    api: {
      listDocuments: vi.fn(),
      deleteDocument: vi.fn(),
      uploadDocuments: vi.fn(),
      chat: vi.fn(),
    },
  };
});

import { api, getUserId } from "./api/client";
import App from "./App";

const listDocuments = vi.mocked(api.listDocuments);

const USER_A = "aaaaaaaa-0000-0000-0000-000000000000";

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  // Seed a known active user so we can switch back to it by label later.
  window.localStorage.setItem("rag.userId", USER_A);
  window.localStorage.setItem("rag.userRoster", JSON.stringify([USER_A]));

  // Documents are scoped by the active tenant id; User A owns "a.md",
  // every other (freshly minted) user owns nothing.
  listDocuments.mockImplementation(async (): Promise<DocumentListResponse> => {
    const id = getUserId();
    return {
      user_id: id,
      documents: id === USER_A ? [{ source: "a.md", chunks: 1 }] : [],
    };
  });
});

describe("App user switching", () => {
  it("re-fetches documents from the API when switching back to an existing user", async () => {
    const user = userEvent.setup();
    renderWithProviders(<App />);

    // User A starts out owning a.md.
    expect(await screen.findByText("a.md")).toBeInTheDocument();

    // Switch to a brand-new user: their corpus is empty (isolation).
    await user.click(screen.getByRole("button", { name: "Switch user" }));
    await user.click(screen.getByRole("menuitem", { name: "New user" }));

    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument();
    expect(screen.queryByText("a.md")).not.toBeInTheDocument();

    const callsBeforeSwitchBack = listDocuments.mock.calls.length;

    // Switch back to User A: this must hit the API again (not a stale cache)
    // and their previously-uploaded document must reappear.
    await user.click(screen.getByRole("button", { name: "Switch user" }));
    await user.click(screen.getByRole("menuitem", { name: /User 1/ }));

    expect(await screen.findByText("a.md")).toBeInTheDocument();
    await waitFor(() =>
      expect(listDocuments.mock.calls.length).toBeGreaterThan(
        callsBeforeSwitchBack,
      ),
    );
  });
});
