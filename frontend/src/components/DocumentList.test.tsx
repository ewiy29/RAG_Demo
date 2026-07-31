import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "../test/utils";
import type {
  DeleteDocumentResponse,
  DocumentListResponse,
} from "../api/types";
import { DocumentList } from "./DocumentList";

// Mock only the network boundary; keep ApiError (and everything else) real so
// the real hooks run against a real QueryClient. This exercises the
// invalidate/refetch wiring for genuine integration coverage.
vi.mock("../api/client", async (importActual) => {
  const actual = await importActual<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      listDocuments: vi.fn(),
      deleteDocument: vi.fn(),
      uploadDocuments: vi.fn(),
    },
  };
});

import { api } from "../api/client";

const listDocuments = vi.mocked(api.listDocuments);
const deleteDocument = vi.mocked(api.deleteDocument);

function listResponse(
  documents: DocumentListResponse["documents"],
): DocumentListResponse {
  return { user_id: "u1", documents };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DocumentList", () => {
  it("shows the empty state when the user has no documents", async () => {
    listDocuments.mockResolvedValue(listResponse([]));

    renderWithProviders(<DocumentList />);

    expect(
      await screen.findByText(/No documents yet/),
    ).toBeInTheDocument();
  });

  it("renders a row per ingested document", async () => {
    listDocuments.mockResolvedValue(
      listResponse([{ source: "a.md", chunks: 2 }]),
    );

    renderWithProviders(<DocumentList />);

    expect(await screen.findByText("a.md")).toBeInTheDocument();
    expect(screen.getByText("2 chunks")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Delete a.md" }),
    ).toBeInTheDocument();
  });

  it("deletes via the confirm dialog and refetches the list", async () => {
    const user = userEvent.setup();
    listDocuments.mockResolvedValue(
      listResponse([{ source: "a.md", chunks: 1 }]),
    );
    deleteDocument.mockResolvedValue({
      user_id: "u1",
      source: "a.md",
      status: "deleted",
    } satisfies DeleteDocumentResponse);

    renderWithProviders(<DocumentList />);
    await screen.findByText("a.md");
    expect(listDocuments).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Delete a.md" }));
    expect(await screen.findByText("Delete document?")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^Delete$/ }));

    await waitFor(() =>
      expect(deleteDocument).toHaveBeenCalledWith(
        "a.md",
        expect.objectContaining({ userId: expect.any(String) }),
      ),
    );
    // onSuccess invalidates the documents query -> a refetch fires.
    await waitFor(() => expect(listDocuments).toHaveBeenCalledTimes(2));
  });
});
