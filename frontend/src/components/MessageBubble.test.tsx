import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";

import { renderWithProviders } from "../test/utils";
import type { ChatMessage } from "./MessageBubble";
import { MessageBubble } from "./MessageBubble";
import type { Citation } from "../api/types";

function message(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: "1",
    role: "assistant",
    content: "the answer",
    ...overrides,
  };
}

const GROUNDED = "Grounded";
const REFUSED = "No answer in your documents";

describe("MessageBubble", () => {
  it("shows the grounded chip for a grounded assistant answer", () => {
    renderWithProviders(<MessageBubble message={message({ grounded: true })} />);
    expect(screen.getByText(GROUNDED)).toBeInTheDocument();
    expect(screen.queryByText(REFUSED)).not.toBeInTheDocument();
  });

  it("shows the refusal chip when the answer is not grounded", () => {
    renderWithProviders(<MessageBubble message={message({ grounded: false })} />);
    expect(screen.getByText(REFUSED)).toBeInTheDocument();
    expect(screen.queryByText(GROUNDED)).not.toBeInTheDocument();
  });

  it("renders a placeholder and hides content while pending", () => {
    renderWithProviders(
      <MessageBubble message={message({ pending: true, content: "the answer" })} />,
    );
    expect(screen.getByText("Thinking…")).toBeInTheDocument();
    expect(screen.queryByText("the answer")).not.toBeInTheDocument();
  });

  it("suppresses the status chip on an error turn but still shows content", () => {
    renderWithProviders(
      <MessageBubble message={message({ error: true, content: "it broke" })} />,
    );
    expect(screen.getByText("it broke")).toBeInTheDocument();
    expect(screen.queryByText(GROUNDED)).not.toBeInTheDocument();
    expect(screen.queryByText(REFUSED)).not.toBeInTheDocument();
  });

  it("renders citations when the assistant turn has them", () => {
    const citations: Citation[] = [
      { marker: 1, source: "a.md", chunk_index: 0, score: 0.5, quote: "q" },
    ];
    renderWithProviders(
      <MessageBubble message={message({ grounded: true, citations })} />,
    );
    expect(screen.getByText("1 source")).toBeInTheDocument();
  });

  it("shows no status chip for a user turn", () => {
    renderWithProviders(
      <MessageBubble message={message({ role: "user", content: "my question" })} />,
    );
    expect(screen.getByText("my question")).toBeInTheDocument();
    expect(screen.queryByText(GROUNDED)).not.toBeInTheDocument();
    expect(screen.queryByText(REFUSED)).not.toBeInTheDocument();
  });
});
