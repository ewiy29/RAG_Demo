import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";

import { renderWithProviders } from "../test/utils";
import type { Citation } from "../api/types";
import { Citations } from "./Citations";

function citation(overrides: Partial<Citation> = {}): Citation {
  return {
    marker: 1,
    source: "notes.md",
    chunk_index: 0,
    score: 0.42,
    quote: "the quoted text",
    ...overrides,
  };
}

describe("Citations", () => {
  it("renders nothing when there are no citations", () => {
    const { container } = renderWithProviders(<Citations citations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("uses a singular label for one citation", () => {
    renderWithProviders(<Citations citations={[citation()]} />);
    expect(screen.getByText("1 source")).toBeInTheDocument();
  });

  it("uses a plural label for multiple citations", () => {
    renderWithProviders(
      <Citations citations={[citation(), citation({ chunk_index: 1 })]} />,
    );
    expect(screen.getByText("2 sources")).toBeInTheDocument();
  });

  it("formats the score to three decimals", () => {
    renderWithProviders(<Citations citations={[citation({ score: 0.1 })]} />);
    expect(screen.getByText("score 0.100")).toBeInTheDocument();
  });
});
