import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// jsdom doesn't implement Element.scrollTo; stub it so components that
// auto-scroll on mount/update (e.g. the chat transcript) don't throw.
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = vi.fn();
}

// Unmount React trees and clear jsdom between tests so state never leaks across
// test cases.
afterEach(() => {
  cleanup();
});
