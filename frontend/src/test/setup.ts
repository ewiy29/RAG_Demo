import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Unmount React trees and clear jsdom between tests so state never leaks across
// test cases.
afterEach(() => {
  cleanup();
});
