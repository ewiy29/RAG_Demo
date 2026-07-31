import type { ReactNode } from "react";
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { UserProvider, useUser } from "./UserContext";

const USER_A = "aaaaaaaa-0000-0000-0000-000000000000";
const USER_B = "bbbbbbbb-1111-1111-1111-111111111111";

function wrapper({ children }: { children: ReactNode }) {
  return <UserProvider>{children}</UserProvider>;
}

function renderUser() {
  return renderHook(() => useUser(), { wrapper });
}

beforeEach(() => {
  window.localStorage.clear();
});

describe("UserProvider identity", () => {
  it("mints and persists an id (seeding the roster) when storage is empty", () => {
    const { result } = renderUser();

    expect(result.current.userId).toBeTruthy();
    expect(result.current.roster).toContain(result.current.userId);
    expect(window.localStorage.getItem("rag.userId")).toBe(
      result.current.userId,
    );
  });

  it("hydrates the active id and roster from localStorage on mount", () => {
    window.localStorage.setItem("rag.userId", USER_A);
    window.localStorage.setItem(
      "rag.userRoster",
      JSON.stringify([USER_A, USER_B]),
    );

    const { result } = renderUser();

    expect(result.current.userId).toBe(USER_A);
    expect(result.current.roster).toEqual([USER_A, USER_B]);
  });

  it("addUser appends a new active id and persists the roster", () => {
    window.localStorage.setItem("rag.userId", USER_A);
    window.localStorage.setItem("rag.userRoster", JSON.stringify([USER_A]));
    const { result } = renderUser();

    let minted = "";
    act(() => {
      minted = result.current.addUser();
    });

    expect(minted).not.toBe(USER_A);
    expect(result.current.userId).toBe(minted);
    expect(result.current.roster).toEqual([USER_A, minted]);
    expect(
      JSON.parse(window.localStorage.getItem("rag.userRoster") ?? "[]"),
    ).toEqual([USER_A, minted]);
  });

  it("switchUser makes an existing id active and resets the conversation", () => {
    window.localStorage.setItem("rag.userId", USER_A);
    window.localStorage.setItem(
      "rag.userRoster",
      JSON.stringify([USER_A, USER_B]),
    );
    const { result } = renderUser();

    act(() => {
      result.current.setConversationId("conv-1");
    });
    expect(result.current.conversationId).toBe("conv-1");

    act(() => {
      result.current.switchUser(USER_B);
    });

    expect(result.current.userId).toBe(USER_B);
    expect(result.current.conversationId).toBe("");
    expect(window.localStorage.getItem("rag.userId")).toBe(USER_B);
  });

  it("never persists the conversation id to localStorage", () => {
    const { result } = renderUser();

    act(() => {
      result.current.setConversationId("conv-42");
    });

    expect(result.current.conversationId).toBe("conv-42");
    expect(window.localStorage.getItem("rag.conversationId")).toBeNull();
    // No stray key holds the conversation id either.
    const values = Object.values({ ...window.localStorage });
    expect(values).not.toContain("conv-42");
  });

  it("resetConversation clears the in-memory thread", () => {
    const { result } = renderUser();

    act(() => {
      result.current.setConversationId("conv-7");
    });
    act(() => {
      result.current.resetConversation();
    });

    expect(result.current.conversationId).toBe("");
  });
});
