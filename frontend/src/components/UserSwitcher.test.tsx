import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "../test/utils";
import { UserSwitcher } from "./UserSwitcher";

const USER_A = "aaaaaaaa-0000-0000-0000-000000000000";
const USER_B = "bbbbbbbb-1111-1111-1111-111111111111";

beforeEach(() => {
  window.localStorage.clear();
  window.localStorage.setItem("rag.userId", USER_A);
  window.localStorage.setItem("rag.userRoster", JSON.stringify([USER_A, USER_B]));
});

describe("UserSwitcher", () => {
  it("labels the active user and lists the roster with the active one checked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <UserSwitcher activeUserId={USER_A} onSwitch={vi.fn()} onAddUser={vi.fn()} />,
    );

    // Trigger shows the active user's label + short id.
    expect(screen.getByRole("button", { name: "Switch user" })).toHaveTextContent(
      "User 1",
    );

    await user.click(screen.getByRole("button", { name: "Switch user" }));

    const items = screen.getAllByRole("menuitem");
    expect(items.map((el) => el.textContent)).toEqual([
      expect.stringContaining("User 1"),
      expect.stringContaining("User 2"),
      "New user",
    ]);
    // The active row is highlighted (MUI marks it with the Mui-selected class).
    expect(items[0]).toHaveClass("Mui-selected");
    expect(items[1]).not.toHaveClass("Mui-selected");
  });

  it("fires onSwitch with the chosen id when another user is picked", async () => {
    const user = userEvent.setup();
    const onSwitch = vi.fn();
    renderWithProviders(
      <UserSwitcher activeUserId={USER_A} onSwitch={onSwitch} onAddUser={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Switch user" }));
    await user.click(screen.getByRole("menuitem", { name: /User 2/ }));

    expect(onSwitch).toHaveBeenCalledWith(USER_B);
  });

  it("does not fire onSwitch when the already-active user is picked", async () => {
    const user = userEvent.setup();
    const onSwitch = vi.fn();
    renderWithProviders(
      <UserSwitcher activeUserId={USER_A} onSwitch={onSwitch} onAddUser={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Switch user" }));
    await user.click(screen.getByRole("menuitem", { name: /User 1/ }));

    expect(onSwitch).not.toHaveBeenCalled();
  });

  it("fires onAddUser when 'New user' is chosen", async () => {
    const user = userEvent.setup();
    const onAddUser = vi.fn();
    renderWithProviders(
      <UserSwitcher activeUserId={USER_A} onSwitch={vi.fn()} onAddUser={onAddUser} />,
    );

    await user.click(screen.getByRole("button", { name: "Switch user" }));
    await user.click(screen.getByRole("menuitem", { name: "New user" }));

    expect(onAddUser).toHaveBeenCalledTimes(1);
  });
});
