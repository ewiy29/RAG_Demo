// Single source of truth for the client's identity.
//
// Design:
//   * `userId` + `roster` are the demo's tenant identity. They persist across
//     reloads, but localStorage is touched ONLY here: read once on mount to
//     hydrate ("saturate") this context, and written back whenever they change.
//     Nothing else in the app reads or writes storage.
//   * `conversationId` is deliberately in-memory only. It is a transient thread
//     handle, has no business in localStorage, and resets on reload / user swap.

import type { ReactNode } from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

const USER_ID_KEY = "rag.userId";
const USER_ROSTER_KEY = "rag.userRoster";

/** Generate a fresh tenant GUID, preferring the platform crypto UUID. */
function mintUserId(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {
    // fall through to the manual hex id below
  }
  return Array.from({ length: 32 }, () =>
    Math.floor(Math.random() * 16).toString(16),
  ).join("");
}

function readStored(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function readStoredRoster(): string[] {
  try {
    const raw = window.localStorage.getItem(USER_ROSTER_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : [];
    if (Array.isArray(parsed)) {
      return parsed.filter((id): id is string => typeof id === "string" && !!id);
    }
  } catch {
    // Corrupt/blocked storage: fall back to an empty roster.
  }
  return [];
}

export interface Identity {
  userId: string;
  conversationId: string;
}

export interface UserContextValue {
  userId: string;
  roster: string[];
  conversationId: string;
  /** Make an existing roster id active, starting a fresh thread. */
  switchUser: (id: string) => void;
  /** Mint a brand-new tenant, add it to the roster, and make it active. */
  addUser: () => string;
  /** Forget the current thread so the next message starts a new conversation. */
  resetConversation: () => void;
  /** Adopt the conversation id echoed by /chat (in memory only). */
  setConversationId: (id: string) => void;
}

const UserContext = createContext<UserContextValue | null>(null);

/** Read (or seed) the active id + roster from storage exactly once, ensuring
 * the active id is always present in the roster. */
function readInitialIdentity(): { userId: string; roster: string[] } {
  const userId = readStored(USER_ID_KEY) || mintUserId();
  const roster = readStoredRoster();
  if (!roster.includes(userId)) {
    roster.unshift(userId);
  }
  return { userId, roster };
}

export function UserProvider({ children }: { children: ReactNode }) {
  // Runs once: this is the only place storage is read to hydrate context.
  const [initial] = useState(readInitialIdentity);
  const [userId, setUserId] = useState<string>(initial.userId);
  const [roster, setRoster] = useState<string[]>(initial.roster);
  const [conversationId, setConversationIdState] = useState<string>("");

  // The single writer: persist tenant identity whenever it changes.
  useEffect(() => {
    try {
      window.localStorage.setItem(USER_ID_KEY, userId);
      window.localStorage.setItem(USER_ROSTER_KEY, JSON.stringify(roster));
    } catch {
      // Ignore storage failures (private mode, disabled storage): the app still
      // works for this session, identity just won't persist.
    }
  }, [userId, roster]);

  const switchUser = useCallback((id: string) => {
    setUserId(id);
    setRoster((prev) => (prev.includes(id) ? prev : [...prev, id]));
    setConversationIdState("");
  }, []);

  const addUser = useCallback(() => {
    const id = mintUserId();
    setRoster((prev) => [...prev, id]);
    setUserId(id);
    setConversationIdState("");
    return id;
  }, []);

  const resetConversation = useCallback(() => {
    setConversationIdState("");
  }, []);

  const setConversationId = useCallback((id: string) => {
    if (id) {
      setConversationIdState(id);
    }
  }, []);

  const value = useMemo<UserContextValue>(
    () => ({
      userId,
      roster,
      conversationId,
      switchUser,
      addUser,
      resetConversation,
      setConversationId,
    }),
    [
      userId,
      roster,
      conversationId,
      switchUser,
      addUser,
      resetConversation,
      setConversationId,
    ],
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

/** Access the current identity + identity actions. Throws outside a provider. */
// eslint-disable-next-line react-refresh/only-export-components -- co-locating the hook with its provider is the conventional context pattern
export function useUser(): UserContextValue {
  const ctx = useContext(UserContext);
  if (!ctx) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return ctx;
}
