import { useMutation } from "@tanstack/react-query";

import { api } from "../api/client";
import type { ChatResponse } from "../api/types";

/** Send a chat turn; the client threads the conversation id automatically. */
export function useChat() {
  return useMutation<ChatResponse, Error, string>({
    mutationFn: (query: string) => api.chat(query),
  });
}
