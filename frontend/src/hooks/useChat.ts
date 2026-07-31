import { useMutation } from "@tanstack/react-query";

import { api } from "../api/client";
import type { ChatResponse } from "../api/types";
import { useUser } from "../context/UserContext";

/** Send a chat turn, threading the conversation id from context and adopting
 * the id the server echoes so the next turn continues the same thread. */
export function useChat() {
  const { userId, conversationId, setConversationId } = useUser();
  return useMutation<ChatResponse, Error, string>({
    mutationFn: async (query: string) => {
      const result = await api.chat(query, { userId, conversationId });
      if (result.conversationId) {
        setConversationId(result.conversationId);
      }
      return result.data;
    },
  });
}
