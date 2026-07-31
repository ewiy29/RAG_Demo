import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import type { DocumentInfo } from "../api/types";

export const documentsKey = ["documents"] as const;

/** Load the current user's ingested sources (with chunk counts). */
export function useDocuments() {
  return useQuery<DocumentInfo[]>({
    queryKey: documentsKey,
    queryFn: async () => (await api.listDocuments()).documents,
  });
}
