import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { DeleteDocumentResponse } from "../api/types";
import { useUser } from "../context/UserContext";
import { documentsKey } from "./useDocuments";

/** Delete a single ingested source, then refresh the document list. */
export function useDeleteDocument() {
  const queryClient = useQueryClient();
  const { userId } = useUser();
  return useMutation<DeleteDocumentResponse, Error, string>({
    mutationFn: (source: string) => api.deleteDocument(source, { userId }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentsKey });
    },
  });
}
