import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { DeleteDocumentResponse } from "../api/types";
import { documentsKey } from "./useDocuments";

/** Delete a single ingested source, then refresh the document list. */
export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation<DeleteDocumentResponse, Error, string>({
    mutationFn: (source: string) => api.deleteDocument(source),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentsKey });
    },
  });
}
