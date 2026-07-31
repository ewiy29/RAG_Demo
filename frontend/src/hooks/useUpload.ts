import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { IngestResponse } from "../api/types";
import { documentsKey } from "./useDocuments";

/** Upload one or more files, then refresh the document list. */
export function useUpload() {
  const queryClient = useQueryClient();
  return useMutation<IngestResponse, Error, File[]>({
    mutationFn: (files: File[]) => api.uploadDocuments(files),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentsKey });
    },
  });
}
