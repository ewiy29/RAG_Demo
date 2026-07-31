import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { IngestResponse } from "../api/types";
import { useUser } from "../context/UserContext";
import { documentsKey } from "./useDocuments";

/** Upload one or more files, then refresh the document list. */
export function useUpload() {
  const queryClient = useQueryClient();
  const { userId } = useUser();
  return useMutation<IngestResponse, Error, File[]>({
    mutationFn: (files: File[]) => api.uploadDocuments(files, { userId }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: documentsKey });
    },
  });
}
