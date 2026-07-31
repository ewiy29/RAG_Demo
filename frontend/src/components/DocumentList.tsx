import { useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import DescriptionOutlinedIcon from "@mui/icons-material/DescriptionOutlined";
import FindReplaceIcon from "@mui/icons-material/FindReplace";

import { useDocuments } from "../hooks/useDocuments";
import { useUpload } from "../hooks/useUpload";
import { useDeleteDocument } from "../hooks/useDeleteDocument";
import type { FileError } from "../api/types";
import { codeMessage, errorMessage } from "../lib/errorMessage";
import { FileDropzone } from "./FileDropzone";

/** Left pane: upload area + the list of the user's ingested documents. */
export function DocumentList() {
  const documents = useDocuments();
  const upload = useUpload();
  const remove = useDeleteDocument();

  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [failures, setFailures] = useState<FileError[]>([]);
  // Which source a "replace" file-pick is targeting (for a per-row picker).
  const replaceTarget = useRef<string | null>(null);
  const replaceInputRef = useRef<HTMLInputElement | null>(null);

  const handleUpload = (files: File[]) => {
    upload.mutate(files, {
      onSuccess: (res) => setFailures(res.failures),
    });
  };

  const openReplacePicker = (source: string) => {
    replaceTarget.current = source;
    replaceInputRef.current?.click();
  };

  const handleReplacePicked = (fileList: FileList | null) => {
    const file = fileList?.[0];
    replaceTarget.current = null;
    if (replaceInputRef.current) {
      replaceInputRef.current.value = "";
    }
    if (file) {
      // Re-ingesting a same-named file overwrites it server-side
      // (delete-before-readd); a differently-named file is added alongside.
      handleUpload([file]);
    }
  };

  const confirmDelete = () => {
    if (pendingDelete) {
      remove.mutate(pendingDelete);
      setPendingDelete(null);
    }
  };

  const docs = documents.data ?? [];

  return (
    <Stack spacing={2} sx={{ height: "100%" }}>
      <Typography variant="h6">Documents</Typography>

      <FileDropzone onFiles={handleUpload} busy={upload.isPending} />

      <input
        ref={replaceInputRef}
        type="file"
        accept=".md,.txt,.pdf"
        hidden
        onChange={(e) => handleReplacePicked(e.target.files)}
      />

      {upload.isError && (
        <Alert severity="error" onClose={() => upload.reset()}>
          {errorMessage(upload.error)}
        </Alert>
      )}

      {failures.length > 0 && (
        <Alert severity="warning" onClose={() => setFailures([])}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Some files were not ingested:
          </Typography>
          <Box component="ul" sx={{ m: 0, pl: 2 }}>
            {failures.map((f) => (
              <li key={`${f.source}:${f.code}`}>
                <code>{f.source || "(unnamed)"}</code> — {codeMessage(f.code)}{" "}
                <Typography component="span" variant="caption" color="text.secondary">
                  ({f.code})
                </Typography>
              </li>
            ))}
          </Box>
        </Alert>
      )}

      <Box sx={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        {documents.isLoading ? (
          <Stack spacing={1} sx={{ mt: 1 }}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} variant="rounded" height={56} />
            ))}
          </Stack>
        ) : documents.isError ? (
          <Alert severity="error" action={
            <Button color="inherit" size="small" onClick={() => documents.refetch()}>
              Retry
            </Button>
          }>
            {errorMessage(documents.error)}
          </Alert>
        ) : docs.length === 0 ? (
          <Box
            sx={{
              mt: 2,
              textAlign: "center",
              color: "text.secondary",
              py: 4,
            }}
          >
            <DescriptionOutlinedIcon sx={{ fontSize: 36, opacity: 0.5 }} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              No documents yet — drop files above to start.
            </Typography>
          </Box>
        ) : (
          <List dense disablePadding>
            {docs.map((doc) => {
              const isDeleting =
                remove.isPending && remove.variables === doc.source;
              return (
                <ListItem
                  key={doc.source}
                  divider
                  sx={{ opacity: isDeleting ? 0.5 : 1 }}
                  secondaryAction={
                    <Stack direction="row" spacing={0.5}>
                      <Tooltip title="Replace this file">
                        <span>
                          <IconButton
                            edge="end"
                            aria-label={`Replace ${doc.source}`}
                            disabled={isDeleting || upload.isPending}
                            onClick={() => openReplacePicker(doc.source)}
                          >
                            <FindReplaceIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Delete this file">
                        <span>
                          <IconButton
                            edge="end"
                            aria-label={`Delete ${doc.source}`}
                            disabled={isDeleting}
                            onClick={() => setPendingDelete(doc.source)}
                          >
                            <DeleteOutlineIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  }
                >
                  <ListItemText
                    primary={doc.source}
                    secondary={
                      <Chip
                        label={`${doc.chunks} chunk${doc.chunks === 1 ? "" : "s"}`}
                        size="small"
                        variant="outlined"
                        sx={{ mt: 0.5 }}
                      />
                    }
                    slotProps={{
                      primary: { sx: { wordBreak: "break-word", pr: 8 } },
                      secondary: { component: "div" },
                    }}
                  />
                </ListItem>
              );
            })}
          </List>
        )}
      </Box>

      {remove.isError && (
        <Alert severity="error" onClose={() => remove.reset()}>
          {errorMessage(remove.error)}
        </Alert>
      )}

      <Dialog open={pendingDelete !== null} onClose={() => setPendingDelete(null)}>
        <DialogTitle>Delete document?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            <code>{pendingDelete}</code> will be removed from your corpus and can
            no longer be used to answer questions. You can re-upload it later.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPendingDelete(null)}>Cancel</Button>
          <Button color="error" onClick={confirmDelete}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
