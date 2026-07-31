import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CircularProgress from "@mui/material/CircularProgress";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { alpha, useTheme } from "@mui/material/styles";

const ACCEPT = {
  "text/markdown": [".md"],
  "text/plain": [".txt"],
  "application/pdf": [".pdf"],
};

interface FileDropzoneProps {
  onFiles: (files: File[]) => void;
  busy?: boolean;
  compact?: boolean;
}

/** Accessible drag-and-drop upload area restricted to supported file types. */
export function FileDropzone({ onFiles, busy = false, compact = false }: FileDropzoneProps) {
  const theme = useTheme();

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        onFiles(accepted);
      }
    },
    [onFiles],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    disabled: busy,
  });

  return (
    <Box
      {...getRootProps()}
      role="button"
      aria-label="Upload documents by dragging files here or clicking to browse"
      aria-disabled={busy}
      sx={{
        border: "2px dashed",
        borderColor: isDragActive ? "primary.main" : "divider",
        borderRadius: 2,
        p: compact ? 2 : 4,
        textAlign: "center",
        cursor: busy ? "default" : "pointer",
        transition: "border-color 120ms, background-color 120ms",
        bgcolor: isDragActive
          ? alpha(theme.palette.primary.main, 0.06)
          : "transparent",
        outline: "none",
        "&:hover": { borderColor: busy ? "divider" : "primary.light" },
        "&:focus-visible": { borderColor: "primary.main" },
      }}
    >
      <input {...getInputProps()} />
      {busy ? (
        <CircularProgress size={compact ? 22 : 28} aria-label="Uploading" />
      ) : (
        <CloudUploadIcon
          color={isDragActive ? "primary" : "action"}
          sx={{ fontSize: compact ? 28 : 40 }}
        />
      )}
      <Typography
        variant={compact ? "body2" : "body1"}
        sx={{ mt: 1, fontWeight: 500 }}
      >
        {busy
          ? "Uploading…"
          : isDragActive
            ? "Drop the files to upload"
            : "Drag & drop files, or click to browse"}
      </Typography>
      {!compact && (
        <Typography variant="caption" color="text.secondary">
          Supported: .md, .txt, .pdf
        </Typography>
      )}
    </Box>
  );
}
