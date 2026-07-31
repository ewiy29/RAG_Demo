import { alpha, styled } from "@mui/material/styles";
import Box from "@mui/material/Box";

interface DropzoneProps {
  $active: boolean;
  $busy: boolean;
  $compact: boolean;
}

/** Drag-and-drop target whose border/background react to drag and busy state. */
export const Dropzone = styled(Box, {
  shouldForwardProp: (prop) =>
    prop !== "$active" && prop !== "$busy" && prop !== "$compact",
})<DropzoneProps>(({ theme, $active, $busy, $compact }) => ({
  border: "2px dashed",
  borderColor: $active ? theme.palette.primary.main : theme.palette.divider,
  borderRadius: Number(theme.shape.borderRadius) * 2,
  padding: theme.spacing($compact ? 2 : 4),
  textAlign: "center",
  cursor: $busy ? "default" : "pointer",
  transition: "border-color 120ms, background-color 120ms",
  backgroundColor: $active
    ? alpha(theme.palette.primary.main, 0.06)
    : "transparent",
  outline: "none",
  "&:hover": {
    borderColor: $busy ? theme.palette.divider : theme.palette.primary.light,
  },
  "&:focus-visible": { borderColor: theme.palette.primary.main },
}));
