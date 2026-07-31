import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";

/**
 * Shared, reusable styled primitives used across multiple components.
 * Component-specific styling lives in each component's co-located
 * `<Component>.styled.ts` file instead.
 */

/** Outlined surface laid out as a vertical flex column (the app's panes). */
export const PanelPaper = styled(Paper)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  padding: theme.spacing(2),
}));

/** Scrollable region that fills remaining flex space without overflowing. */
export const ScrollArea = styled(Box)({
  flex: 1,
  minHeight: 0,
  overflowY: "auto",
});

/** Centered, muted placeholder shown when a list or panel has no content. */
export const EmptyState = styled(Box)(({ theme }) => ({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  textAlign: "center",
  color: theme.palette.text.secondary,
}));
