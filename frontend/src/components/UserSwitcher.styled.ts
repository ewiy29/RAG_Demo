import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";

/** Header trigger showing the active tenant; normal-case so the id reads cleanly. */
export const SwitcherButton = styled(Button)({
  textTransform: "none",
});

/** Inline label wrapper inside the trigger: name + short id on one baseline. */
export const SwitcherLabel = styled("span")(({ theme }) => ({
  display: "inline-flex",
  alignItems: "baseline",
  gap: theme.spacing(1),
}));

/** Muted, monospaced short-id caption shown beside the user label. */
export const ShortId = styled("span")(({ theme }) => ({
  fontFamily: "monospace",
  fontSize: theme.typography.caption.fontSize,
  color: theme.palette.text.secondary,
}));
