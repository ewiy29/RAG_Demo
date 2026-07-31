import { styled } from "@mui/material/styles";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";

import { ScrollArea } from "../styles/shared";

/** Header row holding the "Chat" title and the reset action. */
export const HeaderRow = styled(Stack)(({ theme }) => ({
  alignItems: "center",
  justifyContent: "space-between",
  marginBottom: theme.spacing(1),
}));

/** Scrollable transcript area with a subtle inset background. */
export const Transcript = styled(ScrollArea)(({ theme }) => ({
  backgroundColor: theme.palette.action.hover,
  borderRadius: Number(theme.shape.borderRadius) * 2,
  padding: theme.spacing(2),
}));

/** Bottom composer row: the message field and send button. */
export const Composer = styled(Stack)(({ theme }) => ({
  alignItems: "flex-end",
  marginTop: theme.spacing(1.5),
}));

/** Send button sized to align with the single-line text field. */
export const SendButton = styled(Button)({
  height: 40,
});
