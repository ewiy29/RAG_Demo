import { styled } from "@mui/material/styles";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";

interface BubbleProps {
  $isUser: boolean;
  $isError?: boolean;
}

/** Row that aligns a turn to the right (user) or left (assistant). */
export const BubbleRow = styled(Stack, {
  shouldForwardProp: (prop) => prop !== "$isUser",
})<{ $isUser: boolean }>(({ $isUser }) => ({
  width: "100%",
  justifyContent: $isUser ? "flex-end" : "flex-start",
}));

/** The message surface, colored by role/error and with an asymmetric corner. */
export const Bubble = styled(Paper, {
  shouldForwardProp: (prop) => prop !== "$isUser" && prop !== "$isError",
})<BubbleProps>(({ theme, $isUser, $isError }) => ({
  maxWidth: "85%",
  paddingInline: theme.spacing(2),
  paddingBlock: theme.spacing(1.25),
  backgroundColor: $isUser
    ? theme.palette.primary.main
    : $isError
      ? theme.palette.error.main
      : theme.palette.background.paper,
  color:
    $isUser || $isError
      ? theme.palette.primary.contrastText
      : theme.palette.text.primary,
  ...($isUser
    ? { borderTopRightRadius: 4 }
    : { borderTopLeftRadius: 4 }),
}));
