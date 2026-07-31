import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import BlockIcon from "@mui/icons-material/Block";
import VerifiedIcon from "@mui/icons-material/Verified";

import type { Citation } from "../api/types";
import { Citations } from "./Citations";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  // Assistant-only metadata.
  grounded?: boolean;
  citations?: Citation[];
  pending?: boolean;
  error?: boolean;
}

interface MessageBubbleProps {
  message: ChatMessage;
}

/** A single chat turn. User turns are right-aligned; assistant turns show
 * grounded/refused status and any supporting citations. */
export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <Stack
      direction="row"
      sx={{ width: "100%", justifyContent: isUser ? "flex-end" : "flex-start" }}
    >
      <Paper
        variant={isUser ? "elevation" : "outlined"}
        elevation={isUser ? 2 : 0}
        sx={{
          maxWidth: "85%",
          px: 2,
          py: 1.25,
          bgcolor: isUser
            ? "primary.main"
            : message.error
              ? "error.main"
              : "background.paper",
          color: isUser || message.error ? "primary.contrastText" : "text.primary",
          borderTopRightRadius: isUser ? 4 : undefined,
          borderTopLeftRadius: isUser ? undefined : 4,
        }}
      >
        {!isUser && !message.error && !message.pending && (
          <Box sx={{ mb: 0.5 }}>
            {message.grounded ? (
              <Chip
                size="small"
                color="success"
                variant="outlined"
                icon={<VerifiedIcon />}
                label="Grounded"
              />
            ) : (
              <Chip
                size="small"
                color="warning"
                variant="outlined"
                icon={<BlockIcon />}
                label="No answer in your documents"
              />
            )}
          </Box>
        )}

        <Typography
          variant="body1"
          sx={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
        >
          {message.pending ? "Thinking…" : message.content}
        </Typography>

        {!isUser && message.citations && message.citations.length > 0 && (
          <Citations citations={message.citations} />
        )}
      </Paper>
    </Stack>
  );
}
