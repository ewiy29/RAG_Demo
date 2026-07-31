import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutlined";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import SendIcon from "@mui/icons-material/Send";

import { resetConversation } from "../api/client";
import { useChat } from "../hooks/useChat";
import { errorMessage } from "../lib/errorMessage";
import { MessageBubble, type ChatMessage } from "./MessageBubble";

let nextId = 0;
const makeId = () => `m${nextId++}`;

/** Right pane: the multi-turn chat over the user's uploaded documents. */
export function ChatPanel() {
  const chat = useChat();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const send = () => {
    const query = draft.trim();
    if (!query || chat.isPending) {
      return;
    }
    const userMsg: ChatMessage = { id: makeId(), role: "user", content: query };
    const pendingId = makeId();
    const pendingMsg: ChatMessage = {
      id: pendingId,
      role: "assistant",
      content: "",
      pending: true,
    };
    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setDraft("");

    chat.mutate(query, {
      onSuccess: (res) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === pendingId
              ? {
                  ...m,
                  pending: false,
                  content: res.answer,
                  grounded: res.grounded,
                  citations: res.citations,
                }
              : m,
          ),
        );
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === pendingId
              ? {
                  ...m,
                  pending: false,
                  error: true,
                  content: errorMessage(err),
                }
              : m,
          ),
        );
      },
    });
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const startNewConversation = () => {
    resetConversation();
    setMessages([]);
    chat.reset();
  };

  return (
    <Stack sx={{ height: "100%" }}>
      <Stack
        direction="row"
        sx={{ mb: 1, alignItems: "center", justifyContent: "space-between" }}
      >
        <Typography variant="h6">Chat</Typography>
        <Tooltip title="Start a new conversation">
          <span>
            <IconButton
              size="small"
              onClick={startNewConversation}
              disabled={messages.length === 0 || chat.isPending}
              aria-label="Start a new conversation"
            >
              <RestartAltIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>

      <Box
        ref={scrollRef}
        sx={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          borderRadius: 2,
          bgcolor: "action.hover",
          p: 2,
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              color: "text.secondary",
              textAlign: "center",
            }}
          >
            <ChatBubbleOutlineIcon sx={{ fontSize: 40, opacity: 0.5 }} />
            <Typography variant="body2" sx={{ mt: 1, maxWidth: 320 }}>
              Ask a question about your uploaded documents. Answers are grounded
              in your files and cite their sources.
            </Typography>
          </Box>
        ) : (
          <Stack spacing={1.5}>
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </Stack>
        )}
      </Box>

      <Stack direction="row" spacing={1} sx={{ mt: 1.5, alignItems: "flex-end" }}>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          size="small"
          placeholder="Ask about your documents…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={chat.isPending}
          aria-label="Chat message"
        />
        <Button
          variant="contained"
          endIcon={<SendIcon />}
          onClick={send}
          disabled={chat.isPending || draft.trim().length === 0}
          sx={{ height: 40 }}
        >
          Send
        </Button>
      </Stack>
    </Stack>
  );
}
