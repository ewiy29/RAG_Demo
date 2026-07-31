import type { ReactNode } from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

interface AppLayoutProps {
  documents: ReactNode;
  chat: ReactNode;
}

/** Responsive two-pane shell: documents on the left, chat on the right. */
export function AppLayout({ documents, chat }: AppLayoutProps) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: "1px solid", borderColor: "divider" }}>
        <Toolbar variant="dense">
          <AutoAwesomeIcon color="primary" sx={{ mr: 1 }} />
          <Typography variant="h6" component="h1" sx={{ flexGrow: 1 }}>
            RAG Demo
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Grounded answers with citations
          </Typography>
        </Toolbar>
      </AppBar>

      <Container
        maxWidth="lg"
        sx={{ flex: 1, minHeight: 0, py: 2, display: "flex" }}
      >
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          sx={{ flex: 1, minHeight: 0, width: "100%" }}
        >
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              width: { xs: "100%", md: 360 },
              flexShrink: 0,
              minHeight: { xs: 280, md: 0 },
              display: "flex",
              flexDirection: "column",
            }}
          >
            {documents}
          </Paper>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              flex: 1,
              minHeight: { xs: 400, md: 0 },
              display: "flex",
              flexDirection: "column",
            }}
          >
            {chat}
          </Paper>
        </Stack>
      </Container>
    </Box>
  );
}
