import type { ReactNode } from "react";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

import { AppRoot, ContentPanel, HeaderBar, SidebarPanel } from "./AppLayout.styled";

interface AppLayoutProps {
  documents: ReactNode;
  chat: ReactNode;
}

/** Responsive two-pane shell: documents on the left, chat on the right. */
export function AppLayout({ documents, chat }: AppLayoutProps) {
  return (
    <AppRoot>
      <HeaderBar position="static" color="default" elevation={0}>
        <Toolbar variant="dense">
          <AutoAwesomeIcon color="primary" sx={{ mr: 1 }} />
          <Typography variant="h6" component="h1" sx={{ flexGrow: 1 }}>
            RAG Demo
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Grounded answers with citations
          </Typography>
        </Toolbar>
      </HeaderBar>

      <Container
        maxWidth="lg"
        sx={{ flex: 1, minHeight: 0, py: 2, display: "flex" }}
      >
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          sx={{ flex: 1, minHeight: 0, width: "100%" }}
        >
          <SidebarPanel variant="outlined">{documents}</SidebarPanel>
          <ContentPanel variant="outlined">{chat}</ContentPanel>
        </Stack>
      </Container>
    </AppRoot>
  );
}
