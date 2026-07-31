import { styled } from "@mui/material/styles";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";

import { PanelPaper } from "../styles/shared";

/** Full-height flex column that hosts the header and content area. */
export const AppRoot = styled(Box)({
  display: "flex",
  flexDirection: "column",
  height: "100vh",
});

/** Top app bar with a subtle divider beneath it. */
export const HeaderBar = styled(AppBar)(({ theme }) => ({
  borderBottom: `1px solid ${theme.palette.divider}`,
}));

/** Fixed-width left pane on desktop, full-width when stacked on mobile. */
export const SidebarPanel = styled(PanelPaper)(({ theme }) => ({
  flexShrink: 0,
  width: "100%",
  minHeight: 280,
  [theme.breakpoints.up("md")]: {
    width: 360,
    minHeight: 0,
  },
}));

/** Flexible right pane that grows to fill remaining space. */
export const ContentPanel = styled(PanelPaper)(({ theme }) => ({
  flex: 1,
  minHeight: 400,
  [theme.breakpoints.up("md")]: {
    minHeight: 0,
  },
}));
