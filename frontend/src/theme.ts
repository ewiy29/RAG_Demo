import { createTheme } from "@mui/material/styles";

// A single, restrained theme. Uses the system light/dark preference so the app
// respects the OS setting without a manual toggle (kept simple by design).
const prefersDark =
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-color-scheme: dark)").matches;

export const theme = createTheme({
  palette: {
    mode: prefersDark ? "dark" : "light",
    primary: { main: "#2563eb" },
    secondary: { main: "#7c3aed" },
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily:
      '"Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    h6: { fontWeight: 600 },
  },
  components: {
    MuiButton: { defaultProps: { disableElevation: true } },
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
  },
});
