import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend API host to proxy to during development. Every RAG endpoint is
// forwarded so the browser talks to the Vite dev server on one origin and never
// trips CORS locally. Override with VITE_API_PROXY_TARGET if the API runs
// elsewhere.
const API_TARGET = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

const API_PATHS = [
  "/ingest",
  "/ask",
  "/chat",
  "/documents",
  "/conversations",
  "/health",
];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [path, { target: API_TARGET, changeOrigin: true }]),
    ),
  },
  build: {
    // Split heavy, rarely-changing vendor code out of the app bundle so it
    // caches independently across deploys and the initial payload is smaller.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) {
            return undefined;
          }
          if (id.includes("@mui") || id.includes("@emotion")) {
            return "mui";
          }
          if (id.includes("@tanstack")) {
            return "query";
          }
          if (
            id.includes("react-dom") ||
            id.includes("/react/") ||
            id.includes("scheduler")
          ) {
            return "react";
          }
          return "vendor";
        },
      },
    },
  },
});
