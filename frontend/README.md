# RAG Demo - Web UI

A small React front end for the RAG Demo API: drag-and-drop document upload,
per-file management (delete / replace), and a grounded, cited chat over your
uploaded documents.

Built with **Vite + React + TypeScript + MUI**, with server state managed by
**TanStack Query** and drag-and-drop via **react-dropzone**.

## Prerequisites

- Node.js 18+ and npm.
- The RAG API running locally (see the repository root README):

```bash
uvicorn rag.api:app --reload
```

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # optional; the defaults work for local dev
npm run dev
```

Open the printed URL (defaults to http://localhost:5173).

During development the Vite dev server **proxies** the API endpoints
(`/ingest`, `/chat`, `/documents`, `/conversations`, `/health`, `/ask`) to the
backend at `http://127.0.0.1:8000`, so the browser talks to a single origin and
there are no local CORS issues. Point the proxy elsewhere with
`VITE_API_PROXY_TARGET`.

For a deployed backend, set `VITE_API_BASE_URL` to the API's absolute URL and
build with `npm run build` (output in `dist/`).

## How identity works

The API is multi-tenant via an `X-User-Id` header (a correlation/tenant key,
**not** authentication). On the first request the server mints an id and echoes
it back; the client stores it in `localStorage` and reuses it, so your uploaded
corpus and conversations persist across reloads. Chat threads are continued via
the `X-Conversation-Id` header, also persisted; use the "new conversation"
button to start a fresh thread.

## Scripts

| Script            | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `npm run dev`     | Start the Vite dev server with API proxy |
| `npm run build`   | Type-check and build for production      |
| `npm run preview` | Preview the production build             |
| `npm run lint`    | Lint the source                          |

## Structure

```
frontend/src/
├── api/            # typed API client (identity headers, error envelope) + types
├── hooks/          # TanStack Query hooks (documents, upload, delete, chat)
├── components/     # AppLayout, FileDropzone, DocumentList, ChatPanel, MessageBubble, Citations
├── lib/            # error-code -> message mapping
├── theme.ts        # MUI theme
├── App.tsx
└── main.tsx
```
