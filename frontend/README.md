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
**not** authentication). Identity lives in a single React context
(`src/context/UserContext.tsx`); nothing else reads or writes storage. The
provider seeds a tenant GUID on first load (the API also accepts any
client-supplied id, minting one itself if absent) and persists only the tenant
id and user roster to `localStorage` -- read once on load to hydrate the
context, and written back when you switch/add a user -- so your uploaded corpus
persists across reloads.

Chat threads are continued via the `X-Conversation-Id` header, but the
conversation id is kept **in memory only** (context state, never `localStorage`):
it resets on reload or when you switch users. Use the "new conversation" button
to start a fresh thread.

### Demo: switching users to show isolation

The header has a **user switcher** that lets you flip between a small roster of
tenant GUIDs (`User 1`, `User 2`, ... plus `+ New user`). Each entry is a
separate `X-User-Id`, so switching changes which tenant's corpus and
conversations the API returns. Try it:

1. Upload a document as `User 1`.
2. Switch to a new user - the document list is empty (a different tenant).
3. Switch back to `User 1` - the app clears its client cache and re-fetches
   `GET /documents`, so your original document reappears.

This visibly demonstrates that each user's uploads are isolated to that user.

It is a **demo affordance, not authentication**: anyone can hand-set any
`X-User-Id`, so this proves data partitioning, not a security boundary. In
production, identity would arrive as a validated signed token (e.g. an
OIDC/JWT bearer token) verified in the backend, with the tenant key extracted
from a trusted claim rather than a client-asserted header - while managed
identity handles service-to-service auth for the backend's own dependencies.

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
├── components/     # AppLayout, UserSwitcher, FileDropzone, DocumentList, ChatPanel, MessageBubble, Citations
├── lib/            # error-code -> message mapping
├── theme.ts        # MUI theme
├── App.tsx
└── main.tsx
```
