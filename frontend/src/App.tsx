import { AppLayout } from "./components/AppLayout";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentList } from "./components/DocumentList";

export default function App() {
  return <AppLayout documents={<DocumentList />} chat={<ChatPanel />} />;
}
