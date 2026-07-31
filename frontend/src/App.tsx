import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { addUser, ensureUserId, switchUser } from "./api/client";
import { AppLayout } from "./components/AppLayout";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentList } from "./components/DocumentList";
import { UserSwitcher } from "./components/UserSwitcher";

export default function App() {
  const queryClient = useQueryClient();
  const [activeUserId, setActiveUserId] = useState(() => ensureUserId());

  // Changing identity must not leak the previous tenant's data: clear the
  // React Query cache so the remounted panels re-fetch (e.g. GET /documents)
  // for the newly active user instead of serving the prior user's cache.
  const applyIdentity = (id: string) => {
    queryClient.clear();
    setActiveUserId(id);
  };

  const handleSwitch = (id: string) => {
    switchUser(id);
    applyIdentity(id);
  };

  const handleAddUser = () => {
    applyIdentity(addUser());
  };

  return (
    <AppLayout
      userControl={
        <UserSwitcher
          activeUserId={activeUserId}
          onSwitch={handleSwitch}
          onAddUser={handleAddUser}
        />
      }
      documents={<DocumentList key={activeUserId} />}
      chat={<ChatPanel key={activeUserId} />}
    />
  );
}
