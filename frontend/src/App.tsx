import { useQueryClient } from "@tanstack/react-query";

import { AppLayout } from "./components/AppLayout";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentList } from "./components/DocumentList";
import { UserSwitcher } from "./components/UserSwitcher";
import { useUser } from "./context/UserContext";

export default function App() {
  const queryClient = useQueryClient();
  const { userId, switchUser, addUser } = useUser();

  // Changing identity must not leak the previous tenant's data: clear the
  // React Query cache so the remounted panels re-fetch (e.g. GET /documents)
  // for the newly active user instead of serving the prior user's cache.
  const handleSwitch = (id: string) => {
    switchUser(id);
    queryClient.clear();
  };

  const handleAddUser = () => {
    addUser();
    queryClient.clear();
  };

  return (
    <AppLayout
      userControl={
        <UserSwitcher
          activeUserId={userId}
          onSwitch={handleSwitch}
          onAddUser={handleAddUser}
        />
      }
      documents={<DocumentList key={userId} />}
      chat={<ChatPanel key={userId} />}
    />
  );
}
