import type { ReactElement, ReactNode } from "react";
import { render, type RenderOptions } from "@testing-library/react";
import { ThemeProvider } from "@mui/material/styles";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { UserProvider } from "../context/UserContext";
import { theme } from "../theme";

/** Fresh QueryClient per render with retries off so failing queries surface
 * immediately instead of retrying through the test's timeout. */
export function makeTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

interface ProvidersOptions extends Omit<RenderOptions, "wrapper"> {
  queryClient?: QueryClient;
}

/** Render a component inside the same React Query + MUI theme context it gets
 * in the real app. Returns the created (or supplied) QueryClient so tests can
 * assert on cache behaviour. */
export function renderWithProviders(
  ui: ReactElement,
  { queryClient = makeTestQueryClient(), ...options }: ProvidersOptions = {},
) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <UserProvider>
          <ThemeProvider theme={theme}>{children}</ThemeProvider>
        </UserProvider>
      </QueryClientProvider>
    );
  }
  return { queryClient, ...render(ui, { wrapper: Wrapper, ...options }) };
}
