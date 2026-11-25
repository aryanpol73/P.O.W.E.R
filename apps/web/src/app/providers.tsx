"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { LiveProvider } from "@/lib/live";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          // A free-tier API sleeps when idle and needs about a minute to wake: keep retrying for ~90 s.
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
            retry: 7,
            retryDelay: (n) => Math.min(2_000 * 2 ** n, 20_000),
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <LiveProvider>{children}</LiveProvider>
    </QueryClientProvider>
  );
}
