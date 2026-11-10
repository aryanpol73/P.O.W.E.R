// Service worker (Serwist). Precaches the app shell, serves an offline page for navigations, and
// keeps the most recent API reads (network-first) so the last snapshot is still visible offline.

import { defaultCache } from "@serwist/next/worker";
import type { PrecacheEntry, SerwistGlobalConfig } from "serwist";
import { ExpirationPlugin, NetworkFirst, Serwist } from "serwist";

declare global {
  interface WorkerGlobalScope extends SerwistGlobalConfig {
    __SW_MANIFEST: (PrecacheEntry | string)[] | undefined;
  }
}

declare const self: ServiceWorkerGlobalScope;

const serwist = new Serwist({
  precacheEntries: self.__SW_MANIFEST,
  skipWaiting: true,
  clientsClaim: true,
  navigationPreload: true,
  runtimeCaching: [
    {
      matcher: ({ url, request }) => request.method === "GET" && url.pathname.startsWith("/api/v1/"),
      handler: new NetworkFirst({
        cacheName: "power-api",
        networkTimeoutSeconds: 4,
        plugins: [new ExpirationPlugin({ maxEntries: 200, maxAgeSeconds: 24 * 3600 })],
      }),
    },
    ...defaultCache,
  ],
  fallbacks: {
    entries: [{ url: "/offline", matcher: ({ request }) => request.destination === "document" }],
  },
});

serwist.addEventListeners();
