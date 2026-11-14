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

// ------------------------------------------------------------------ Web Push (grid alerts)
type PushPayload = { title?: string; body?: string; url?: string; tag?: string; level?: string };

self.addEventListener("push", (event: PushEvent) => {
  let p: PushPayload = {};
  try {
    p = event.data?.json() ?? {};
  } catch {
    p = { body: event.data?.text() };
  }
  event.waitUntil(
    self.registration.showNotification(p.title ?? "P.O.W.E.R. grid alert", {
      body: p.body ?? "",
      tag: p.tag ?? "power-alert", // a newer alert replaces the older one instead of stacking
      icon: "/icons/icon-192.png",
      badge: "/icons/icon-maskable-192.png",
      requireInteraction: p.level === "CRITICAL",
      data: { url: p.url ?? "/" },
    }),
  );
});

self.addEventListener("notificationclick", (event: NotificationEvent) => {
  event.notification.close();
  const url = new URL((event.notification.data?.url as string) ?? "/", self.location.origin).href;
  event.waitUntil(
    (async () => {
      const wins = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
      for (const w of wins) {
        if ("focus" in w) {
          await (w as WindowClient).navigate(url).catch(() => undefined);
          return (w as WindowClient).focus();
        }
      }
      return self.clients.openWindow(url);
    })(),
  );
});
