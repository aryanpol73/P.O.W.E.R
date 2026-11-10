import withSerwistInit from "@serwist/next";
import type { NextConfig } from "next";

// The service worker is only built for production (`next build`); in dev it is disabled so hot
// reload is not cached.
const withSerwist = withSerwistInit({
  swSrc: "src/app/sw.ts",
  swDest: "public/sw.js",
  disable: process.env.NODE_ENV !== "production",
});

const nextConfig: NextConfig = {
  // A separate output folder lets a production build run next to `next dev` (which uses .next).
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
      { source: "/sw.js", headers: [{ key: "Cache-Control", value: "no-cache, no-store, must-revalidate" }] },
    ];
  },
};

export default withSerwist(nextConfig);
