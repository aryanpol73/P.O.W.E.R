import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";

import { AppShell } from "@/components/AppShell";

import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import { Providers } from "./providers";

const sans = Manrope({ variable: "--font-body", subsets: ["latin"], display: "swap" });
const mono = IBM_Plex_Mono({ variable: "--font-plexmono", subsets: ["latin"], weight: ["400", "500", "600"], display: "swap" });

export const metadata: Metadata = {
  title: "P.O.W.E.R. Grid Console",
  description: "Real-time EV charging demand forecasts, overload alerts and explanations.",
  applicationName: "P.O.W.E.R.",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "P.O.W.E.R.", statusBarStyle: "black-translucent" },
  formatDetection: { telephone: false },
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/icons/apple-touch-icon.png", sizes: "180x180" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#090c18",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="dark">
      <body className={`${sans.variable} ${mono.variable}`}>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
