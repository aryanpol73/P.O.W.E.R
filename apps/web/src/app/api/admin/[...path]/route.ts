// Server-side proxy for admin actions (sim control, alert ack). The admin token is read from the
// server environment (ADMIN_TOKEN) and never reaches the browser bundle.

import { NextResponse, type NextRequest } from "next/server";

const API = (process.env.POWER_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const ALLOWED = [
  /^sim\/(control|tick)$/,
  /^alerts\/\d+\/ack$/,
  /^agents\/(drift\/run|discovery\/run|retrain|retrain\/rollback|retrain\/[A-Za-z0-9]+\/activate)$/,
];

export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = path.join("/");
  if (!ALLOWED.some((re) => re.test(target))) {
    return NextResponse.json({ detail: "not an admin endpoint" }, { status: 404 });
  }
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (process.env.ADMIN_TOKEN) headers.authorization = `Bearer ${process.env.ADMIN_TOKEN}`;
  const body = await req.text();
  const res = await fetch(`${API}/api/v1/${target}`, { method: "POST", headers, body: body || undefined, cache: "no-store" });
  return new NextResponse(await res.text(), { status: res.status, headers: { "content-type": "application/json" } });
}
