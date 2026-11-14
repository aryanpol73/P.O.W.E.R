// Server-side proxy to the copilot on the MCP service. The copilot token (and, on the MCP service, the
// Anthropic API key) never reach the browser; the response is streamed through unchanged (NDJSON events).

import { NextResponse, type NextRequest } from "next/server";

const MCP = (process.env.POWER_MCP_URL ?? "http://127.0.0.1:8765").replace(/\/$/, "");

function headers(): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (process.env.POWER_COPILOT_TOKEN) h.authorization = `Bearer ${process.env.POWER_COPILOT_TOKEN}`;
  return h;
}

export async function GET() {
  try {
    const res = await fetch(`${MCP}/copilot/status`, { cache: "no-store" });
    return new NextResponse(await res.text(), { status: res.status, headers: { "content-type": "application/json" } });
  } catch {
    return NextResponse.json({ enabled: false, detail: "MCP service unreachable" });
  }
}

export async function POST(req: NextRequest) {
  const body = await req.text();
  if (body.length > 8000) return NextResponse.json({ detail: "message too long" }, { status: 413 });
  let res: Response;
  try {
    res = await fetch(`${MCP}/copilot`, { method: "POST", headers: headers(), body, cache: "no-store" });
  } catch {
    return NextResponse.json({ detail: "MCP service unreachable" }, { status: 502 });
  }
  return new NextResponse(res.body, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") ?? "application/x-ndjson", "cache-control": "no-store" },
  });
}
