// TEMP DIAGNOSTIC HANDLER — paste over api/playground.js
// Purpose: show exactly what the function sees at runtime (no secrets revealed)

const CHAT_URL = "https://api.openai.com/v1/chat/completions";

function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",").map(s => s.trim()).filter(Boolean);
}
function allow(res, origin) {
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (!origin) return false;
  const list = origins();
  if (!list.length || list.some(a => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  if (req.method === "OPTIONS") { allow(res, origin); return res.status(204).end(); }

  // DEBUG: if GET, return the environment we see (sanitized)
  if (req.method === "GET") {
    allow(res, origin);
    const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "");
    const PRODUCTS_URL = String((process.env.PRODUCTS_URL || "").trim() ||
      "https://cdn.shopify.com/s/files/1/0797/0576/8213/files/products.json?v=1758262885");

    return res.status(200).json({
      ok: true,
      vercel: {
        env: process.env.VERCEL_ENV,
        url: process.env.VERCEL_URL,
        project: process.env.VERCEL_PROJECT_PRODUCTION_URL
      },
      seenEnvVars: [
        "OPENAI_API_KEY",
        "PRODUCTS_URL",
        "ALLOWED_ORIGIN"
      ],
      values: {
        OPENAI_API_KEY_length: OPENAI_KEY.length, // 0 means NOT set in this project/env
        PRODUCTS_URL,
        ALLOWED_ORIGIN: process.env.ALLOWED_ORIGIN || null
      }
    });
  }

  // NORMAL POST path (kept minimal; still verifies key exists)
  if (req.method !== "POST") { allow(res, origin); return res.status(405).json({ message: "Method not allowed" }); }
  try {
    if (!allow(res, origin)) return res.status(403).json({ message: "Forbidden origin" });

    const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "").trim();
    if (!OPENAI_KEY) {
      return res.status(500).json({ message: "Server config error", details: "OPENAI_API_KEY env var is empty" });
    }

    // …you can keep your real logic here; we just short-circuit to confirm key presence
    return res.status(200).json({ ok: true, message: "Key is present. You can switch back to the real handler." });

  } catch (e) {
    return res.status(500).json({ message: "Server error", details: String(e?.message || e) });
  }
}
