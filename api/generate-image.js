// api/generate-image.js
// Kid-safe image generation endpoint for Be Cre8v AI Playground

const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";

/* ---------- CORS helpers (MATCH playground.js style) ---------- */

function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function allow(res, origin) {
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  // IMPORTANT: Only allow known origins
  if (!origin) return false;

  const list = origins();

  // If allowlist is empty, block
  if (!list.length) return false;

  if (list.some((a) => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

/* ---------- helpers ---------- */
function clamp(s, max = 500) {
  const t = String(s || "").trim();
  return t.length > max ? t.slice(0, max) : t;
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // OPTIONS preflight
  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  // DEBUG endpoint (so we can see exact env + origin reaching server)
  // Open in browser: https://.../api/generate-image?debug=1
  if (req.method === "GET" && req.query && req.query.debug === "1") {
    const list = origins();
    return res.status(200).json({
      seenOrigin: origin || null,
      allowedOriginEnv: process.env.ALLOWED_ORIGIN || "",
      allowedOriginsParsed: list
    });
  }

  if (req.method !== "POST") {
    allow(res, origin);
    return res.status(405).json({ message: "Method not allowed" });
  }

  // CORS allow check
  if (!allow(res, origin)) {
    return res.status(403).json({
      message: "Forbidden origin",
      details: {
        seenOrigin: origin || null,
        allowedOriginEnv: process.env.ALLOWED_ORIGIN || ""
      }
    });
  }

  // Env key check
  const apiKey = String(process.env.OPENAI_API_KEY || "").trim();
  if (!apiKey) {
    return res.status(500).json({ message: "Missing OPENAI_API_KEY" });
  }

  try {
    const { prompt = "" } = req.body || {};
    const p = clamp(prompt, 400);
    if (!p) return res.status(400).json({ message: "Prompt is required" });

    const safePrompt = [
      "Create a kid-safe, playful illustration for children.",
      "No violence, no adult content, no copyrighted logos, no watermark, no readable text.",
      "Bright colors, clean shapes, friendly style.",
      "",
      `Prompt: ${p}`
    ].join("\n");

    const r = await fetch(OPENAI_IMAGE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-image-1",
        prompt: safePrompt,
        size: "1024x1024"
      })
    });

    if (!r.ok) {
      const err = await r.text().catch(() => "");
      return res.status(r.status).json({
        message: "OpenAI image error",
        details: err.slice(0, 1200)
      });
    }

    const data = await r.json();
    const b64 = data?.data?.[0]?.b64_json;

    if (!b64) {
      return res.status(500).json({ message: "No image returned" });
    }

    const dataUrl = `data:image/png;base64,${b64}`;
    return res.status(200).json({ dataUrl, image: dataUrl });
  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e)
    });
  }
}
