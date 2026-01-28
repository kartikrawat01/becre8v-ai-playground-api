// api/generate-image.js
// Image generation endpoint (used by "/image ..." command)

const IMG_URL = "https://api.openai.com/v1/images/generations";

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
  if (!origin) return false;
  const list = origins();
  if (!list.length || list.some((a) => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

function clamp(s, max = 500) {
  const t = String(s || "").trim();
  return t.length > max ? t.slice(0, max) : t;
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    allow(res, origin);
    return res.status(405).json({ message: "Method not allowed" });
  }

  if (!allow(res, origin)) {
    return res.status(403).json({ message: "Forbidden origin" });
  }

  const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "").trim();
  if (!OPENAI_KEY) {
    return res.status(500).json({ message: "OPENAI_API_KEY missing" });
  }

  try {
    const { prompt = "" } = req.body || {};
    const p = clamp(prompt, 400);
    if (!p) return res.status(400).json({ message: "Empty prompt" });

    const safePrompt =
      "Create a kid-safe, playful illustration for Be Cre8v. " +
      "No real people, no violence, no adult themes, no copyrighted logos. " +
      "Bright colors, clean shapes. Prompt: " + p;

    const r = await fetch(IMG_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`,
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
        details: err.slice(0, 800)
      });
    }

    const data = await r.json();
    const b64 = data?.data?.[0]?.b64_json;

    if (!b64) {
      return res.status(500).json({ message: "No image returned" });
    }

    return res.status(200).json({
      dataUrl: `data:image/png;base64,${b64}`
    });
  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e)
    });
  }
}
