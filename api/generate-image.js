// api/generate-image.js
// Image generation endpoint for Be Cre8v AI Playground (kid-safe)

const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";

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

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  if (!allow(res, origin)) {
    return res.status(403).json({ message: "Forbidden origin" });
  }

  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const { prompt } = req.body || {};

    if (!prompt || !String(prompt).trim()) {
      return res.status(400).json({ message: "Prompt is required" });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ message: "Missing OPENAI_API_KEY" });
    }

    const safePrompt = `
Kid-safe illustration for children.
No text, no watermark.
Bright colors, friendly style.

${prompt}
`.trim();

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
      const err = await r.text();
      return res
        .status(r.status)
        .json({ message: "Image API error", details: err.slice(0, 500) });
    }

    const data = await r.json();

    const imageBase64 = data?.data?.[0]?.b64_json;
    if (!imageBase64) {
      return res.status(500).json({ message: "No image returned" });
    }

    return res.status(200).json({
      image: `data:image/png;base64,${imageBase64}`
    });
  } catch (e) {
    return res
      .status(500)
      .json({ message: "Server error", details: String(e) });
  }
}
