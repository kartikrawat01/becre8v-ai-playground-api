// api/generate-image.js
// Kid-safe image generation endpoint for Be Cre8v AI Playground

const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";

/* -------------------- CORS -------------------- */

function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function setCors(res, origin) {
  // Always set these headers (even on errors)
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  const list = origins();

  // If no allowlist configured, keep it locked
  if (!list.length) return { ok: false, reason: "ALLOWED_ORIGIN not set" };

  // For browser calls we require an Origin header
  // (direct browser navigation to the endpoint has no Origin - that's normal)
  if (!origin) return { ok: false, reason: "Missing Origin header" };

  const ok = list.some((a) => origin.startsWith(a));
  if (ok) res.setHeader("Access-Control-Allow-Origin", origin);

  return { ok, reason: ok ? "ok" : `Origin not allowed: ${origin}` };
}

/* -------------------- helpers -------------------- */

function clamp(s, max = 500) {
  const t = String(s || "").trim();
  return t.length > max ? t.slice(0, max) : t;
}

/* -------------------- handler -------------------- */

export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  const cors = setCors(res, origin);

  // Preflight must always succeed (CORS headers already set above)
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  // Only POST is allowed
  if (req.method !== "POST") {
    return res.status(405).json({
      message:
        "Method not allowed. Use POST from the AI Playground (this endpoint blocks direct browser opens)."
    });
  }

  // Block requests whose origin isn't in allowlist
  if (!cors.ok) {
    return res.status(403).json({
      message: "Forbidden origin",
      details: cors.reason
    });
  }

  // Validate env key
  const apiKey = String(process.env.OPENAI_API_KEY || "").trim();
  if (!apiKey) {
    return res.status(500).json({ message: "Missing OPENAI_API_KEY" });
  }

  try {
    const { prompt = "" } = req.body || {};
    const p = clamp(prompt, 400);
    if (!p) return res.status(400).json({ message: "Prompt is required" });

    // Kid-safe instruction wrapper
    const safePrompt = [
      "Create a kid-safe, playful illustration for children.",
      "No violence, no adult content, no realistic faces of real people.",
      "No copyrighted logos, no watermark, no readable text.",
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

    // Return BOTH keys so frontend never mismatches
    const dataUrl = `data:image/png;base64,${b64}`;
    return res.status(200).json({
      dataUrl,
      image: dataUrl
    });
  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e)
    });
  }
}
