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
  // If no list is set, allow all (optional). You can remove this if you want strict mode only.
  if (!list.length) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }

  if (list.some((a) => origin === a || origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }

  return false;
}

// Works even when req.body is undefined (common in some serverless setups)
async function readJsonBody(req) {
  if (req.body && typeof req.body === "object") return req.body;

  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8");

  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  if (!allow(res, origin)) {
    return res.status(403).json({
      message: "Forbidden origin",
      origin,
      allowed: origins(),
    });
  }

  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const body = await readJsonBody(req);
    const prompt = body?.prompt;

    if (!prompt || !String(prompt).trim()) {
      return res.status(400).json({
        message: "Prompt is required",
        got: typeof prompt,
      });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ message: "Missing OPENAI_API_KEY" });
    }

    const safePrompt = `
Kid-safe illustration for children.
No text, no watermark, no logos.
Bright colors, friendly style.
No scary, violent, or unsafe content.

${String(prompt).trim()}
`.trim();

    const r = await fetch(OPENAI_IMAGE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-image-1",
        prompt: safePrompt,
        size: "1024x1024",
        // Smaller payloads than PNG (often fixes hosting/proxy limits)
        output_format: "webp",
        // quality can be "low"|"medium"|"high" or "auto" depending on docs; keep auto-safe:
        quality: "auto",
        n: 1,
      }),
    });

    const openaiRequestId =
      r.headers.get("x-request-id") || r.headers.get("request-id") || null;

    if (!r.ok) {
      const errText = await r.text();
      return res.status(r.status).json({
        message: "Image API error",
        status: r.status,
        openai_request_id: openaiRequestId,
        details: errText.slice(0, 1500),
      });
    }

    const data = await r.json();

    const imageBase64 = data?.data?.[0]?.b64_json;
    if (!imageBase64) {
      return res.status(500).json({
        message: "No image returned",
        openai_request_id: openaiRequestId,
        raw_keys: Object.keys(data || {}),
      });
    }

    return res.status(200).json({
      // Match output_format above
      image: `data:image/webp;base64,${imageBase64}`,
      openai_request_id: openaiRequestId,
    });
  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e),
    });
  }
}
