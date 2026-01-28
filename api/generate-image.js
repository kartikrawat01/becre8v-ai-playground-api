// api/generate-image.js
// Image generation endpoint for Be Cre8v AI Playground (kid-safe) - single page only

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

function looksLikeWorksheet(prompt) {
  const p = String(prompt || "").toLowerCase();
  return (
    p.includes("worksheet") ||
    p.includes("printable") ||
    p.includes("sheet") ||
    p.includes("activity") ||
    p.includes("fill in") ||
    p.includes("quiz") ||
    p.includes("mcq") ||
    p.includes("match") ||
    p.includes("timeline") ||
    p.includes("critical thinking")
  );
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

    const isWorksheet = looksLikeWorksheet(prompt);

    const safety = `
Kid-safe content for children.
No sexual content. No gore. No weapons instructions.
No watermark. No logos.
`.trim();

    const realisticStyle = `
Style:
- high quality, realistic (not cartoon), natural lighting
- warm, child-friendly, but realistic
- clean composition, sharp details
- no exaggerated anime/cartoon look
`.trim();

    const premiumWorksheetStyle = `
Create a single-page premium educational worksheet for ages 10–15.
Page format:
- A4 portrait layout (print-friendly)
- clear grid layout with consistent margins and spacing
- visually rich but uncluttered
Design language:
- modern textbook + museum exhibit feel (premium)
- tasteful color accents (not loud), mostly white background
- include small high-quality illustrative visuals relevant to the topic (realistic illustration style)
Content:
- informative + critical thinking
- include 4–6 sections max (example: quick facts, timeline, cause-effect, compare/contrast, reflection question, mini quiz)
- include answer boxes/lines where needed
- keep text readable and aligned
Do not look like a basic black-and-white form. It should feel designed by a professional education brand.
`.trim();

    const finalPrompt = `
${safety}

${isWorksheet ? premiumWorksheetStyle : realisticStyle}

User request:
${String(prompt).trim()}
`.trim();

    const r = await fetch(OPENAI_IMAGE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-image-1",
        prompt: finalPrompt,
        // A4-ish portrait improves worksheet layout a lot
        size: isWorksheet ? "1024x1536" : "1024x1024",
        // single image only
        n: 1
      })
    });

    if (!r.ok) {
      const err = await r.text();
      return res.status(r.status).json({
        message: "Image API error",
        details: err.slice(0, 1500)
      });
    }

    const data = await r.json();
    const b64 = data?.data?.[0]?.b64_json;

    if (!b64) {
      return res.status(500).json({ message: "No image returned" });
    }

    // webp is smaller; browser displays + downloads fine
    const mime = "image/webp";
    const dataUrl = `data:${mime};base64,${b64}`;

    return res.status(200).json({
      mime,
      image: dataUrl
    });
  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e)
    });
  }
}
