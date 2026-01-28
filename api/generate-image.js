// api/generate-image.js
// Image generation endpoint for Be Cre8v AI Playground (kid-safe + worksheet-ready + more realistic by default)

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

function isWorksheetPrompt(p) {
  const t = String(p || "").toLowerCase();
  return [
    "worksheet",
    "printable",
    "printable sheet",
    "activity sheet",
    "workbook",
    "quiz sheet",
    "exercise sheet",
    "practice sheet",
    "lesson sheet",
    "handout",
    "pdf",
    "a4",
    "classroom",
  ].some((k) => t.includes(k));
}

function wantsCartoonStyle(p) {
  const t = String(p || "").toLowerCase();
  return [
    "cartoon",
    "animated",
    "anime",
    "illustration",
    "vector",
    "flat",
    "comic",
    "pixar",
    "kids style",
    "storybook",
    "cute",
  ].some((k) => t.includes(k));
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  if (!allow(res, origin)) {
    return res.status(403).json({ message: "Forbidden origin", origin, allowed: origins() });
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

    const worksheetMode = isWorksheetPrompt(prompt);
    const keepCartoon = wantsCartoonStyle(prompt);

    const size = worksheetMode ? "1024x1024" : "1024x1024";

    const baseRules = `
Kid-safe image for children and families.
No hateful content, no nudity, no sexual content, no violence/gore.
No text watermark, no logos, no brand names.
`.trim();

    const styleRules = worksheetMode
      ? `
Make a printable worksheet page layout.
A4 portrait feel, clean white background, neat margins.
Clear section boxes with headings and spaces for writing.
High contrast, printer friendly (minimal color accents).
Simple small icons/illustrations only as decoration.
Keep text short and large (avoid tiny paragraphs).
`.trim()
      : `
Make it look high quality and not cartoon by default.
Photorealistic or semi-realistic, natural lighting, clean composition.
Avoid exaggerated cartoon outlines.
`.trim();

    const finalStyleRules = keepCartoon && !worksheetMode
      ? `
Friendly illustrated style is allowed because the user asked for it.
Still: no watermark, no logos, no text overlay unless the user explicitly requests text.
`.trim()
      : styleRules;

    const safePrompt = `
${baseRules}

${finalStyleRules}

User request:
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
        size,
      }),
    });

    if (!r.ok) {
      const err = await r.text();
      return res.status(r.status).json({
        message: "Image API error",
        details: err.slice(0, 800),
      });
    }

    const data = await r.json();

    const imageBase64 = data?.data?.[0]?.b64_json;
    if (!imageBase64) {
      return res.status(500).json({ message: "No image returned" });
    }

    // Most responses are usable as PNG; if OpenAI returns webp, browser will still render it.
    // We return a data URL and a filename so frontend can download.
    const mime = worksheetMode ? "image/png" : "image/png";
    const filename = worksheetMode ? "worksheet.png" : "image.png";

    return res.status(200).json({
      image: `data:${mime};base64,${imageBase64}`,
      filename,
      openai_request_id: data?.openai_request_id || null,
      meta: { worksheetMode, size },
    });
  } catch (e) {
    return res.status(500).json({ message: "Server error", details: String(e?.message || e) });
  }
}
