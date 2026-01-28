// api/generate-image.js
// Be Cre8v AI Playground — Image Generation (kid-safe)
// Focus: premium, highly-informational, A4 single-page worksheets (vintage / illustrated style)

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
    p.includes("mcq") ||
    p.includes("quiz") ||
    p.includes("match") ||
    p.includes("timeline") ||
    p.includes("cause") ||
    p.includes("effect") ||
    p.includes("compare") ||
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
Kid-safe educational content for children.
No sexual content. No gore. No violence instructions.
No watermark. No logos.
`.trim();

    // This is the key: push style + layout + density HARD
    const premiumVintageWorksheet = `
Create ONE SINGLE PAGE printable worksheet (A4 portrait).
Make it extremely beautiful and premium, like a museum-style illustrated worksheet / vintage parchment poster.

Visual style (very important):
- vintage parchment paper background with subtle texture
- warm earthy palette (tan, sepia, muted reds/greens/blues)
- decorative header banner, small icons, elegant borders/dividers
- high-quality hand-painted illustration style (not cartoon), like an old history book illustration
- very aesthetic, balanced, professional layout, lots of visual polish

Layout rules:
- single page only, no multi-page, no repeated pages
- clean grid layout with consistent margins
- 6 to 8 content blocks maximum, clearly separated
- readable hierarchy: big title, section headings, body text, checkboxes/lines
- include 1 hero illustration and 2–3 small supporting illustrations/icons
- include fill-in lines and small checkboxes where appropriate

Make it highly informational (very important):
Include these sections (adapt to the topic):
1) quick facts (4–6 bullets)
2) key people / key terms box (short)
3) timeline (5–7 points)
4) causes vs effects (two-column)
5) map/places/events (small box, even if symbolic)
6) critical thinking (3 prompts)
7) mini quiz (4 questions with options A/B/C or true/false)
8) reflection (1 longer prompt with lines)

Text rendering note:
- keep text short, clear, aligned, and large enough to read
- prefer bullets and short lines
- no tiny paragraphs
`.trim();

    const realisticIllustration = `
Style:
- realistic, high-quality illustration (not cartoon)
- natural lighting, clean composition
- no exaggerated anime/cartoon look
`.trim();

    const finalPrompt = `
${safety}

${isWorksheet ? premiumVintageWorksheet : realisticIllustration}

User request:
${String(prompt).trim()}

Hard constraints:
- single page only
- printable worksheet/poster aesthetic
- very information-dense but not cluttered
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
        // portrait for worksheets gives far better layouts
        size: isWorksheet ? "1024x1536" : "1024x1024",
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

    // Use webp for size; browser download works fine
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
