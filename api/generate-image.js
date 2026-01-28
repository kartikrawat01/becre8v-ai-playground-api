// api/generate-image.js
// Be Cre8v AI Playground — Image Generation (kid-safe)
// Focus: premium, highly-informational, A4 single-page worksheets (vintage / illustrated style)

const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";
const OPENAI_CHAT_URL  = "https://api.openai.com/v1/chat/completions";

/* ------------------------------------------------------------------ */
/* PROMPT ENHANCER SYSTEM PROMPT (NEW)                                 */
/* ------------------------------------------------------------------ */
const PROMPT_ENHANCER_SYSTEM_PROMPT = `
You are the Be Cre8v AI Playground Image Prompt Enhancer.

Your job is to rewrite user image requests into extremely high-quality,
structured, descriptive prompts suitable for premium educational image generation.

Rules:
- Do NOT mention AI, prompts, or instructions in the output
- Output ONLY the final rewritten image prompt text
- Make images educational, aesthetic, information-rich
- Prefer printable, clean layouts when applicable
- Avoid cartoons unless explicitly requested
- Default age range: children 8–15
- Do not add animation unless explicitly requested
- Do not remove factual density
- Improve layout clarity, hierarchy, and visual guidance

You do NOT generate images.
You ONLY rewrite prompts.
`.trim();

/* ------------------------------------------------------------------ */
/* CORS HELPERS (UNCHANGED)                                            */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* INTENT DETECTION (UNCHANGED)                                        */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* PROMPT ENHANCER (NEW, SAFE, INTERNAL)                               */
/* ------------------------------------------------------------------ */
async function enhanceImagePrompt(rawPrompt, apiKey) {
  const r = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [
        { role: "system", content: PROMPT_ENHANCER_SYSTEM_PROMPT },
        { role: "user", content: rawPrompt }
      ]
    })
  });

  if (!r.ok) {
    // Fail safe: return original prompt if enhancer fails
    return rawPrompt;
  }

  const data = await r.json();
  return data?.choices?.[0]?.message?.content?.trim() || rawPrompt;
}

/* ------------------------------------------------------------------ */
/* MAIN HANDLER                                                        */
/* ------------------------------------------------------------------ */
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

    /* ------------------ SAFETY + STYLE (UNCHANGED) ------------------ */
    const safety = `
Kid-safe educational content for children.
No sexual content. No gore. No violence instructions.
No watermark. No logos.
`.trim();

    const premiumVintageWorksheet = `
`
Create ONE SINGLE PAGE printable worksheet (A4 portrait).
Make it extremely beautiful and premium, like a museum-style illustrated worksheet / vintage parchment poster.

Visual style:
- vintage parchment background with subtle texture
- warm earthy palette (tan, sepia, muted reds/greens/blues)
- decorative header banner, elegant borders and dividers
- hand-painted historical illustration style (not cartoon)
- very balanced, professional, high-polish layout

Layout rules:
- single page only
- clean grid layout, strong margins
- 6–8 content blocks max
- clear hierarchy: title → sections → prompts
- include 1 hero illustration + 2–3 small icons
- include fill-in lines and checkboxes

Content density:
1) quick facts
2) key people / terms
3) timeline
4) causes vs effects
5) places / map box
6) critical thinking
7) mini quiz
8) reflection
`.trim();

    const realisticIllustration = `
Style:
- realistic, high-quality illustration
- clean composition
- educational clarity
- not cartoon or anime
`.trim();

    /* ------------------ BASE PROMPT (UNCHANGED) ------------------ */
    const basePrompt = `
${safety}

${isWorksheet ? premiumVintageWorksheet : realisticIllustration}

User request:
${String(prompt).trim()}

Hard constraints:
- single page only
- printable educational aesthetic
- high information density without clutter
`.trim();

    /* ------------------ ENHANCE PROMPT (NEW STEP) ------------------ */
    const enhancedPrompt = await enhanceImagePrompt(basePrompt, apiKey);

    /* ------------------ IMAGE GENERATION ------------------ */
    const r = await fetch(OPENAI_IMAGE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-image-1",
        prompt: enhancedPrompt,
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
