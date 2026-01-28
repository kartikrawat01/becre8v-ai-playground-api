// api/generate-image.js
// Be Cre8v AI Playground — Image Generation (kid-safe)
// Goal: noticeably better image quality + worksheet aesthetics using a prompt-enhancer
// API unchanged: POST { prompt } -> { mime, image (dataURL), filename }

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
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
    p.includes("critical thinking") ||
    p.includes("a4") ||
    p.includes("handout") ||
    p.includes("classroom")
  );
}

function clampText(s, maxChars) {
  const t = String(s || "").trim();
  if (t.length <= maxChars) return t;
  return t.slice(0, maxChars - 1) + "…";
}

function safeFilename(base) {
  const cleaned = String(base || "image")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${cleaned || "image"}-${stamp}.webp`;
}

function isProbablyTooLongForServerless(prompt) {
  // Avoid huge prompts that may cause platform/request issues.
  // We still accept long prompts, but trim aggressively before OpenAI calls.
  return String(prompt || "").length > 6000;
}

async function fetchJsonWithTimeout(url, options, timeoutMs = 60000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: ctrl.signal });
    return resp;
  } finally {
    clearTimeout(t);
  }
}

/* ---------------- Prompt Enhancer ---------------- */

function buildPromptEnhancerSystem() {
  return `
You are Be Cre8v Image Prompt Enhancer.
Your job: rewrite the user's request into a BEST-POSSIBLE image-generation prompt for a kids STEM/learning brand.
Output ONLY the final improved prompt text. No explanations, no markdown, no labels.

Global rules:
- Always keep the user's core intent, topic, and constraints.
- If the user asks for a worksheet/printable: make it ONE SINGLE PAGE, A4 portrait, beautifully designed, information-dense, readable, with clear sections and space to write.
- If not a worksheet: make a premium, clean, kid-friendly illustration/poster/photo-style image as appropriate (still safe).
- Use solid-color design language cues (no neon gradients). Prefer clean backgrounds, tasteful accents, strong typography hierarchy.
- Must be kid-safe: no sexual content, no gore, no self-harm, no weapon-making, no hate, no adult themes.
- No watermarks, no brand logos, no copyrighted character references.

For worksheets specifically:
- A4 portrait single page, printable, with margins, neat grid, clear headings.
- 6–9 sections max. Use bullets, checkboxes, short lines, not tiny paragraphs.
- Include at least: quick facts, timeline, cause/effect, vocabulary/key people, 3 critical-thinking prompts, 4-question mini quiz, reflection lines.
- Add tasteful small icons/illustrations relevant to the topic (2–4), plus 1 main hero illustration.
- Ensure text is large enough to read.
- Style option: "premium illustrated classroom handout" or "vintage history worksheet" ONLY if it fits the topic. Otherwise use "clean modern classroom worksheet".

For general images:
- Decide best style: clean vector illustration, realistic illustration, poster, product mock, etc.
- Use clear composition and high quality.
- Add useful details but avoid clutter.
`.trim();
}

async function enhancePrompt({ apiKey, userPrompt, isWorksheet }) {
  // Trim user prompt so enhancer never gets insanely long input
  const trimmedUser = clampText(userPrompt, 2500);

  // Add a tiny hint to help the enhancer decide worksheet vs non-worksheet.
  const classifierHint = isWorksheet
    ? "The user is requesting a worksheet/printable."
    : "The user is requesting a general image (not necessarily a worksheet).";

  const payload = {
    model: "gpt-4o-mini",
    temperature: 0.25,
    messages: [
      { role: "system", content: buildPromptEnhancerSystem() },
      {
        role: "user",
        content:
          `${classifierHint}\n\nUser request:\n` +
          trimmedUser +
          `\n\nHard constraints:\n- kid-safe\n- no watermarks/logos\n- keep it single image output`,
      },
    ],
  };

  const r = await fetchJsonWithTimeout(
    OPENAI_CHAT_URL,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    45000
  );

  if (!r.ok) {
    const err = await r.text().catch(() => "");
    return {
      ok: false,
      prompt: null,
      error: `Prompt enhancer error status=${r.status} ${err.slice(0, 600)}`,
    };
  }

  const data = await r.json().catch(() => ({}));
  const out =
    data?.choices?.[0]?.message?.content &&
    String(data.choices[0].message.content).trim();

  if (!out) {
    return { ok: false, prompt: null, error: "Prompt enhancer returned empty output." };
  }

  // Clamp again to be safe for image endpoint
  const finalEnhanced = clampText(out, 3500);
  return { ok: true, prompt: finalEnhanced, error: null };
}

/* ---------------- Main Handler ---------------- */

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

    const userPromptRaw = String(prompt).trim();
    const isWorksheet = looksLikeWorksheet(userPromptRaw);

    // If prompt is extremely long, trim early so platforms don’t choke
    const userPrompt = isProbablyTooLongForServerless(userPromptRaw)
      ? clampText(userPromptRaw, 5500)
      : userPromptRaw;

    // 1) Enhance prompt (big quality jump)
    const enhanced = await enhancePrompt({ apiKey, userPrompt, isWorksheet });

    // Fallback: still work even if enhancer fails
    const finalPrompt = (enhanced.ok && enhanced.prompt) ? enhanced.prompt : userPrompt;

    // 2) Image generation
    // Important: request base64 explicitly so we can return a dataURL reliably
    const imgPayload = {
      model: "gpt-image-1",
      prompt: finalPrompt,
      size: isWorksheet ? "1024x1536" : "1024x1024",
      n: 1,
      response_format: "b64_json",
    };

    const r = await fetchJsonWithTimeout(
      OPENAI_IMAGE_URL,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(imgPayload),
      },
      90000
    );

    if (!r.ok) {
      const err = await r.text().catch(() => "");
      return res.status(r.status).json({
        message: "Image API error",
        details: err.slice(0, 1500),
        // useful for debugging quickly:
        debug: {
          isWorksheet,
          enhancerOk: enhanced.ok,
          enhancerError: enhanced.ok ? null : enhanced.error,
        },
      });
    }

    const data = await r.json().catch(() => ({}));
    const b64 = data?.data?.[0]?.b64_json;

    if (!b64) {
      return res.status(500).json({
        message: "No image returned",
        debug: {
          isWorksheet,
          enhancerOk: enhanced.ok,
          enhancerError: enhanced.ok ? null : enhanced.error,
          hasData: !!data?.data?.length,
          firstKeys: data?.data?.[0] ? Object.keys(data.data[0]) : [],
        },
      });
    }

    const mime = "image/webp";
    const dataUrl = `data:${mime};base64,${b64}`;

    // Better filename
    const filename = isWorksheet
      ? safeFilename("worksheet")
      : safeFilename("image");

    return res.status(200).json({
      mime,
      image: dataUrl,
      filename,
      // Optional: quick debug toggle (keep false by default)
      // debug: { isWorksheet, enhancerOk: enhanced.ok, enhancerError: enhanced.error, usedPrompt: finalPrompt }
    });
  } catch (e) {
    // Always return JSON (so frontend gets a proper response, not just "Failed to fetch")
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e),
    });
  }
}
