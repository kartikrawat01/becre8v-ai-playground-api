// api/generate-image.js
// Be Cre8v AI Playground — Image Generation (Planner-enhanced, kid-safe)

const OPENAI_CHAT_URL  = "https://api.openai.com/v1/chat/completions";
const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";

/* -------------------- CORS -------------------- */
function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
}

function allow(res, origin) {
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (!origin) return false;
  const list = origins();
  if (!list.length || list.some(a => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

/* -------------------- HELPERS -------------------- */
function looksLikeWorksheet(prompt) {
  const p = String(prompt || "").toLowerCase();
  return [
    "worksheet","printable","sheet","activity","quiz","mcq",
    "critical thinking","match","timeline","cause","effect",
    "compare","classroom","homework"
  ].some(k => p.includes(k));
}

/* -------------------- PLANNER PROMPT -------------------- */
const PLANNER_SYSTEM_PROMPT = `
You are Be Cre8v AI Prompt Planner.

Your job is to convert a raw user request into the BEST possible prompt
for an image generation model.

Rules:
- Do NOT explain anything
- Do NOT add commentary
- Output ONLY the final image prompt
- Decide intent first (worksheet, poster, illustration, diagram, free image)
- Compress long educational instructions into visual layout guidance
- Prefer clean structure over long text
- Always keep content kid-safe and classroom-safe
- Optimize for clarity, balance, printability, and aesthetics
`;

/* -------------------- PLANNER CALL -------------------- */
async function planImagePrompt(userPrompt, apiKey) {
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
        { role: "system", content: PLANNER_SYSTEM_PROMPT },
        { role: "user", content: userPrompt }
      ]
    })
  });

  if (!r.ok) {
    const t = await r.text();
    throw new Error("Planner error: " + t.slice(0, 300));
  }

  const data = await r.json();
  return data.choices[0].message.content.trim();
}

/* -------------------- HANDLER -------------------- */
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

    /* ---- STEP 1: PLAN PROMPT ---- */
    const plannedPrompt = await planImagePrompt(prompt, apiKey);

    /* ---- STEP 2: IMAGE GENERATION ---- */
    const r = await fetch(OPENAI_IMAGE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-image-1",
        prompt: plannedPrompt,
        size: isWorksheet ? "1024x1536" : "1024x1024",
        n: 1
      })
    });

    if (!r.ok) {
      const err = await r.text();
      return res.status(r.status).json({
        message: "Image API error",
        details: err.slice(0, 1000)
      });
    }

    const data = await r.json();
    const b64 = data?.data?.[0]?.b64_json;
    if (!b64) {
      return res.status(500).json({ message: "No image returned" });
    }

    const mime = "image/webp";
    const image = `data:${mime};base64,${b64}`;

    return res.status(200).json({ mime, image });

  } catch (e) {
    return res.status(500).json({
      message: "Server error",
      details: String(e?.message || e)
    });
  }
}
