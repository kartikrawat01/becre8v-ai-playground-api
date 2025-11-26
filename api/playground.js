// api/playground.js
// Simple Chat-style Playground for Robocoders Kit

const CHAT_URL = "https://api.openai.com/v1/chat/completions";

// Single product we support for now
const PRODUCT = {
  name: "Robocoders Kit",
  behavior: {
    tone: "kid-safe, friendly, step-by-step",
    guardrails: [
      "Always keep answers age-appropriate for school kids.",
      "Always mention adult supervision when using tools, electricity, sharp objects or heat.",
      "Never give dangerous, illegal, or irreversible instructions.",
      "No personal data collection. No adult or violent themes."
    ]
  }
};

/* ---------- CORS helpers ---------- */
function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
}
function allow(res, origin) {
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

/* ---------- Knowledge helpers ---------- */
async function fetchJsonOrText(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    const ct = (r.headers.get("content-type") || "").toLowerCase();
    if (ct.includes("application/json")) return await r.json();
    const text = await r.text();
    return { __markdown: true, text };
  } catch {
    return null;
  }
}
function take(arr, n = 6) {
  return Array.isArray(arr) ? arr.slice(0, n) : [];
}
function clampChars(s, max = 2000) {
  if (!s) return "";
  return s.length > max ? s.slice(0, max) : s;
}

function selectContextFromKB(kb) {
  if (!kb) return "";
  if (kb.__markdown) return clampChars(kb.text, 1200);

  const bullets = (label, arr, n = 6) =>
    arr && arr.length ? `\n${label}:\n• ${take(arr, n).join("\n• ")}` : "";

  let out = "";
  if (kb.tagline) out += `Tagline: ${kb.tagline}\n`;
  if (Array.isArray(kb.tone)) {
    const toneLines = take(kb.tone.map(t => `${t.rule}: ${t.description}`), 5);
    out += bullets("Tone", toneLines, 5);
  }

  out += bullets(
    "What is in the box",
    kb?.contents?.map(c => `${c.name}: ${c.description}`) || [],
    8
  );
  out += bullets(
    "Starter Projects",
    kb?.projects?.map(p => `${p.name}: ${p.description}`) || [],
    6
  );
  out += bullets("Skills", kb?.skills || [], 6);

  return clampChars(out, 1500);
}

async function gatherKnowledge(knowledgeUrls = []) {
  if (!knowledgeUrls.length) return "";
  const items = await Promise.all(knowledgeUrls.map(fetchJsonOrText));
  const parts = items
    .filter(Boolean)
    .map(kb => selectContextFromKB(kb))
    .filter(Boolean);
  return clampChars(parts.join("\n\n"), 2000);
}

/* ---------- Handler ---------- */
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

  try {
    if (!allow(res, origin)) {
      return res.status(403).json({ message: "Forbidden origin" });
    }

    const { input = "", messages = [] } = req.body || {};
    if (!input.trim()) {
      return res.status(400).json({ message: "Empty input" });
    }

    const product = PRODUCT;

    // Build knowledge URLs from env
    const knowledgeUrls = [];
    const envUrl = (process.env.KNOWLEDGE_URL || "").trim();
    if (envUrl) knowledgeUrls.push(envUrl);

    const kbContext = await gatherKnowledge(knowledgeUrls);

    const guards = (product?.behavior?.guardrails || []).join(" | ");
    const tone =
      product?.behavior?.tone || "kid-safe, friendly, step-by-step";

    const sys = `You are Be Cre8v Kid-Safe Assistant for the product "${product.name}".
Rules: ${guards}.
Always answer as if you are helping a child and their parent build and use this kit.
Keep replies short, encouraging, and very practical.
If the user asks for ideas, steps, troubleshooting, Python help, or safety tips, you can answer all of that naturally in chat form.
Use the facts below about the kit only when relevant:
${kbContext}`;

    // Take last few messages from chat history
    const history = Array.isArray(messages)
      ? messages
          .slice(-8)
          .map(m => ({
            role: m.role === "assistant" ? "assistant" : "user",
            content: clampChars(String(m.content || ""), 600)
          }))
      : [];

    const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "").trim();
    if (!OPENAI_KEY) {
      return res.status(500).json({
        message: "Server config error",
        details: "OPENAI_API_KEY env var is empty"
      });
    }

    const oai = await fetch(CHAT_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: sys },
          ...history,
          { role: "user", content: input }
        ],
        max_tokens: 400,
        temperature: 0.5
      })
    });

    if (!oai.ok) {
      const err = await oai.text().catch(() => "");
      return res
        .status(oai.status)
        .json({ message: "OpenAI error", details: err.slice(0, 800) });
    }

    const data = await oai.json();
    const text =
      (data.choices?.[0]?.message?.content || "").trim() || "No response.";
    return res.status(200).json({ text });
  } catch (e) {
    return res
      .status(500)
      .json({ message: "Server error", details: String(e?.message || e) });
  }
}
