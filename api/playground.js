// api/playground.js
// Simple Chat-style Playground for Robocoders Kit using structured knowledge

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

/* ---------- Knowledge loader & formatter ---------- */

function clampChars(s, max = 2000) {
  if (!s) return "";
  const str = String(s);
  return str.length > max ? str.slice(0, max) : str;
}

// Load JSON from KNOWLEDGE_URL and parse it
async function loadKnowledge() {
  const url = (process.env.KNOWLEDGE_URL || "").trim();
  if (!url) return null;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    const txt = await res.text();
    return JSON.parse(txt);
  } catch {
    return null;
  }
}

// Turn the JSON into a compact, friendly text summary
function buildKnowledgeContext(kb) {
  if (!kb || typeof kb !== "object") return "";

  const bullets = (label, arr) =>
    Array.isArray(arr) && arr.length
      ? `\n${label}:\n• ${arr.join("\n• ")}`
      : "";

  let sections = [];

  if (kb.tagline) {
    sections.push(`Tagline: ${clampChars(kb.tagline, 400)}`);
  }

  if (Array.isArray(kb.contents) && kb.contents.length) {
    const items = kb.contents
      .slice(0, 20)
      .map(c => `${c.name}: ${clampChars(c.description || "", 120)}`);
    sections.push(`What comes in the box (kit components):\n• ${items.join("\n• ")}`);
  }

  if (Array.isArray(kb.projects) && kb.projects.length) {
    const items = kb.projects
      .slice(0, 15)
      .map(p => `${p.name}: ${clampChars(p.description || "", 160)}`);
    sections.push(`Starter projects included in this kit:\n• ${items.join("\n• ")}`);
  }

  if (Array.isArray(kb.skills) && kb.skills.length) {
    const items = kb.skills.slice(0, 10).map(s => clampChars(s, 120));
    sections.push(`Skills kids practice with this kit:\n• ${items.join("\n• ")}`);
  }

  return clampChars(sections.join("\n\n"), 4000);
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

    // Load knowledge JSON and turn into structured text
    const kb = await loadKnowledge();
    const kbContext = buildKnowledgeContext(kb);

    const guards = (product?.behavior?.guardrails || []).join(" | ");

    const sys = `
You are the official Be Cre8v "Robocoders Kit" Assistant.

You have structured knowledge about this kit, summarised in the "Knowledge" section below.
This Knowledge is accurate and should be treated as the main source of truth.

VERY IMPORTANT INSTRUCTIONS:
- When the user asks "what comes in the box", "what is inside the kit", or similar:
  • Answer ONLY using the "What comes in the box (kit components)" list.
- When the user asks for "starter projects", "projects included", "what can I build", etc.:
  • Answer using the "Starter projects included in this kit" section.
- When the user asks what they will learn or what skills this kit builds:
  • Answer using the "Skills kids practice with this kit" section.
- You may combine these facts to give ideas, steps, troubleshooting, Python help, or safety tips.
- Do NOT invent extra hardware or features that are not consistent with the Knowledge text.
- If something is clearly not covered (for example, about a totally different kit), say:
  "This specific detail is not in my Robocoders data, but here is a safe general suggestion..."
- Always respond in kid-safe, simple, friendly language, as if talking to a child and their parent.
- Always mention adult supervision when tools, electricity, motors, or batteries are involved.

Here are the guardrails from Be Cre8v:
${guards}

Knowledge about the Robocoders Kit:
${kbContext}
`.trim();

    // Take last few messages from chat history
    const history = Array.isArray(messages)
      ? messages.slice(-8).map(m => ({
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
