// api/playground.js
// Simple Chat-style Playground for Robocoders Kit using raw JSON knowledge

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
    .map((s) => s.trim())
    .filter(Boolean);
}
function allow(res, origin) {
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

/* ---------- Knowledge loader (RAW JSON TEXT) ---------- */
async function loadKnowledgeText() {
  const url = (process.env.KNOWLEDGE_URL || "").trim();
  if (!url) return "";
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return "";
    // we keep it as TEXT so the full JSON is visible to the model
    const text = await res.text();
    return text || "";
  } catch {
    return "";
  }
}

function clampChars(s, max = 2000) {
  if (!s) return "";
  return s.length > max ? s.slice(0, max) : s;
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

    // Load raw JSON knowledge text from env URL
    const kbRaw = await loadKnowledgeText();
    // Optional: clamp if file is huge
    const kbContext = clampChars(kbRaw, 8000);

    const guards = (product?.behavior?.guardrails || []).join(" | ");

    const sys = `
You are the official Be Cre8v "Robocoders Kit" Assistant.

VERY IMPORTANT RULES (follow strictly):
1. You MUST treat the following JSON as the single source of truth about the Robocoders Kit.
2. When answering about components, what is inside the box, projects, skills, steps, wiring, or safety:
   - FIRST carefully read the JSON.
   - THEN answer ONLY using details that are consistent with the JSON.
3. Do NOT invent extra parts, sensors, boards, or features that are NOT present in the JSON.
4. If the user asks for something that is not described in the JSON, say clearly:
   "This part is not listed in the Robocoders Kit data. Here is a safe general suggestion..."
5. ALWAYS answer in kid-safe, friendly, simple language as if talking to a child and their parent.
6. ALWAYS add a gentle safety reminder when tools, electricity, motors, or batteries are involved.
7. You may help with ideas, steps, troubleshooting, Python help, and safety, but all related to this kit.

Here are your guardrails from Be Cre8v:
${guards}

Below is the JSON data (Knowledge) for the Robocoders Kit.
You MUST read this carefully and base your answers on it:

JSON_KNOWLEDGE_START
${kbContext}
JSON_KNOWLEDGE_END
`;

    // Take last few messages from chat history
    const history = Array.isArray(messages)
      ? messages.slice(-8).map((m) => ({
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
