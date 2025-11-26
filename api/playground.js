// api/playground.js
// Simple Chat-style Playground for Robocoders Kit using structured JSON knowledge

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

function clampChars(s, max = 2000) {
  if (!s) return "";
  const str = String(s);
  return str.length > max ? str.slice(0, max) : str;
}

/* ---------- Knowledge loader & formatter ---------- */

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

// Turn the JSON into small JSON-like blobs the model can use directly
function buildKnowledgeContext(kb) {
  if (!kb || typeof kb !== "object") return "";

  // Components in the box
  const contents = Array.isArray(kb.contents)
    ? kb.contents.map((c) => ({
        name: c.name,
        type: c.type || "",
        description: c.description || ""
      }))
    : [];

  // Project names only (descriptions already live in model if needed)
  const projects = Array.isArray(kb.projects)
    ? kb.projects.map((p) => ({
        name: p.name,
        description: p.description || ""
      }))
    : [];

  // Skills – keep as simple strings
  const skills = Array.isArray(kb.skills) ? kb.skills : [];

  // We encode them as JSON-in-text so the model can "read" them like tables.
  let ctx = "";

  if (contents.length) {
    const shortContents = contents.slice(0, 60); // plenty for one kit
    ctx +=
      "KIT_CONTENTS_JSON = " +
      JSON.stringify(shortContents).slice(0, 6000) +
      "\n\n";
  }

  if (projects.length) {
    const shortProjects = projects.slice(0, 80);
    ctx +=
      "KIT_PROJECTS_JSON = " +
      JSON.stringify(shortProjects).slice(0, 6000) +
      "\n\n";
  }

  if (skills.length) {
    const shortSkills = skills.slice(0, 40);
    ctx +=
      "KIT_SKILLS = " +
      JSON.stringify(shortSkills).slice(0, 4000) +
      "\n\n";
  }

  return ctx.trim();
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

You have three important data tables:

1) KIT_CONTENTS_JSON  → exact list of kit components (each has "name", "type", "description")
2) KIT_PROJECTS_JSON  → exact list of project ideas included with this kit (each has "name", "description")
3) KIT_SKILLS         → list of skills kids build with this kit

These JSON tables are the SINGLE source of truth about the kit.

VERY STRICT RULES (follow exactly):
- When the user asks "what comes in the box", "components", "what is inside the kit", etc.:
  • Read KIT_CONTENTS_JSON and list the "name" values EXACTLY as written there (Robocoders Brain, USB Cable, DC Motor, LED Module, etc.).
  • You may optionally give a SHORT explanation using the "description" for each, but do not rename items.
  • Do NOT invent extra parts like "chassis", "wheels", "sensors pack" unless they appear in KIT_CONTENTS_JSON.

- When the user asks for "starter projects", "projects included", "what can I build", etc.:
  • Read KIT_PROJECTS_JSON and give project names EXACTLY as in the "name" field (Hello World, Smart Box, Mood Lamp, Coin Collector, etc.).
  • You may summarise descriptions, but keep the official names.

- When the user asks what they will "learn", "skills", "outcomes":
  • Use KIT_SKILLS to answer.

- For troubleshooting, ideas, Python help, or step-by-step guidance:
  • Base your answers on these same components and projects.
  • Never talk about hardware or features that clearly do not exist in these tables.

- If a user asks about something completely outside these tables (for example: a different kit, or a component that is missing):
  • Clearly say that it is not listed in your Robocoders data and then give a generic but safe suggestion.

- Always respond in kid-safe, simple, friendly language for ages 8–14.
- Always include a gentle safety reminder when using tools, electricity, motors, or batteries.

Guardrails from Be Cre8v:
${guards}

Here is your data:
${kbContext}
`.trim();

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
        temperature: 0.4
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
