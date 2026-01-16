// api/playground.js
// Chat Playground for Robocoders Kit + image review (vision) via gpt-4o-mini

const CHAT_URL = "https://api.openai.com/v1/chat/completions";

// Single product we support for now (we can extend to multiple later)
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

function isDataUrlImage(s) {
  return typeof s === "string" && s.startsWith("data:image/");
}

/* ---------- Knowledge loader ---------- */

let lastKbDebug = "";

async function loadKnowledge() {
  const url = (process.env.KNOWLEDGE_URL || "").trim();
  lastKbDebug = `env.KNOWLEDGE_URL="${url}"`;

  if (!url) {
    lastKbDebug += " | no URL set";
    return null;
  }

  try {
    const res = await fetch(url, { cache: "no-store" });
    lastKbDebug += ` | fetchStatus=${res.status}`;

    if (!res.ok) {
      lastKbDebug += " | fetchNotOk";
      return null;
    }

    const txt = await res.text();
    lastKbDebug += ` | bodyLength=${txt.length}`;

    const parsed = JSON.parse(txt);
    lastKbDebug += " | jsonParsedOk";
    return parsed;
  } catch (e) {
    lastKbDebug += ` | error=${String(e)}`;
    return null;
  }
}

function buildKnowledgeContext(kb) {
  if (!kb || typeof kb !== "object") return "";

  const contents = Array.isArray(kb.contents)
    ? kb.contents.map((c) => ({
        name: c.name,
        type: c.type || "",
        description: c.description || ""
      }))
    : [];

  const projects = Array.isArray(kb.projects)
    ? kb.projects.map((p) => ({
        name: p.name,
        description: p.description || ""
      }))
    : [];

  const skills = Array.isArray(kb.skills) ? kb.skills : [];

  let ctx = "";

  if (contents.length) {
    ctx +=
      "KIT_CONTENTS_JSON = " +
      JSON.stringify(contents).slice(0, 6000) +
      "\n\n";
  }

  if (projects.length) {
    ctx +=
      "KIT_PROJECTS_JSON = " +
      JSON.stringify(projects).slice(0, 6000) +
      "\n\n";
  }

  if (skills.length) {
    ctx += "KIT_SKILLS = " + JSON.stringify(skills).slice(0, 4000) + "\n\n";
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

const { input = "", messages = [], attachment = null } = req.body || {};
if (!String(input).trim()) {
  return res.status(400).json({ message: "Empty input" });
}


    // Attachment support (image only for now)
    let att = null;
    if (attachment && typeof attachment === "object") {
      const { kind, dataUrl, name } = attachment;
      if (kind === "image" && isDataUrlImage(dataUrl)) {
        if (String(dataUrl).length > 4_500_000) {
          return res.status(400).json({ message: "Image too large" });
        }
        att = { kind: "image", dataUrl: String(dataUrl), name: String(name || "image") };
      }
    }

    const product = PRODUCT;

    const kb = await loadKnowledge();
    const kbContext = buildKnowledgeContext(kb);

    // DEBUG: type "__debug_kb__"
    if (String(input) === "__debug_kb__") {
      return res.status(200).json({
        text:
          "DEBUG KB V2\n" +
          `hasKb: ${!!kb}\n` +
          `contentsCount: ${Array.isArray(kb?.contents) ? kb.contents.length : 0}\n` +
          `firstContentName: ${kb?.contents?.[0]?.name || "NONE"}\n` +
          `lastKbDebug: ${lastKbDebug || "EMPTY"}\n\n` +
          "kbContextPreview:\n" +
          clampChars(kbContext, 400)
      });
    }

    const guards = (product?.behavior?.guardrails || []).join(" | ");

    const sys = `
You are the official Be Cre8v "Robocoders Kit" Assistant.

You have three important data tables:

1) KIT_CONTENTS_JSON  → exact list of kit components (each has "name", "type", "description")
2) KIT_PROJECTS_JSON  → exact list of project ideas included with this kit
3) KIT_SKILLS         → list of skills kids build with this kit

STRICT RULES:
- Never invent parts. ONLY use names from KIT_CONTENTS_JSON.
- Never invent projects. ONLY use names from KIT_PROJECTS_JSON.
- If a user asks what is inside the box, list the components exactly.
- If a user asks for starter projects, list the project names exactly.
- Use KIT_SKILLS if the user asks about skills.
- Stay kid-safe and Indian-English friendly.
- Always include a safety reminder for tools, electricity, motors, or heat.

IMAGE RULES (if an image is attached):
- First describe what you see in 1-2 lines.
- Then answer the user’s question.
- If the image shows wiring/tools/motors/electricity, remind adult supervision.

Guardrails:
${guards}

Here is your data:
${kbContext}
`.trim();

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

    const userMsg = att?.kind === "image"
      ? {
          role: "user",
          content: [
            { type: "text", text: String(input) },
            { type: "image_url", image_url: { url: att.dataUrl } }
          ]
        }
      : { role: "user", content: String(input) };

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
          userMsg
        ],
        max_tokens: 450,
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
