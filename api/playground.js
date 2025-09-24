// api/playground.js
// Be Cre8v AI Playground — kid-safe proxy with knowledge files (JSON/Markdown)
// Uses Chat Completions for wide availability

const CHAT_URL = "https://api.openai.com/v1/chat/completions";

/* ---------- CORS helpers ---------- */
function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",").map(s => s.trim()).filter(Boolean);
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
    const text = await r.text(); // allow Markdown / plain text
    return { __markdown: true, text };
  } catch { return null; }
}
function take(arr, n = 6) { return Array.isArray(arr) ? arr.slice(0, n) : []; }
function clampChars(s, max = 2000) { if (!s) return ""; return s.length > max ? s.slice(0, max) : s; }

function selectContextFromKB(kb, actionId) {
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

  if (actionId === "troubleshoot") {
    out += bullets("Safety", kb.safety || [], 6);
    out += bullets("Components", kb?.contents?.map(c => `${c.name}: ${c.description}`) || [], 6);
  } else if (actionId === "python_help") {
    const python = (kb.software || []).find(s => (s.platform || "").toLowerCase().includes("python"));
    if (python) {
      out += `\nPython Setup (short): ${python.setup_short || ""}`;
      out += `\nPython Setup (detailed): ${python.setup_detailed ? clampChars(python.setup_detailed, 600) : ""}`;
    }
    out += bullets("Hardware Basics", kb?.contents?.map(c => `${c.name} — ${c.type}`) || [], 6);
  } else if (actionId === "explain_steps") {
    out += bullets("What’s in the box", kb?.contents?.map(c => `${c.name}: ${c.description}`) || [], 8);
    out += bullets("Starter Projects", kb?.projects?.map(p => `${p.name}: ${p.description}`) || [], 6);
  } else {
    // suggest_idea, draw/style/twist, etc.
    out += bullets("Project Ideas", kb?.projects?.map(p => p.name) || [], 10);
    out += bullets("Skills", kb?.skills || [], 6);
  }

  return clampChars(out, 1500);
}

async function gatherKnowledge(knowledgeUrls = [], actionId) {
  if (!knowledgeUrls.length) return "";
  const items = await Promise.all(knowledgeUrls.map(fetchJsonOrText));
  const parts = items
    .filter(Boolean)
    .map(kb => selectContextFromKB(kb, actionId))
    .filter(Boolean);
  return clampChars(parts.join("\n\n"), 2000);
}

/* ---------- Handler ---------- */
export default async function handler(req, res) {
  const origin = req.headers.origin || "";
  if (req.method === "OPTIONS") { allow(res, origin); return res.status(204).end(); }
  if (req.method !== "POST") { allow(res, origin); return res.status(405).json({ message: "Method not allowed" }); }

  try {
    if (!allow(res, origin)) return res.status(403).json({ message: "Forbidden origin" });

    const { productId, actionId, input = "", cardId = null } = req.body || {};
    if (!productId || !actionId) return res.status(400).json({ message: "Missing productId or actionId" });

    // Load product catalog
    const PRODUCTS_URL =
      (process.env.PRODUCTS_URL || "").trim() ||
      "https://cdn.shopify.com/s/files/1/0797/0576/8213/files/products.json";
    const cfg = await fetch(PRODUCTS_URL, { cache: "no-store" });
    if (!cfg.ok) return res.status(500).json({ message: "Failed to load products.json" });
    const products = await cfg.json();

    const product = products?.[productId];
    if (!product) return res.status(400).json({ message: "Unknown productId" });

    // Gather knowledge from external URLs you set in products.json
    const kbContext = await gatherKnowledge(product.knowledge_urls || [], actionId);

    const guards = (product?.behavior?.guardrails || []).join(" | ");
    const tone = product?.behavior?.tone || "kid-safe, friendly, step-by-step";

    // Build prompts
    const sys = `You are Be Cre8v Kid-Safe Assistant for "${product.name}".
Rules: ${guards}. No PII. No adult/violent themes. Keep replies short and encouraging.
Use the facts below only if they are relevant to the user's request.
Knowledge:
${kbContext}`;

    const usr = buildUserTurn(product, actionId, input);

    // Ensure API key exists and is trimmed
    const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "").trim();
    if (!OPENAI_KEY) {
      return res.status(500).json({ message: "Server config error", details: "OPENAI_API_KEY env var is empty" });
    }

    // Call OpenAI (Chat Completions)
    const oai = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Authorization": `Bearer ${OPENAI_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gpt-4o-mini", // if you see model_not_found, try "gpt-4o-mini-2024-07-18" or "gpt-4o"
        messages: [
          { role: "system", content: `${tone}\n${sys}` },
          { role: "user",   content: usr }
        ],
        max_tokens: 300,
        temperature: ["draw_card","style_card","twist_card","suggest_idea"].includes(actionId) ? 0.7 : 0.4
      })
    });

    if (!oai.ok) {
      const err = await oai.text().catch(()=> "");
      return res.status(oai.status).json({ message: "OpenAI error", details: err.slice(0, 800) });
    }

    const data = await oai.json();
    const text = (data.choices?.[0]?.message?.content || "").trim() || "No response.";
    return res.status(200).json({ text });

  } catch (e) {
    return res.status(500).json({ message: "Server error", details: String(e?.message || e) });
  }
}

/* ---------- user-route helper ---------- */
function buildUserTurn(p, actionId, input) {
  const routes = {
    suggest_idea : `Suggest 5 creative ideas for "${p.name}". Use 1 line each in kid-safe language. User idea: ${input || 'Surprise me.'}`,
    troubleshoot : `Troubleshoot likely issues with "${p.name}". Short bullets. Ask 1 clarifying question at the end.`,
    python_help  : `Give a tiny Python snippet related to "${p.name}", then a 2-line explanation kids can follow.`,
    explain_steps: `Explain 5 simple build steps for "${p.name}" with tick-style bullets. End with 1 safety nudge.`,
    safety_tips  : `Provide 5 kid-friendly safety tips for using "${p.name}". Mention adult supervision.`,
    draw_card    : `Output a fun drawing prompt for kids: a vivid scene with 1 quirky detail. End with "Now draw it!"\nIdea: ${input || ''}`,
    style_card   : `List 3 art/story styles (1 line each) and why kids will like each.`,
    twist_card   : `Give 3 playful twist ideas (what-if changes) in 1 line each.`
  };
  return routes[actionId] || `Help with "${p.name}". Idea: ${input || 'Make it fun and safe.'}\nInclude a gentle safety nudge.`;
}
