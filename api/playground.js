// api/playground.js
// Be Cre8v AI Playground — CORS-safe proxy with kid-safe prompts

const CHAT_URL = "https://api.openai.com/v1/chat/completions";

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

/* ---------- Handler ---------- */
export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // Preflight
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

    const { productId, actionId, input = "" } = req.body || {};
    if (!productId || !actionId) {
      return res.status(400).json({ message: "Missing productId or actionId" });
    }

    // Load product definitions (env var with safe fallback)
    const PRODUCTS_URL =
      (process.env.PRODUCTS_URL || "").trim() ||
      "https://cdn.shopify.com/s/files/1/0797/0576/8213/files/products.json?v=1758262885";

    const cfgResp = await fetch(PRODUCTS_URL, { cache: "no-store" });
    if (!cfgResp.ok) {
      const txt = await cfgResp.text().catch(() => "");
      return res.status(500).json({ message: "Failed to load products.json", details: txt.slice(0, 400) });
    }
    const products = await cfgResp.json();

    const product = products?.[productId];
    if (!product) return res.status(400).json({ message: "Unknown productId" });

    const action = (product.actions || []).find(a => a.id === actionId);
    if (!action) return res.status(400).json({ message: "Unknown actionId" });

    // Kid-safe prompt
    const tone = product?.behavior?.tone || "kid-safe, friendly, step-by-step";
    const guards = (product?.behavior?.guardrails || []).join(" | ");
    const sys = `You are Be Cre8v Kid-Safe Assistant for "${product.name}".
Rules: ${guards}. No PII. No adult/violent themes. Keep replies short and encouraging.`;
    const usr = `${action.label} for "${product.name}". Idea: ${input || "Surprise me."}`;

    // Ensure API key exists and is trimmed
    const OPENAI_KEY = String(process.env.OPENAI_API_KEY || "").trim();
    if (!OPENAI_KEY) {
      return res.status(500).json({ message: "Server config error", details: "OPENAI_API_KEY env var is empty" });
    }

    // Call OpenAI (Chat Completions)
    const oai = await fetch(CHAT_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",            // swap to a model you have access to if needed
        messages: [
          { role: "system", content: `${tone}\n${sys}` },
          { role: "user", content: usr }
        ],
        max_tokens: 300,
        temperature: ["draw_card","style_card","twist_card","suggest_idea"].includes(actionId) ? 0.7 : 0.4
      })
    });

    if (!oai.ok) {
      const err = await oai.text().catch(() => "");
      return res.status(oai.status).json({ message: "OpenAI error", details: err.slice(0, 800) });
    }

    const data = await oai.json();
    const text = (data.choices?.[0]?.message?.content || "").trim() || "No response.";
    return res.status(200).json({ text });

  } catch (e) {
    return res.status(500).json({ message: "Server error", details: String(e?.message || e) });
  }
}
