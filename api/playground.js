// api/playground.js — CORS-safe proxy for Be Cre8v Playground

const OAI_URL = "https://api.openai.com/v1/chat/completions";

// Allow multiple origins via comma-separated env var
function isAllowedOrigin(origin) {
  const list = (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
  if (!list.length) return true; // allow all if unset (not recommended)
  return !!(origin && list.some(a => origin.startsWith(a)));
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  // reflect only if allowed; otherwise do not set ACAO (browser will block)
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // Preflight
  if (req.method === "OPTIONS") {
    if (isAllowedOrigin(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
    }
    setCors(res);
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    if (isAllowedOrigin(origin)) res.setHeader("Access-Control-Allow-Origin", origin);
    setCors(res);
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    if (!isAllowedOrigin(origin)) {
      // No ACAO header -> browser blocks. Return 403 for clarity.
      return res.status(403).json({ message: "Forbidden origin" });
    }
    res.setHeader("Access-Control-Allow-Origin", origin);
    setCors(res);

    const { productId, actionId, input = "" } = req.body || {};
    if (!productId || !actionId) {
      return res.status(400).json({ message: "Missing productId or actionId" });
    }

    // Load products config from Shopify Files CDN
    const cfgResp = await fetch(process.env.PRODUCTS_URL, { cache: "no-store" });
    if (!cfgResp.ok) return res.status(500).json({ message: "Failed to load products.json" });
    const products = await cfgResp.json();

    const product = products?.[productId];
    if (!product) return res.status(400).json({ message: "Unknown productId" });

    const action = (product.actions || []).find(a => a.id === actionId);
    if (!action) return res.status(400).json({ message: "Unknown actionId" });

    // Build a compact, kid-safe prompt
    const systemTone =
      product?.behavior?.tone ||
      "You are a helpful, kid-safe assistant. Keep it short, friendly, and age-appropriate.";
    const guardrails = (product?.behavior?.guardrails || []).join(" | ");

    const messages = [
      { role: "system", content: `${systemTone}\nRules: ${guardrails}\nNo PII. Avoid adult/violent themes.` },
      { role: "user", content: `${action.label} for "${product.name}". Idea: ${input || "Surprise me."}` }
    ];

    // Call OpenAI
    const oai = await fetch(OAI_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages,
        max_tokens: 200,
        temperature: ["draw_card","style_card","twist_card","suggest_idea"].includes(actionId) ? 0.7 : 0.4
      })
    });

    if (!oai.ok) {
      const errText = await oai.text().catch(() => "");
      return res.status(oai.status).json({ message: "OpenAI error", details: errText.slice(0, 400) });
    }

    const data = await oai.json();
    const text = (data.choices?.[0]?.message?.content || "").trim() || "No response.";
    return res.status(200).json({ text });
  } catch (e) {
    return res.status(500).json({ message: "Server error", details: String(e?.message || e) });
  }
}
