// api/playground.js
import fetch from "node-fetch";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ message: "Method not allowed" });
  }

  try {
    const { productId, actionId, input } = req.body;

    // Load products.json from the CDN (your env var)
    const productsRes = await fetch(process.env.PRODUCTS_URL);
    const productsData = await productsRes.json();
    const product = productsData[productId];

    if (!product) {
      return res.status(400).json({ message: "Unknown productId" });
    }

    const action = (product.actions || []).find(a => a.id === actionId);
    if (!action) {
      return res.status(400).json({ message: "Unknown actionId" });
    }

    // Build kid-safe prompt
    const messages = [
      { role: "system", content: product.behavior?.tone || "You are a helpful kid-safe assistant." },
      { role: "user", content: `${action.label}: ${input || ""}` }
    ];

    // Call OpenAI
    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",  // lightweight, safe
        messages,
        max_tokens: 120
      })
    });

    if (!openaiRes.ok) {
      const errText = await openaiRes.text();
      return res.status(500).json({ message: "OpenAI error", details: errText });
    }

    const data = await openaiRes.json();
    const text = data.choices?.[0]?.message?.content?.trim();

    return res.status(200).json({ text });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "Server error", error: err.message });
  }
}
