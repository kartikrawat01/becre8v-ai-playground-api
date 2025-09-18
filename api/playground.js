export default function handler(req, res) {
  // Minimal CORS so the browser can call it while we set up the real code later
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(204).end();

  return res.status(200).json({
    ok: true,
    message: 'Be Cre8v Playground API placeholder is live.',
    next: 'In Step 3B/3C we’ll add env vars + the real OpenAI proxy.'
  });
}
