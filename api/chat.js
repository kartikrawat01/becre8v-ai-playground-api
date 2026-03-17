// =============================================================================
// Be Cre8v AI Backend — Multi-Product RAG (Robocoders + Spin Genius)
// =============================================================================
const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings";
const EMBED_MODEL = "text-embedding-3-small";

function origins() {
  return (process.env.ALLOWED_ORIGIN || "").split(",").map((s) => s.trim()).filter(Boolean);
}

function allow(res, origin) {
  res.setHeader("Vary", "Origin");
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

const CHAT_PLANNER_PROMPT = `
You are Be Cre8v AI Conversation Planner.
Rewrite the user's message into a clearer, more intelligent version BEFORE it is answered.
Rules:
- Do NOT answer the user.
- Output ONLY the rewritten prompt text.
- Preserve intent and key details.
- Make it easier to answer with steps, structure, and the right questions.
- Keep child-friendly, encouraging tone.
- If the user asks something product-specific but doesn't specify the project/module name, include a short clarification question in the rewritten prompt.
- Do not add adult/unsafe content.
`.trim();

async function planChatPrompt(userText, apiKey) {
  const r = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [
        { role: "system", content: CHAT_PLANNER_PROMPT },
        { role: "user", content: String(userText || "").trim() },
      ],
    }),
  });
  if (!r.ok) { const t = await r.text().catch(() => ""); throw new Error("Planner error: " + t.slice(0, 800)); }
  const data = await r.json();
  return data?.choices?.[0]?.message?.content?.trim() || String(userText || "").trim();
}

async function embedText(text, apiKey) {
  const r = await fetch(OPENAI_EMBED_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: String(text || "").trim() }),
  });
  if (!r.ok) { const t = await r.text().catch(() => ""); throw new Error("Embedding error: " + t.slice(0, 800)); }
  const data = await r.json();
  return data.data[0].embedding;
}

async function queryPinecone(queryEmbedding, namespace, topK = 6) {
  const host = process.env.PINECONE_INDEX_HOST;
  const apiKey = process.env.PINECONE_API_KEY;
  if (!host || !apiKey) throw new Error("PINECONE_INDEX_HOST or PINECONE_API_KEY not set in env.");
  const r = await fetch(`${host}/query`, {
    method: "POST",
    headers: { "Api-Key": apiKey, "Content-Type": "application/json" },
    body: JSON.stringify({ vector: queryEmbedding, topK, includeMetadata: true, namespace }),
  });
  if (!r.ok) { const t = await r.text().catch(() => ""); throw new Error("Pinecone query error: " + t.slice(0, 800)); }
  const data = await r.json();
  return (data.matches || []).map((m) => ({
    text: m.metadata?.text || "",
    type: m.metadata?.type || "general",
    projectName: m.metadata?.projectName || null,
    patternImage: m.metadata?.patternImage || null,
    boardPosition: m.metadata?.boardPosition || null,
    stickPosition: m.metadata?.stickPosition || null,
    patternName: m.metadata?.patternName || null,
    score: m.score,
  }));
}

// =============================================================================
// IMAGE PREPROCESSING — Convert to grayscale before sending to GPT
//
// ROOT CAUSE OF COLOR MISIDENTIFICATION:
// GPT-4o-mini uses color as a visual shortcut even when told not to in the prompt.
// Prompt instructions alone cannot fully override the model's internal vision
// processing which detects color automatically.
//
// THE FIX — Strip color BEFORE the API call:
// By converting the image to grayscale using sharp, we physically eliminate all
// color data. The model literally cannot see color because it does not exist in
// the image. This permanently solves:
//   Problem 1: Same pattern, different pen color → now correctly identified
//   Problem 2: Two different patterns in same color → now distinguished by shape
//
// .normalise() boosts contrast so line structure is sharper and easier to analyse.
// Falls back to original image if sharp fails — handler never crashes.
// =============================================================================
async function toGrayscaleBase64(imageData) {
  try {
    const sharp = require("sharp");
    const base64 = imageData.startsWith("data:") ? imageData.split(",")[1] : imageData;
    const inputBuffer = Buffer.from(base64, "base64");
    const outputBuffer = await sharp(inputBuffer)
      .grayscale()
      .normalise()
      .jpeg({ quality: 90 })
      .toBuffer();
    return "data:image/jpeg;base64," + outputBuffer.toString("base64");
  } catch (err) {
    console.warn("Grayscale conversion failed, using original:", err.message);
    return imageData.startsWith("data:") ? imageData : `data:image/png;base64,${imageData}`;
  }
}

// =============================================================================
// Pattern name lookup table — ALL 50 patterns, shape-based names ONLY
// Rules:
//   1. NO color words (no golden, pink, green, dark, black etc.)
//   2. Specific multi-word phrases listed before short single words
//   3. 12-J listed BEFORE 10-A so "net","grid","mesh" match garden net first
//   4. findImageByPatternName sorts names by length desc so longer phrases win
//   5. For duplicate board positions (e.g. 10-Q, 9-B), both stickPositions map
//      to the same image filename — image naming is handled by the knowledge base
// =============================================================================
const PATTERN_NAME_MAP = [
  // ── RING / CHAIN PATTERNS ──────────────────────────────────────────────────
  { image: "3-D.jpeg",  names: ["magic bubble chain","bubble ring chain","bubble chain","chain of linked loops","chain of rings","linked ring chain","loopy chain","bubble ring","ring chain"] },
  { image: "1-E.jpeg",  names: ["soapy bubble garland","floating bubble ring","bubble garland","soapy ring","ring of circles","bubble circles","garland of bubbles"] },
  { image: "2-F.jpeg",  names: ["soap bubble ring","bubble dream ring","faint bubble ring","tiny soap bubbles","gentle bubble ring","quiet bubble ring","dream bubble","soap bubbles floating"] },
  { image: "9-O.jpeg",  names: ["bird nest ring","cozy bird nest","delicate circle ring","nest of circles","dancing circles ring","thin loop ring","bird nest"] },

  // ── NET / GRID PATTERNS — listed before 10-A so "net","grid","mesh" match first ──
  { image: "12-J.jpeg", names: ["magic garden fence","secret garden net","circular grid","circular net","circular mesh","woven circle","garden fence","garden net","fishing net","grid pattern","net pattern","mesh pattern","lattice pattern","woven fence","square grid","grid","net","lattice","mesh","woven"] },

  // ── CROWN / LACE PATTERNS ──────────────────────────────────────────────────
  { image: "6-N.jpeg",  names: ["royal crown lace","princess lace crown","lace crown","lace border","open center crown","fancy lace pattern","crown pattern","lace pattern","crown","lace"] },
  { image: "4-N.jpeg",  names: ["fancy lace tablecloth","lace tablecloth","lace ring","wavy lace","tablecloth lace","lace weave","fancy lace ring","wavy loop ring","tablecloth"] },
  { image: "6-Q.jpeg",  names: ["bunny ear ring","bunny ears circle","tall thin loops","upright loop ring","tall loop ring","bunny ears","tall loops circle","bunny ear"] },
  { image: "4-H.jpeg",  names: ["lacy flower ring","lacy donut ring","lacy ring","lace flower ring","tiny loops ring","hundreds of loops","fancy donut ring","lacy circle","lace donut"] },

  // ── FLOWER PATTERNS ────────────────────────────────────────────────────────
  { image: "7-J.jpeg",  names: ["happy little flower","petal flower","bouncy flower","overlapping petals","loopy petals","rosette pattern","loopy flower","flower pattern","floral pattern","small flower","compact flower","flower","rosette","petals"] },
  { image: "1-P.jpeg",  names: ["blooming rose","bold rose","large petal flower","big petal flower","rose pattern","overlapping rose petals","rose bloom","bold flower"] },
  { image: "14-I.jpeg", names: ["blooming sunburst","garden sunburst","soft petal sunburst","symmetrical sunburst","blooming garden flower","spinning flower","sunburst flower","garden flower"] },
  { image: "7-C.jpeg",  names: ["fairy stamp flower","eight petal flower","lucky flower","eight petals","eight round petals","stamp flower","clean flower stamp","simple eight petal"] },
  { image: "2-C.jpeg",  names: ["crystal ball flower","secret crystal garden","magical crystal flower","crystal garden","flower inside crystal","tiny garden flower","crystal ball","hidden flower","secret garden inside"] },
  { image: "3-L.jpeg",  names: ["birthday cake swirl","swirling petals","cake decoration swirl","swirling flower","curving petals","calm whirlpool petals","fancy petal swirl","swirl petals"] },
  { image: "3-C.jpeg",  names: ["geometric puzzle bloom","puzzle flower","neat geometric flower","round puzzle loops","geometric bloom","puzzle loops flower","tidy bloom","puzzle pieces flower"] },
  { image: "14-H.jpeg", names: ["dandelion dream","dandelion petals","skinny dandelion loops","long petal dandelion","dandelion flower","long skinny petals","morning sun petals","dandelion"] },
  { image: "9-M.jpeg",  names: ["spinning hula hoop","vibrant hula hoop","hula hoop ring","energetic ring","bouncy vibrant ring","hula ring","vibrant ring","hula hoop"] },

  // ── SPIDERWEB / MANDALA ─────────────────────────────────────────────────────
  { image: "10-A.jpeg", names: ["shining spiderweb","spiderweb mandala","spiral web like structure","spiral web like","spiral web","web like structure","web-like structure","radiating lines","diamond gaps","diamond shapes","starburst pattern","mandala pattern","spoke pattern","diamond web","web pattern","spider web","spiderweb","web structure","starburst","mandala"] },
  { image: "7-B.jpeg",  names: ["artist spider star","spiderweb star","large loop star","loops crossing middle","spiderweb loops","large crossing loops","artist spider","simple spiderweb star"] },
  { image: "11-A.jpeg", names: ["royal thread crown","crown of light","thread crown","crossing lines circle","glowing ring crown","shiny thread ring","straight crossing circle","crown of threads"] },

  // ── DENSE / SOLID RING ──────────────────────────────────────────────────────
  { image: "3-R.jpeg",  names: ["spinning galaxy swirl","dense whirlpool","fuzzy donut","dense donut","packed spiral","solid ring","solid donut","tightly packed lines","thick donut","thick ring","whirlpool pattern","donut pattern","whirlpool","galaxy swirl","vortex","swirl"] },
  { image: "11-L.jpeg", names: ["giant bold donut","giant donut","solid thick ring","bold donut ring","thick ring with hole","big donut","bold ring","strong donut","giant ring"] },
  { image: "15-A.jpeg", names: ["spinning scribble wheel","fuzzy tire","scribble wheel","messy circle tire","tire scribbles","dense messy circle","fuzzy wheel","spinning wheel scribble"] },
  { image: "G-6.jpeg",  names: ["glowing magic coin","spinning badge","magic coin","glowing badge","thread sun","squeezed center badge","glowing coin","tiny spinning sun","badge coin"] },
  { image: "16-O.jpeg", names: ["deep sea shell","spiraling seashell","seashell tunnel","deep tunnel spiral","busy seashell","spiral seashell","tunnel seashell","deep spiral","seashell"] },
  { image: "12-C.jpeg", names: ["happy glowing sun","glowing sun","solid sun center","rays all directions","packed sun","dense sun pattern","glowing rays","solid bright sun"] },

  // ── STAR PATTERNS ──────────────────────────────────────────────────────────
  { image: "8-A.jpeg",  names: ["magic thread snowflake","magic snowflake star","thin star snowflake","delicate snowflake star","snowflake star","magic thread star","thin star","magic snowflake"] },
  { image: "10-R.jpeg", names: ["magic castle window","castle star window","star window castle","steady star castle","straight line star","strong star","castle window","balanced star"] },
  // 10-Q has TWO different configurations — each has its own unique image
  { image: "10-Q-1.jpeg", names: ["ancient mystery star","triangle star","overlapping triangles star","geometric triangle star","mystery star","triangle circle","triangles in circle"] },
  { image: "10-Q-2.jpeg", names: ["night air sparkle","spinning sparkle","thin rotating points","thin sharp sparkle","fast spinning star","sparkle star","rotating sparkle"] },
  { image: "6-L.jpeg",  names: ["beautiful tangle","tangle star","complex crossing star","neat tangle","detailed crossing star","crossing line star","complex star tangle","tangle pattern"] },
  // 9-B has TWO different configurations — each has its own unique image
  { image: "9-B-1.jpeg",  names: ["treasure map star","compass star","explorer star","long thin star points","skinny star","compass rose star","twinkling compass","star compass","skinny star points"] },
  { image: "4-P.jpeg",  names: ["royal star crown","star crown","pointy star crown","star crown border","triangular crown","sharp triangle crown","king queen crown star","crown star"] },
  { image: "2-J.jpeg",  names: ["star bicycle wheel","star wheel","bicycle wheel stars","thin crisp star ring","neat star ring","round star path","bicycle wheel","star ring"] },
  { image: "15-Q.jpeg", names: ["dancing loop ring","criss cross loop ring","criss crossing loops","magical loop ring","hundreds of tiny loops","criss cross ring","loop dance ring","dancing loops circle"] },
  { image: "9-B-2.jpeg",  names: ["explorer medal","double star knot","two stars tied","two stars knot","brave explorer medal","double star","medal star","explorer medal star"] },

  // ── SPIRAL / SWIRL / CURVES ─────────────────────────────────────────────────
  { image: "4-C.jpeg",  names: ["candy swirl","lollipop spiral","lollipop candy","sweet spiral","converging spiral","candy shop spiral","lollipop pattern","candy lollipop"] },
  { image: "5-L.jpeg",  names: ["deep ocean swirl","ocean whirlpool","bold wave swirl","ocean wave circle","swooping wave circle","water whirlpool","ocean swirl","bold waves","wave circle"] },
  { image: "16-L.jpeg", names: ["giant snowflake","magical snowflake","smooth snowflake","sweeping snowflake curves","flowy snowflake","giant flowy snowflake","large snowflake","smooth curves snowflake"] },
  { image: "11-C.jpeg", names: ["spinning fan","fan blades","fan blade pattern","curved fan lines","breezy fan","fan rotation","sweeping fan blades","cool fan pattern"] },
  { image: "2-M.jpeg",  names: ["giant clock gear","clock gear","gear teeth","clock gear teeth","sharp gear teeth","mechanical gear","ticking gear","clock mechanism"] },
  { image: "2-P.jpeg",  names: ["butterfly catcher","butterfly net","wide crossing loops","airy loop net","open space loops","wide loop net","net for butterflies","bouncing ball loops"] },
  { image: "15-F.jpeg", names: ["spinning sawblade","sawblade gear","futuristic gear","zigzag spinning","sawblade pattern","fast spinning blade","zigzag gear","rotating sawblade"] },
  { image: "13-R.jpeg", names: ["happy dancing loops","dancer loops","loose playful loops","messy fun loops","dancer spinning","playful messy pattern","dancer stage loops","loose loops"] },
  { image: "1-H.jpeg",  names: ["lucky pinecone","pinecone shield","pinecone scales","overlapping scales","spinning shield scales","pinecone pattern","scale pattern","pinecone"] },
  { image: "1-I.jpeg",  names: ["tunnel of light","light tunnel","thread tunnel","woven light tunnel","woven thread circle","thread path circle","intricate thread ring","light path tunnel"] },
  { image: "7-A.jpeg",  names: ["fluffy dandelion","dandelion fluff","fuzzy dandelion","soft fluffy pattern","lines reaching out fluffy","dandelion wind","soft fluffy star","fluffy reaching lines"] },
  { image: "7-H.jpeg",  names: ["christmas wreath","cozy wreath","thick wreath","heavy round wreath","overlapping circles wreath","wreath pattern","front door wreath","overlapping circle ring","round wreath"] },
];

function findImageByPatternName(queryText) {
  const q = queryText.toLowerCase();
  for (const entry of PATTERN_NAME_MAP) {
    const sorted = [...entry.names].sort((a, b) => b.length - a.length);
    if (sorted.some(name => q.includes(name))) return entry.image;
  }
  return null;
}

// =============================================================================
// Vision Pattern Classifier
// Receives a GRAYSCALE image — color already eliminated before this call.
// temperature=0 + JSON-only output = deterministic, reliable classification.
// =============================================================================
const VISION_CLASSIFIER_PROMPT = `You are a Spin Genius pattern classifier. You will be shown a GRAYSCALE spirograph pattern image.

Your ONLY job: identify which of the 6 known patterns is shown and return a JSON object.
The image is grayscale on purpose — identify by GEOMETRY AND LINE STRUCTURE ONLY.

THE 6 KNOWN PATTERNS:

PATTERN A — Board 12-J, Sticks 5-5, Image: 12-J.jpeg
  Structure: Lines cross forming a repeating GRID or MESH all the way around a circle.
  Gap shape: SQUARE or rectangular.
  Center hole: STAR-SHAPED / POLYGON with sharp pointed corners pointing inward.
  Think: graph paper bent into a ring. Star-shaped center = strongest identifier.

PATTERN B — Board 10-A, Sticks 4-4, Image: 10-A.jpeg
  Structure: Thin lines RADIATE outward from center like wheel spokes.
  Gap shape: DIAMOND or rhombus shaped.
  Center hole: SMOOTH PERFECTLY ROUND — zero sharp corners, no star shape.

PATTERN C — Board 7-J, Sticks 1-1, Image: 7-J.jpeg
  Structure: Several loopy PETAL shapes overlapping, all meeting at center.
  Small compact rosette / doodle flower shape.

PATTERN D — Board 6-N, Sticks 5-6, Image: 6-N.jpeg
  Structure: Decorative loops ONLY on the outer edge.
  The entire center is a vast open empty space. Like a crown or lacy border.

PATTERN E — Board 3-R, Sticks 4-2, Image: 3-R.jpeg
  Structure: Lines packed SO tightly the ring looks nearly SOLID.
  Thick dense donut. Almost no empty space in the ring anywhere.

PATTERN F — Board 3-D, Sticks 3-3, Image: 3-D.jpeg
  Structure: Round bubble loops LINKED together like a chain forming one large circle.

DECISION STEPS — follow in order, stop at first match:
1. Center hole has SHARP POINTED star/polygon corners?     → PATTERN A (12-J)
2. Ring looks nearly solid, almost no gaps at all?         → PATTERN E (3-R)
3. Lines radiate from center + diamond gaps + round hole?  → PATTERN B (10-A)
4. Overlapping loopy petals meeting at center?             → PATTERN C (7-J)
5. Loops only on outer edge, huge empty center?            → PATTERN D (6-N)
6. Linked bubble/ring shapes in a chain circle?            → PATTERN F (3-D)

Respond ONLY with valid JSON — no other text whatsoever:
{"boardPosition":"12-J","stickPosition":"5-5","patternImage":"12-J.jpeg"}`;

async function classifyPatternFromImage(grayscaleImageUrl, apiKey) {
  const r = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0,
      max_tokens: 60,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: VISION_CLASSIFIER_PROMPT },
          { type: "image_url", image_url: { url: grayscaleImageUrl } }
        ]
      }]
    }),
  });
  if (!r.ok) return null;
  const data = await r.json();
  const raw = data?.choices?.[0]?.message?.content?.trim() || "";
  try {
    const clean = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);
    const known = ["3-D","12-J","6-N","7-J","3-R","10-A","9-M","2-P","4-P","4-N","1-P","15-A","2-J","16-L","9-B","2-F","2-C","4-C","5-L","6-Q","11-C","16-O","7-B","14-H","15-F","1-H","11-A","9-O","11-J","14-I","7-C","8-A","12-C","7-H","3-L","2-M","1-E","13-R","3-C","1-I","7-A","11-L","10-R","10-Q","6-L","4-H","G-6","15-Q"];
    if (!parsed?.boardPosition || !known.includes(parsed.boardPosition)) return null;

    // Boards with MULTIPLE configurations — disambiguated with -1 / -2 suffix
    // 10-Q sticks 1-1 → 10-Q-1.jpeg  |  10-Q sticks 1-2 → 10-Q-2.jpeg
    // 9-B  sticks 4-2 → 9-B-1.jpeg   |  9-B  sticks 2-1 → 9-B-2.jpeg
    const multiConfigMap = {
      "10-Q": { "1-1": "10-Q-1.jpeg", "1-2": "10-Q-2.jpeg" },
      "9-B":  { "4-2": "9-B-1.jpeg",  "2-1": "9-B-2.jpeg"  },
    };

    let patternImage;
    const boardMap = multiConfigMap[parsed.boardPosition];
    if (boardMap && parsed.stickPosition && boardMap[parsed.stickPosition]) {
      patternImage = boardMap[parsed.stickPosition];
    } else {
      patternImage = `${parsed.boardPosition}.jpeg`;
    }

    return { boardPosition: parsed.boardPosition, stickPosition: parsed.stickPosition, patternImage };
  } catch (_) {}
  return null;
}

// =============================================================================
// Handler
// =============================================================================
module.exports = async function handler(req, res) {
  const origin = req.headers.origin || "";

  if (req.method === "OPTIONS") { allow(res, origin); return res.status(204).end(); }
  if (!allow(res, origin)) return res.status(403).json({ error: "Forbidden origin", origin, allowed: origins() });
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed. Use POST." });

  try {
    const body = req.body || {};
    const product = (typeof body.product === "string" ? body.product : "robocoders").toLowerCase().trim();
    const message = typeof body.message === "string" ? body.message : typeof body.input === "string" ? body.input : "";
    const history = Array.isArray(body.history) ? body.history : Array.isArray(body.messages) ? body.messages : [];
    const attachment = body.attachment || null;

    if ((typeof message !== "string" || !message.trim()) && !attachment) {
      return res.status(400).json({ error: "Missing message or image input." });
    }

    const rawUserText = String(message || "").trim() || (attachment ? "Analyze the uploaded image and describe what you see in detail." : "");
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return res.status(500).json({ error: "OPENAI_API_KEY is not set in env." });

    // ========================================================================
    // ROBOCODERS PATH
    // ========================================================================
    if (product === "robocoders") {
      const knowledgeUrl = process.env.KNOWLEDGE_URL;
      if (!knowledgeUrl) return res.status(500).json({ error: "KNOWLEDGE_URL is not set in env." });

      const kbResp = await fetch(knowledgeUrl, { cache: "no-store" });
      if (!kbResp.ok) return res.status(500).json({ error: `Failed to fetch knowledge JSON. status=${kbResp.status}` });
      const kb = await kbResp.json();

      const { projectNames, projectsByName, lessonsByProject, canonicalPinsText, safetyText,
        kitOverview, componentsSummary, projectsSummary, componentsMap, supportConfig } = buildIndexes(kb);

      const rawIntent = detectIntent(rawUserText, projectNames, componentsMap);
      const rawDetectedProject = detectProject(rawUserText, projectNames);
      let detectedComponent = detectComponent(rawUserText, componentsMap);

      if (rawIntent.type === "KIT_OVERVIEW") {
        return res.status(200).json({ text: kitOverview, debug: { intent: rawIntent, kbMode: "deterministic_overview", product } });
      }
      if (rawIntent.type === "COMPONENTS_LIST") {
        return res.status(200).json({ text: formatComponentsList(componentsSummary), debug: { intent: rawIntent, kbMode: "deterministic_components", product } });
      }
      if (rawIntent.type === "LIST_PROJECTS" && !rawDetectedProject) {
        return res.status(200).json({
          text: `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\n\nHere are the ${projectsSummary.totalCount} live projects:\n\n` +
            projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n") + "\n\nTell me which project you'd like to learn more about!",
          debug: { intent: rawIntent, kbMode: "deterministic", product },
        });
      }
      if (rawIntent.type === "PROJECT_VIDEOS") {
        if (!rawDetectedProject) {
          return res.status(200).json({ text: "Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I'll share all the relevant lesson videos for that project.", debug: { intent: rawIntent, product } });
        }
        const videos = lessonsByProject[rawDetectedProject] || [];
        if (!videos.length) {
          return res.status(200).json({ text: `I found the project "${rawDetectedProject}", but no lesson videos are mapped for it yet.`, debug: { intent: rawIntent, product } });
        }
        const out = `Lesson videos for ${rawDetectedProject}:\n\n` + videos.map((v, idx) => {
          const linksText = (v.videoLinks || []).map((u) => `- ${u}`).join("\n");
          return `${idx + 1}. ${v.lessonName}\n${v.explainLine ? `Why this helps: ${v.explainLine}\n` : ""}Links:\n${linksText}`;
        }).join("\n\n");
        return res.status(200).json({ text: out, debug: { intent: rawIntent, kbMode: "deterministic", product } });
      }

      let plannedUserText = rawUserText;
      try { plannedUserText = await planChatPrompt(rawUserText, apiKey); } catch (_) {}

      const { lastProject, lastComponent } = resolveContextFromHistory(history, projectNames, componentsMap);
      const detectedProject = rawDetectedProject || detectProject(plannedUserText, projectNames) || lastProject;
      detectedComponent = detectComponent(plannedUserText, componentsMap) || detectedComponent || lastComponent;

      const supportReason = detectSupportFailure({
        userText: rawUserText, intent: rawIntent, detectedProject,
        projectContext: detectedProject ? projectsByName[detectedProject] : null,
        detectedComponent, componentsMap,
      });
      if (supportReason && supportConfig?.enabled && supportConfig.show_when?.includes(supportReason)) {
        return res.status(200).json({
          text: `⚠️ Need help?\n\n${supportConfig.message}\n\n📧 ${supportConfig.contact.email}\n📞 ${supportConfig.contact.phone}\n⏰ ${supportConfig.contact.hours}`,
          debug: { supportTriggered: true, supportReason, detectedProject, detectedComponent, product },
        });
      }

      let ragContext = "";
      let ragChunks = [];
      try {
        const queryEmbedding = await embedText(plannedUserText || rawUserText, apiKey);
        ragChunks = await queryPinecone(queryEmbedding, "robocoders", 6);
        if (ragChunks.length > 0) {
          ragContext = "=== RETRIEVED KNOWLEDGE (from vector search) ===\n" +
            ragChunks.map((c, i) => `[Chunk ${i + 1} | type: ${c.type}${c.projectName ? ` | project: ${c.projectName}` : ""}]\n${c.text}`).join("\n\n---\n\n");
        }
      } catch (ragErr) {
        console.error("RAG error:", ragErr.message);
      }

      const projectContext = detectedProject ? projectsByName[detectedProject] || null : null;
      const deterministicContext = buildGroundedContext({ detectedProject, projectContext, lessonsByProject, canonicalPinsText, safetyText, kitOverview, componentsSummary, projectsSummary });
      const fullContext = ragContext ? ragContext + "\n\n" + deterministicContext : deterministicContext;
      const systemPrompt = buildRobocodersSystemPrompt(fullContext);

      let userContent = [];
      if (plannedUserText?.trim()) userContent.push({ type: "text", text: plannedUserText });
      if (attachment) {
        const imageUrl = attachment.startsWith("data:") ? attachment : `data:image/png;base64,${attachment}`;
        userContent.push({ type: "image_url", image_url: { url: imageUrl } });
      }

      const messages = [{ role: "system", content: systemPrompt }, ...buildConversationHistory(history), { role: "user", content: userContent }];
      console.log("Product: robocoders | RAG chunks:", ragChunks.length, "| Intent:", rawIntent.type);

      const r = await fetch(OPENAI_CHAT_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o-mini", temperature: 0.7, max_tokens: 1000, messages }),
      });
      if (!r.ok) { const t = await r.text().catch(() => ""); return res.status(500).json({ error: "OpenAI API error", details: t.slice(0, 800) }); }
      const data = await r.json();
      let reply = data?.choices?.[0]?.message?.content?.trim() || "";
      reply = reply.replace(/\*\*(.*?)\*\*/g, "$1").replace(/^\s*#{1,6}\s*(.+)$/gm, "• $1").replace(/\n{3,}/g, "\n\n");

      return res.status(200).json({
        text: reply,
        debug: { product, detectedProject: detectedProject || null, detectedComponent: detectedComponent || null, intent: rawIntent, kbMode: "rag+llm", ragChunksRetrieved: ragChunks.length, ragChunkTypes: ragChunks.map((c) => c.type) },
      });
    }

    // ========================================================================
    // SPIN GENIUS PATH
    // ========================================================================
    if (product === "spingenius") {
      let plannedUserText = rawUserText;
      try { plannedUserText = await planChatPrompt(rawUserText, apiKey); } catch (_) {}

      let ragContext = "";
      let ragChunks = [];
      let patternImages = [];
      let grayscaleImageUrl = null;
      let classifiedPattern = null;

      // Convert image to grayscale ONCE — used for both classifier and main GPT call
      if (attachment) {
        grayscaleImageUrl = await toGrayscaleBase64(attachment);
      }

      try {
        if (!grayscaleImageUrl) {
          const queryEmbedding = await embedText(plannedUserText || rawUserText, apiKey);
          ragChunks = await queryPinecone(queryEmbedding, "spingenius", 3);
        }

        if (ragChunks.length > 0) {
          ragContext = "=== RETRIEVED KNOWLEDGE ===\n" +
            ragChunks.map((c, i) => `[Chunk ${i + 1} | type: ${c.type}]\n${c.text}`).join("\n\n---\n\n");
        }

        const configChunks = ragChunks.filter(c => c.type === "configuration" && c.patternImage);
        const queryText = (plannedUserText || rawUserText).toLowerCase();

        // Strategy 0: Grayscale image → dedicated vision classifier (temperature=0)
        

        if (grayscaleImageUrl) {
          try {
            const classified = await classifyPatternFromImage(grayscaleImageUrl, apiKey);
            console.log("Detected board:", classified?.boardPosition);
            console.log("Detected stick:", classified?.stickPosition);
            if (classified?.patternImage) {
              patternImages = [classified.patternImage];
              classifiedPattern = classified;   // STORE THE RESULT
              console.log("Vision classifier result:", classified);
            }

          } catch (classifyErr) {
            console.error("Vision classifier error:", classifyErr.message);
          }
        }

        // Strategies 1-4: text-only path (no image, or classifier failed)
        if (!grayscaleImageUrl || patternImages.length === 0) {
          const boardMatch = queryText.match(/\b([0-9]{1,2}-[a-z])\b/i)
            || queryText.match(/\b(g-[0-9])\b/i);  // handle G-6 style boards
          const askedBoard = boardMatch?.[1]?.toUpperCase();
          const stickMatch = queryText.match(/stick[s]?\s*[:\-\s]*([0-9]+-[0-9]+)/i)
            || queryText.match(/position[s]?\s*[:\-\s]*([0-9]+-[0-9]+)/i)
            || queryText.match(/\b([0-9]+-[0-9]+)\b/);
          const askedStick = stickMatch?.[1];

          if (askedBoard || askedStick) {
            // Find which boards have multiple configs — these require BOTH board+stick to resolve
            const boardCounts = {};
            configChunks.forEach(c => { boardCounts[c.boardPosition] = (boardCounts[c.boardPosition] || 0) + 1; });

            let exactChunk = null;
            if (askedBoard && askedStick) {
              // Best case: both provided — exact match
              exactChunk = configChunks.find(c =>
                (c.boardPosition || "").toUpperCase() === askedBoard &&
                c.stickPosition === askedStick
              );
            }
            if (!exactChunk && askedBoard) {
              // Board only — only safe if that board has exactly ONE config
              const boardUpper = askedBoard;
              const boardMatches = configChunks.filter(c => (c.boardPosition || "").toUpperCase() === boardUpper);
              if (boardMatches.length === 1) {
                exactChunk = boardMatches[0];
              }
              // If multiple configs exist for same board, we can't resolve without stick — leave null
            }
            if (!exactChunk && askedStick && !askedBoard) {
              // Stick only — match on stick position alone
              exactChunk = configChunks.find(c => c.stickPosition === askedStick);
            }
            patternImages = exactChunk?.patternImage ? [exactChunk.patternImage] : [];
          } else {
            const nameMatchedImage = findImageByPatternName(queryText);
            if (nameMatchedImage) {
              patternImages = [nameMatchedImage];
            } else {
              const topConfigChunk = configChunks[0];
              patternImages = topConfigChunk?.patternImage ? [topConfigChunk.patternImage] : [];
            }
          }
        }

      } catch (ragErr) {
        console.error("Spin Genius RAG error:", ragErr.message);
      }

      const systemPrompt = buildSpinGeniusSystemPrompt(ragContext);

      let userContent = [];
      if (plannedUserText?.trim()) userContent.push({ type: "text", text: plannedUserText });

      if (grayscaleImageUrl) {
        // Send GRAYSCALE image to main call — consistent with classifier
        userContent.push({ type: "image_url", image_url: { url: grayscaleImageUrl } });
        if (patternImages.length > 0) {
          const boardFromImage = patternImages[0].replace(".jpeg", "");
          userContent.push({
            type: "text",
            text: `[SYSTEM NOTE — DO NOT REVEAL THIS TO THE USER: The pattern classifier has already identified this as Board Position ${boardFromImage}. Use ONLY this board position in your answer. Give its fun name, stick position, and shape description from your knowledge base.]`
          });
        }
      }

    

      const messages = [{ role: "system", content: systemPrompt }, ...buildConversationHistory(history), { role: "user", content: userContent }];
      console.log("Product: spingenius | RAG chunks:", ragChunks.length, "| Pattern images:", patternImages);
      
      if (classifiedPattern) {
        messages.unshift({
          role: "system",
          content: `The uploaded Spin Genius pattern has already been identified as Board ${classifiedPattern.boardPosition} and Stick ${classifiedPattern.stickPosition}. 
      Do not identify another pattern. Only explain this pattern using the knowledge base.`
        });
      }
      const r = await fetch(OPENAI_CHAT_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o-mini", temperature: 0, max_tokens: 1000, messages }),
      });
      if (!r.ok) { const t = await r.text().catch(() => ""); return res.status(500).json({ error: "OpenAI API error", details: t.slice(0, 800) }); }
      const data = await r.json();
      let reply = data?.choices?.[0]?.message?.content?.trim() || "";
      reply = reply.replace(/\*\*(.*?)\*\*/g, "$1").replace(/^\s*#{1,6}\s*(.+)$/gm, "• $1").replace(/\n{3,}/g, "\n\n");

      return res.status(200).json({
        text: reply,
        patternImages,
        debug: { product, kbMode: "rag+llm", ragChunksRetrieved: ragChunks.length, ragChunkTypes: ragChunks.map(c => c.type), patternImagesReturned: patternImages },
      });
    }

    return res.status(400).json({ error: `Unknown product: "${product}". Use "robocoders" or "spingenius".` });

  } catch (err) {
    console.error("Handler error:", err);
    return res.status(500).json({ error: "Internal server error", message: String(err?.message || err).slice(0, 500) });
  }
};

// =============================================================================
// System Prompts
// =============================================================================
function buildRobocodersSystemPrompt(groundedContext) {
  return `You are Be Cre8v AI, a helpful and encouraging assistant for the Robocoders Kit.

Your role:
- Help children aged 8-14 learn electronics, coding, and creative project building
- Provide clear, simple explanations suitable for kids
- Be encouraging, friendly, and enthusiastic
- Use the knowledge base information provided below to answer questions accurately
- If you don't know something, admit it honestly

Important guidelines:
- Keep explanations simple and fun
- Use examples and analogies kids can relate to
- Always prioritize safety
- When asked about SAFETY: It is SAFE to plug/unplug sensors while the Robocoders Brain is on (low voltage 5V USB)
- When asked about PROJECT COUNT: Always say "There are 50 projects, out of which 21 are live. One project per week shall be launched."
- STRICT SCOPE: If user asks about Spin Genius, spirograph, or drawing machines, say: "I'm the Robocoders assistant! For Spin Genius questions, switch the product from the dropdown above 🤖"

KNOWLEDGE BASE:
${groundedContext}

Base your answers on the knowledge base provided. If information is not in the knowledge base, say so honestly.`;
}

function buildSpinGeniusSystemPrompt(groundedContext) {
  return `You are Be Cre8v AI, a friendly and fun assistant for the Spin Genius mechanical spirograph toy by Be Cre8v. You love talking to kids and use exciting, encouraging language!

WHAT YOU CAN HELP WITH:
- Explaining how Spin Genius works (gears, sticks, board positions)
- Identifying what pattern a configuration creates (e.g. board 3-D, sticks 3-3)
- Looking at a pattern image and identifying which configuration made it
- Describing patterns using their fun kid-friendly names
- Troubleshooting drawing issues
- Suggesting configurations to try
- Teaching geometry through spirograph patterns

ABOUT PATTERN IMAGES:
- You CANNOT generate or draw images yourself — you are a text AI
- When a user asks to "generate an image" or "show me the pattern image", respond:
  "I can't draw images myself, but the picture of this pattern will pop up automatically below my answer if it's in my knowledge base! 🌀"
- When a user UPLOADS a photo, analyse the GEOMETRIC SHAPE AND STRUCTURE ONLY

STRICT OUT-OF-SCOPE RULE:
Only redirect if user asks about: Robocoders, electronics, LEDs, coding, sensors, motors.
Do NOT redirect for: spirograph, drawing, gears, patterns, configurations, sticks, board positions, geometry, art, images of patterns.
If truly out of scope: "I'm the Spin Genius assistant! For other Be Cre8v products, switch from the dropdown above! 🌀"

PATTERN KNOWLEDGE BASE (retrieved from vector search):
${groundedContext || "No specific patterns retrieved. Use your general Spin Genius knowledge."}

PATTERN LOOKUP RULES — VERY IMPORTANT:
When a user asks about a specific configuration or pattern name, ALWAYS answer with ALL of these:
  - Fun Name (e.g. "The Happy Little Flower 🌸")
  - Pattern Name
  - Board Position
  - Stick Position
  - Shape description
  - Difficulty level
  - Tell them: "The pattern picture will appear below my response automatically! 🎨"

ALL 50 PATTERN FUN NAMES (use these when describing patterns to kids):
  Ring / Chain:
  - Board 3-D,  Sticks 3-3 → "The Magic Bubble Chain 🫧"        (Bubble Ring Chain)
  - Board 1-E,  Sticks 4-4 → "The Floating Bubble Ring 🫧"      (Soapy Bubble Garland)
  - Board 2-F,  Sticks 3-3 → "The Bubble Dream Ring 🫧"         (Soap Bubble Ring)
  - Board 9-O,  Sticks 5-5 → "The Cozy Bird Nest 🐦"            (Bird Nest Ring)

  Net / Grid:
  - Board 12-J, Sticks 5-5 → "The Magic Garden Fence 🌿"        (Garden Net)

  Crown / Lace:
  - Board 6-N,  Sticks 5-6 → "The Royal Crown 👑"               (Lace Crown)
  - Board 4-N,  Sticks 5-5 → "The Fancy Lace Ring 🎀"           (Lace Tablecloth)
  - Board 6-Q,  Sticks 5-4 → "The Bunny Ears Circle 🐰"         (Bunny Ear Ring)
  - Board 4-H,  Sticks 3-6 → "The Lacy Donut Ring 🍩"           (Lacy Flower Ring)

  Flower:
  - Board 7-J,  Sticks 1-1 → "The Happy Little Flower 🌸"       (Petal Flower)
  - Board 1-P,  Sticks 4-4 → "The Blooming Rose 🌹"             (Bold Rose)
  - Board 14-I, Sticks 4-4 → "The Blooming Sunburst 🌻"         (Garden Sunburst)
  - Board 7-C,  Sticks 2-2 → "The Fairy Stamp Flower 🧚"        (Eight Petal Flower)
  - Board 2-C,  Sticks 2-2 → "The Secret Crystal Garden 🔮"     (Crystal Ball Flower)
  - Board 3-L,  Sticks 1-5 → "The Birthday Cake Swirl 🎂"       (Swirling Petals)
  - Board 3-C,  Sticks 3-3 → "The Geometric Puzzle Bloom 🧩"    (Puzzle Flower)
  - Board 14-H, Sticks 4-3 → "The Dandelion Dream 🌼"           (Dandelion Petals)
  - Board 9-M,  Sticks 2-7 → "The Spinning Hula Hoop 🌈"        (Vibrant Hula Hoop)

  Spiderweb / Mandala:
  - Board 10-A, Sticks 4-4 → "The Shining Spiderweb ✨"         (Spiderweb Mandala)
  - Board 7-B,  Sticks 7-4 → "The Artist Spider Star 🕷️"        (Spiderweb Star)
  - Board 11-A, Sticks 6-6 → "The Royal Thread Crown 👑"        (Crown of Light)

  Dense / Solid Ring:
  - Board 3-R,  Sticks 4-2 → "The Spinning Galaxy Swirl 🌀"     (Dense Whirlpool)
  - Board 11-L, Sticks 1-1 → "The Giant Bold Donut 🍩"          (Giant Donut)
  - Board 15-A, Sticks 4-6 → "The Spinning Scribble Wheel 🛞"   (Fuzzy Tire)
  - Board G-6,  Sticks 1-2 → "The Glowing Magic Coin 🪙"        (Spinning Badge)
  - Board 16-O, Sticks 6-3 → "The Deep Sea Shell 🐚"            (Spiraling Seashell)
  - Board 12-C, Sticks 3-1 → "The Happy Glowing Sun ☀️"         (Glowing Sun)

  Star:
  - Board 8-A,  Sticks 3-2 → "The Magic Thread Snowflake ✨"    (Magic Snowflake Star)
  - Board 10-R, Sticks 1-1 → "The Magic Castle Window 🏰"       (Castle Star Window)
  - Board 10-Q, Sticks 1-1 → "The Ancient Mystery Star 🔯"      (Triangle Star)
  - Board 10-Q, Sticks 1-2 → "The Night Air Sparkle ✨"         (Spinning Sparkle)
  - Board 6-L,  Sticks 1-1 → "The Beautiful Tangle ⭐"          (Tangle Star)
  - Board 9-B,  Sticks 4-2 → "The Treasure Map Star ⭐"         (Compass Star)
  - Board 9-B,  Sticks 2-1 → "The Explorer's Medal 🎖️"          (Double Star Knot)
  - Board 4-P,  Sticks 4-5 → "The Royal Star Crown 👸"          (Star Crown)
  - Board 2-J,  Sticks 6-6 → "The Star Wheel 🚲"               (Star Bicycle Wheel)
  - Board 15-Q, Sticks 2-1 → "The Dancing Loop Ring 💫"         (Criss-Cross Loop Ring)

  Spiral / Swirl / Other:
  - Board 4-C,  Sticks 2-2 → "The Candy Swirl 🍭"              (Lollipop Spiral)
  - Board 5-L,  Sticks 6-5 → "The Deep Ocean Swirl 🌊"          (Ocean Whirlpool)
  - Board 16-L, Sticks 4-6 → "The Giant Snowflake ❄️"           (Magical Snowflake)
  - Board 11-C, Sticks 5-1 → "The Spinning Fan 🌀"              (Fan Blades)
  - Board 2-M,  Sticks 1-5 → "The Giant Clock Gear ⏰"          (Clock Gear)
  - Board 2-P,  Sticks 6-6 → "The Butterfly Catcher 🦋"         (Butterfly Net)
  - Board 15-F, Sticks 5-3 → "The Spinning Sawblade ⚙️"         (Sawblade Gear)
  - Board 13-R, Sticks 4-2 → "The Happy Dancing Loops 💃"       (Dancer's Loops)
  - Board 1-H,  Sticks 3-5 → "The Lucky Pinecone 🌲"            (Pinecone Shield)
  - Board 1-I,  Sticks 4-3 → "The Tunnel of Light 🌟"           (Light Tunnel)
  - Board 7-A,  Sticks 1-2 → "The Fluffy Dandelion ☁️"          (Dandelion Fluff)
  - Board 7-H,  Sticks 3-3 → "The Christmas Wreath 🎄"          (Cozy Wreath)
  - Board 11-J, Sticks 4-4 → "The Fancy Woven Tire 🍩"          (Woven Donut)

================================================================================
RULE ZERO — COLOR DOES NOT EXIST IN THIS IMAGE
================================================================================
Every uploaded image is converted to grayscale before you see it.
You are seeing ONLY line structure — no color information exists.
Identify ONLY from geometry: line arrangement, gap shapes, center hole shape.

================================================================================
VISUAL ANALYSIS CHECKLIST — when image is uploaded
================================================================================

STEP 1 — CENTER HOLE (most reliable test):
  Sharp pointed/star-shaped corners?    → GARDEN NET  → Board 12-J
  Nearly solid — tiny or no hole?       → DENSE RING  → Board 3-R / 11-L / 15-A / G-6
  Smooth perfectly round open hole?     → Go to Step 2

STEP 2 — LINE STRUCTURE:
  Grid / mesh of crossing lines, square gaps?         → GARDEN NET      → Board 12-J
  Spokes radiating from center, diamond gaps?         → SPIDERWEB       → Board 10-A
  Overlapping loopy petals meeting at center?         → FLOWER          → Board 7-J / 1-P / 14-I / 7-C
  Loops only on outer edge, large empty center?       → CROWN / LACE    → Board 6-N / 6-Q / 4-N
  Linked bubble / O shapes in a chain circle?         → BUBBLE CHAIN    → Board 3-D / 1-E / 2-F
  Sharp points radiating outward as a star?           → STAR PATTERN    → Board 10-R / 10-Q / 8-A / 9-B

STEP 3 — GARDEN NET vs SPIDERWEB tiebreaker:
  Center has sharp pointed star corners? → GARDEN NET → Board 12-J
  Center is smooth round circle?         → SPIDERWEB  → Board 10-A

If config not in knowledge base: "I don't have that one yet — try it out and discover your own secret pattern! Every new combo is a surprise 🎉"`;
}

// =============================================================================
// All Robocoders helper functions
// =============================================================================
function buildIndexes(kb) {
  const projectNames = extractProjectNames(kb);
  const projectsByName = {};
  const lessonsByProject = {};
  for (const pName of projectNames) {
    projectsByName[pName] = extractProjectBlock(kb, pName);
    lessonsByProject[pName] = extractLessons(kb, pName).sort((a, b) => lessonRank(a.lessonName) - lessonRank(b.lessonName));
  }
  return {
    projectNames, projectsByName, lessonsByProject,
    canonicalPinsText: extractCanonicalPins(kb),
    safetyText: extractSafety(kb),
    kitOverview: extractKitOverview(kb),
    componentsSummary: extractComponentsSummary(kb),
    projectsSummary: extractProjectsSummary(kb),
    componentsMap: extractComponentsMap(kb),
    supportConfig: extractSupportConfig(kb),
  };
}

function detectIntent(text, projectNames, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();
  if ((/what.*(is|in|about|contains?).*kit/i.test(lower) || /tell me about.*kit/i.test(lower) || /kit.*overview/i.test(lower) || (/what.*robocoders/i.test(lower) && !/brain/i.test(lower))) && !projectNames.some((proj) => lower.includes(proj.toLowerCase()))) return { type: "KIT_OVERVIEW" };
  if (/what.*(components?|parts?|pieces?).*kit/i.test(lower) || /list.*components?/i.test(lower) || /show.*components?/i.test(lower) || /components?.*list/i.test(lower)) return { type: "COMPONENTS_LIST" };
  const componentKeywords = Object.keys(componentsMap).map((id) => componentsMap[id].name?.toLowerCase()).filter(Boolean);
  const hasComponentMention = componentKeywords.some((comp) => lower.includes(comp));
  const isAskingAboutComponent = hasComponentMention && (/what.*(is|does)/i.test(lower) || /tell me about/i.test(lower) || /how.*(works?|use)/i.test(lower) || /explain/i.test(lower));
  const isAskingAboutProject = projectNames.some((proj) => lower.includes(proj.toLowerCase()));
  if (isAskingAboutComponent && !isAskingAboutProject) return { type: "COMPONENT_INFO" };
  if (/(?:list|show|what are|tell me).*(?:projects?|modules?)/i.test(lower) || /how many projects?/i.test(lower) || /all projects?/i.test(lower)) return { type: "LIST_PROJECTS" };
  if (/(?:video|lesson|tutorial|how to (?:build|make|create)).*(?:project|module)/i.test(lower) || /show.*videos?/i.test(lower) || /(?:project|module).*(?:video|lesson)/i.test(lower)) return { type: "PROJECT_VIDEOS" };
  return { type: "GENERAL" };
}

function detectProject(text, projectNames) {
  const lower = String(text || "").toLowerCase().trim();
  let bestMatch = null, bestScore = 0;
  for (const pName of projectNames) {
    const pLower = pName.toLowerCase();
    let score = 0;
    if (lower === pLower) return pName;
    if (lower.includes(pLower)) score = pLower.length * 2;
    const pWords = pLower.split(/\s+/);
    const tWords = lower.split(/\s+/);
    const matchedWords = pWords.filter((w) => tWords.includes(w));
    if (matchedWords.length > 0) { score += matchedWords.length * 3; if (matchedWords.length === pWords.length) score += 10; }
    const pSimple = pLower.replace(/[^a-z0-9]+/g, "");
    const tSimple = lower.replace(/[^a-z0-9]+/g, "");
    if (tSimple.includes(pSimple)) score += pSimple.length;
    if (score > bestScore) { bestScore = score; bestMatch = pName; }
  }
  return bestScore >= 3 ? bestMatch : null;
}

function detectComponent(text, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();
  let bestMatch = null, bestScore = 0;
  for (const [componentId, componentData] of Object.entries(componentsMap)) {
    const componentName = (componentData.name || "").toLowerCase();
    if (!componentName) continue;
    let score = 0;
    if (lower === componentName) return componentId;
    if (lower.includes(componentName)) score = componentName.length * 2;
    const variations = getComponentVariations(componentName);
    for (const variation of variations) { if (lower.includes(variation)) score += variation.length; }
    if (score > bestScore) { bestScore = score; bestMatch = componentId; }
  }
  return bestScore >= 2 ? bestMatch : null;
}

function resolveContextFromHistory(history, projectNames, componentsMap) {
  let lastProject = null, lastComponent = null;
  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    if (!msg?.content) continue;
    const text = String(msg.content);
    if (!lastProject) { const p = detectProject(text, projectNames); if (p) lastProject = p; }
    if (!lastComponent) { const c = detectComponent(text, componentsMap); if (c) lastComponent = c; }
    if (lastProject && lastComponent) break;
  }
  return { lastProject, lastComponent };
}

function getComponentVariations(name) {
  const variations = [name];
  if (name.includes("robocoders brain")) variations.push("brain", "main board", "robocoders");
  if (name.includes("ir sensor")) variations.push("infrared", "ir", "proximity sensor");
  if (name.includes("ldr")) variations.push("light sensor", "light dependent resistor");
  if (name.includes("potentiometer")) variations.push("knob", "pot", "dial");
  if (name.includes("servo motor")) variations.push("servo");
  if (name.includes("dc motor")) variations.push("motor");
  if (name.includes("rgb led")) variations.push("rgb", "color led");
  if (name.includes("keys pcb")) variations.push("keys", "buttons", "button panel");
  return variations;
}

function buildGroundedContext({ detectedProject, projectContext, lessonsByProject, canonicalPinsText, safetyText, kitOverview, componentsSummary, projectsSummary }) {
  const sections = [];
  sections.push("=== KIT OVERVIEW ===\n" + kitOverview);
  if (safetyText) sections.push("=== SAFETY RULES ===\n" + safetyText + "\n\nIMPORTANT SAFETY NOTES:\n- It is SAFE to plug and unplug sensors and components while the Robocoders Brain is powered on.\n- The system uses low voltage (5V from USB), so there is no risk of electric shock.\n- Always handle components gently to avoid physical damage.\n- Do not force connections - they should fit smoothly.");
  if (canonicalPinsText) sections.push("=== PORT MAPPINGS ===\n" + canonicalPinsText);
  if (componentsSummary) sections.push("=== COMPONENTS SUMMARY ===\n" + `Total Components: ${componentsSummary.totalCount}\nCategories available: ${Object.keys(componentsSummary.categories || {}).join(", ")}`);
  if (projectsSummary) sections.push("=== PROJECTS SUMMARY ===\n" + `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\nAvailable Projects: ${projectsSummary.projectList.join(", ")}`);
  if (detectedProject && projectContext) {
    sections.push(`=== PROJECT: ${detectedProject} ===\n${projectContext}`);
    const lessons = lessonsByProject[detectedProject] || [];
    if (lessons.length) sections.push(`=== LESSONS FOR ${detectedProject} ===\n` + lessons.map((l) => `- ${l.lessonName}\n  ${l.explainLine || ""}\n  Videos: ${l.videoLinks.join(", ")}`).join("\n\n"));
  }
  return sections.join("\n\n");
}

function buildConversationHistory(history) {
  const msgs = [];
  for (const h of history || []) {
    if (h?.role === "user" && h?.content) msgs.push({ role: "user", content: String(h.content) });
    else if (h?.role === "assistant" && h?.content) msgs.push({ role: "assistant", content: String(h.content) });
  }
  return msgs;
}

function extractProjectNames(kb) {
  return ["Hello World!", "Mood Lamp", "Game Controller", "Coin Counter", "Smart Box", "Musical Instrument", "Toll Booth", "Analog Meter", "DJ Nights", "Roll The Dice", "Table Fan", "Disco Lights", "Motion Activated Wave Sensor", "RGB Color Mixer", "The Fruit Game", "The Ping Pong Game", "The UFO Shooter Game", "The Extension Wire", "Light Intensity Meter", "Pulley LED", "Candle Lamp"];
}

function extractKitOverview(kb) {
  if (kb?.overview) {
    const o = kb.overview;
    const parts = [];
    if (o.kitName) parts.push(`Kit: ${o.kitName}`);
    if (o.description) parts.push(o.description);
    if (o.whatIsInside) parts.push(`\nWhat's Inside:\n${o.whatIsInside}`);
    if (o.keyFeatures) parts.push(`\nKey Features:\n${o.keyFeatures.map((f) => `- ${f}`).join("\n")}`);
    if (o.totalProjects) parts.push(`\nTotal Projects: ${o.totalProjects}`);
    if (o.totalComponents) parts.push(`Total Components: ${o.totalComponents}`);
    if (o.ageRange) parts.push(`Age Range: ${o.ageRange}`);
    return parts.join("\n");
  }
  return "The Robocoders Kit is an educational electronics and coding kit for children aged 8-14. It includes 21 exciting projects and 92 components to learn physical computing, Visual Block Coding, and creative project building.";
}

function extractComponentsSummary(kb) {
  if (kb?.componentsSummary) return kb.componentsSummary;
  if (kb?.glossary?.components) return { totalCount: Object.keys(kb.glossary.components).length, components: kb.glossary.components };
  return { totalCount: 92, description: "The kit includes various sensors, actuators, LEDs, motors, structural components, and craft materials.", categories: {} };
}

function extractProjectsSummary(kb) {
  const projectList = extractProjectNames(kb);
  return { totalCount: projectList.length, projectList };
}

function extractSupportConfig(kb) {
  if (kb?.support?.enabled) return kb.support;
  return null;
}

function extractComponentsMap(kb) {
  if (kb?.glossary?.components) return kb.glossary.components;
  const componentsMap = {};
  if (Array.isArray(kb?.pages)) {
    for (const page of kb.pages) {
      if (page?.type === "component" && page?.componentId && page?.componentName) {
        componentsMap[page.componentId] = { name: page.componentName, description: extractDescriptionFromText(page.text), id: page.componentId };
      }
    }
  }
  return componentsMap;
}

function extractDescriptionFromText(text) {
  const match = String(text || "").match(/Description:\s*([^\n]+(?:\n(?!Component|Type|Usage)[^\n]+)*)/i);
  return match ? match[1].trim() : "";
}

function extractProjectBlock(kb, projectName) {
  if (Array.isArray(kb?.pages)) {
    const norm = (s) => s.toLowerCase();
    const pNorm = norm(projectName);
    for (const page of kb.pages) {
      if (page?.type === "project" && norm(page.projectName || "") === pNorm) return sanitizeChunk(page.text || "");
    }
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : "")).filter(Boolean);
    let start = -1;
    for (let i = 0; i < pages.length; i++) {
      if (pages[i].toLowerCase().includes("project name") && pages[i].toLowerCase().includes(pNorm)) { start = i; break; }
    }
    if (start >= 0) return sanitizeChunk(pages.slice(start, start + 6).join("\n\n"));
  }
  const p = (kb?.projects && kb.projects[projectName]) || (Array.isArray(kb?.projects) ? kb.projects.find((x) => x?.name === projectName) : null);
  if (p) {
    const parts = [];
    if (p.description) parts.push(p.description);
    if (p.componentsUsed) parts.push("Components Used:\n" + p.componentsUsed.join("\n"));
    if (p.connections) parts.push("Connections:\n" + p.connections.join("\n"));
    if (p.steps) parts.push("Build Steps:\n" + p.steps.join("\n"));
    return sanitizeChunk(parts.join("\n\n"));
  }
  return "";
}

function extractLessons(kb, projectName) {
  if (Array.isArray(kb?.lessons) && kb.lessons.length > 0) {
    const lessons = [];
    for (const l of kb.lessons) {
      if (l.project !== projectName) continue;
      const links = Array.isArray(l.video_url) ? uniq(l.video_url.map((u) => String(u).trim())) : [String(l.video_url || "").trim()];
      links.forEach((link, i) => { lessons.push({ lessonName: links.length > 1 ? `${l.lesson_name} - Part ${i + 1}` : l.lesson_name, videoLinks: [link], explainLine: "" }); });
    }
    return dedupeLessons(lessons);
  }
  const lessons = [];
  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const p = projectName.toLowerCase();
    let inProject = false;
    for (let i = 0; i < pages.length; i++) {
      const txt = pages[i];
      const low = txt.toLowerCase();
      if (low.includes("project:") || low.includes("project :") || low.includes("project name")) { inProject = low.includes(p); }
      if (!inProject) continue;
      const blocks = txt.split(/(Lesson ID\s*[:\-]|Build\s*\d+|Coding\s*Part\s*\d+)/i);
      for (let b = 1; b < blocks.length; b++) {
        const block = blocks[b];
        const lessonName = matchLine(block, /Lesson Name\s*(?:\(Canonical\))?\s*:\s*([^\n]+)/i) || matchLine(block, /(Build\s*\d+)/i) || matchLine(block, /(Coding\s*Part\s*\d+)/i) || "";
        const links = block.match(/https?:\/\/[^\s\]\)\}\n]+/gi) || [];
        const explainLine = matchLine(block, /What this lesson helps with.*?:\s*([\s\S]*?)(?:When the AI|$)/i) || "";
        if (lessonName && links.length) { lessons.push({ lessonName: lessonName.trim(), videoLinks: uniq(links), explainLine: cleanExplain(explainLine) }); }
      }
    }
  }
  return dedupeLessons(lessons);
}

function extractCanonicalPins(kb) {
  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const idx = pages.findIndex((t) => t.toLowerCase().includes("fixed port mappings"));
    if (idx >= 0) return sanitizeChunk(pages.slice(idx, idx + 3).join("\n\n"));
  }
  return "";
}

function extractSafety(kb) {
  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const idx = pages.findIndex((t) => t.toLowerCase().includes("global safety"));
    if (idx >= 0) return sanitizeChunk(pages.slice(idx, idx + 2).join("\n\n"));
  }
  return "";
}

function formatComponentsList(componentsSummary) {
  const lines = [];
  lines.push(`The Robocoders Kit contains ${componentsSummary.totalCount || 92} components including:\n`);
  if (componentsSummary.categories) {
    const cats = componentsSummary.categories;
    if (cats.controller) { lines.push(`**Controller (${cats.controller.count}):**`); lines.push(`- ${cats.controller.components.join(", ")}`); lines.push(""); }
    if (cats.sensors) { lines.push(`**Sensors (${cats.sensors.count}):**`); lines.push(`- ${cats.sensors.components.join(", ")}`); lines.push(""); }
    if (cats.actuators) { lines.push(`**Actuators (${cats.actuators.count}):**`); lines.push(`- ${cats.actuators.components.join(", ")}`); lines.push(""); }
    if (cats.lights) { lines.push(`**Lights (${cats.lights.count}):**`); lines.push(`- ${cats.lights.components.join(", ")}`); lines.push(""); }
    if (cats.power) { lines.push(`**Power (${cats.power.count}):**`); lines.push(`- ${cats.power.components.join(", ")}`); lines.push(""); }
    if (cats.structural) { lines.push(`**Structural Components (${cats.structural.count}):**`); lines.push(`- ${cats.structural.description}`); lines.push(""); }
    if (cats.craft) { lines.push(`**Craft Materials (${cats.craft.count}):**`); lines.push(`- ${cats.craft.components.join(", ")}`); lines.push(""); }
    if (cats.mechanical) { lines.push(`**Mechanical Parts (${cats.mechanical.count}):**`); lines.push(`- ${cats.mechanical.components.join(", ")}`); lines.push(""); }
    if (cats.wiring) { lines.push(`**Wiring (${cats.wiring.count}):**`); lines.push(`- ${cats.wiring.description}`); lines.push(""); }
  }
  lines.push("\nWould you like to know more about any specific component?");
  return lines.join("\n");
}

function detectSupportFailure({ userText, detectedProject, projectContext, detectedComponent }) {
  const lower = String(userText || "").toLowerCase();
  if (/contact|customer support|support team|call|email|phone|helpline/i.test(lower)) return "USER_REQUESTED_SUPPORT";
  if (/missing|not in kit|not included|lost|component missing|nahi mila|gayab/i.test(lower)) return "PART_MISSING";
  if (/broken|damaged|burnt|burned|melted|smoke|dead|faulty|cracked|not powering/i.test(lower)) return "HARDWARE_DAMAGED";
  if (/sensor|motor|board|wire|led|wheel|fan|blade|battery|switch|sheet/i.test(lower) && !detectedComponent && !detectedProject) return "UNKNOWN_COMPONENT";
  if (detectedProject && !projectContext) return "PROJECT_NOT_IN_KB";
  return null;
}

function lessonRank(lessonName = "") {
  const n = String(lessonName || "").toLowerCase();
  if (n.includes("connection")) return 1;
  if (n.includes("build")) return 2;
  if (n.includes("coding")) return 3;
  if (n.includes("working")) return 4;
  if (n.includes("intro")) return 5;
  return 99;
}

function sanitizeChunk(s) {
  return String(s || "").replace(/\u0000/g, "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

function matchLine(text, regex) {
  const m = String(text || "").match(regex);
  if (!m) return "";
  return (m[1] || "").trim();
}

function uniq(arr) {
  const out = [], seen = new Set();
  for (const x of arr || []) { const k = String(x).trim(); if (!k || seen.has(k)) continue; seen.add(k); out.push(k); }
  return out;
}

function cleanExplain(s) {
  return String(s || "").replace(/\n+/g, " ").replace(/\s+/g, " ").trim();
}

function dedupeLessons(lessons) {
  const out = [], seen = new Set();
  for (const l of lessons || []) {
    const key = (l.lessonName || "").toLowerCase() + "::" + (l.videoLinks || []).join(",");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(l);
  }
  return out;
}
