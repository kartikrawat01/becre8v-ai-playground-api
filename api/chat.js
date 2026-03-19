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
// =============================================================================
// EXACT PATTERN LOOKUP TABLE — hardcoded from spingenius.json
// This is the single source of truth for board+stick → image mapping.
// No Pinecone ranking involved — always exact, never "closest match".
// When classifier identifies a board, this table gives the precise image.
// =============================================================================
const PATTERN_LOOKUP = {
  "3-D|3-3":   { patternImage: "3-D.jpeg", patternName: "Petal Loop Donut", funName: "The Petal Loop Donut 🍩" },
  "12-J|5-5":   { patternImage: "12-J.jpeg", patternName: "Jagged Grid Ring", funName: "The Jagged Grid Ring 🌿" },
  "6-N|5-6":   { patternImage: "6-N.jpeg", patternName: "Swirling Feather Ring", funName: "The Swirling Feather Ring 🪶" },
  "7-J|1-1":   { patternImage: "7-J.jpeg", patternName: "Round Petal Flower", funName: "The Round Petal Flower 🌸" },
  "3-R|4-2":   { patternImage: "3-R.jpeg", patternName: "Woven Scallop Ring", funName: "The Woven Scallop Ring 🍩" },
  "10-A|4-4":   { patternImage: "10-A.jpeg", patternName: "Crossing Arc Ring", funName: "The Crossing Arc Ring 🌟" },
  "9-M|2-7":   { patternImage: "9-M.jpeg", patternName: "Arc Donut", funName: "The Thick Arc Donut 🍩" },
  "2-P|6-6":   { patternImage: "2-P.jpeg", patternName: "Magic Grid Globe", funName: "The Magic Grid Globe 🌐" },
  "4-P|4-5":   { patternImage: "4-P.jpeg", patternName: "Shark Fin Star", funName: "The Spinning Shark Fin Star 🦈" },
  "4-N|5-5":   { patternImage: "4-N.jpeg", patternName: "Spinning Fan Crown", funName: "The Spinning Fan Crown 🌀" },
  "1-P|4-4":   { patternImage: "1-P.jpeg", patternName: "Bold Rose Flower", funName: "The Big Bold Rose 🌹" },
  "15-A|4-6":   { patternImage: "15-A.jpeg", patternName: "Scribble Tire", funName: "The Scribble Tire 🛞" },
  "2-J|6-6":   { patternImage: "2-J.jpeg", patternName: "Diamond Lace Ring", funName: "The Diamond Lace Ring 💎" },
  "16-L|4-6":   { patternImage: "16-L.jpeg", patternName: "Swooping Snowflake", funName: "The Swooping Snowflake ❄️" },
  "9-B|4-2":   { patternImage: "9-B-1.jpeg", patternName: "Loopy Asterisk Star", funName: "The Loopy Asterisk Star ⭐" },
  "2-F|3-3":   { patternImage: "2-F.jpeg", patternName: "Soap Bubble Garland", funName: "The Soap Bubble Garland 🫧" },
  "2-C|2-2":   { patternImage: "2-C.jpeg", patternName: "Flower in a Globe", funName: "The Secret Flower Globe 🔮" },
  "4-C|2-2":   { patternImage: "4-C.jpeg", patternName: "Scribble Donut", funName: "The Little Scribble Donut 🍩" },
  "5-L|6-5":   { patternImage: "5-L.jpeg", patternName: "Feather Wreath", funName: "The Blue Feather Wreath 🪶" },
  "6-Q|5-4":   { patternImage: "6-Q.jpeg", patternName: "Heart Tip Starburst", funName: "The Heart Tip Star 💝" },
  "11-C|5-1":   { patternImage: "11-C.jpeg", patternName: "Spinning Fan Blades", funName: "The Spinning Fan 🌀" },
  "16-O|6-3":   { patternImage: "16-O.jpeg", patternName: "Grid with Curly Edge", funName: "The Curly Grid Ring 🌊" },
  "7-B|7-4":   { patternImage: "7-B.jpeg", patternName: "Petal Star", funName: "The Bold Petal Star 🌟" },
  "14-H|4-3":   { patternImage: "14-H.jpeg", patternName: "Dandelion Petal Ring", funName: "The Dandelion Wish Ring 🌼" },
  "15-F|5-3":   { patternImage: "15-F.jpeg", patternName: "Zigzag Gear", funName: "The Zigzag Gear ⚙️" },
  "1-H|3-5":   { patternImage: "1-H.jpeg", patternName: "Pinecone Scale Ring", funName: "The Lucky Pinecone Ring 🌲" },
  "11-A|6-6":   { patternImage: "11-A.jpeg", patternName: "Thread Ring", funName: "The Golden Thread Ring ✨" },
  "9-O|5-5":   { patternImage: "9-O.jpeg", patternName: "Delicate Loop Ring", funName: "The Delicate Loop Ring 🐦" },
  "11-J|4-4":   { patternImage: "11-J.jpeg", patternName: "Scalloped Woven Disc", funName: "The Scalloped Wonder Disc 🌀" },
  "14-I|4-4":   { patternImage: "14-I.jpeg", patternName: "Grand Rose", funName: "The Grand Rose 🌹" },
  "7-C|2-2":   { patternImage: "7-C.jpeg", patternName: "Eight Round Petals", funName: "The Eight Petal Stamp 🌼" },
  "8-A|3-2":   { patternImage: "8-A.jpeg", patternName: "Wispy Asterisk", funName: "The Wispy Magic Star ✨" },
  "12-C|3-1":   { patternImage: "12-C.jpeg", patternName: "Exploding Sun", funName: "The Exploding Sun ☀️" },
  "7-H|3-3":   { patternImage: "7-H.jpeg", patternName: "Circle Wreath", funName: "The Cozy Circle Wreath 🎄" },
  "3-L|1-5":   { patternImage: "3-L.jpeg", patternName: "Bold Swirling Petals", funName: "The Bold Swirling Petals 🌸" },
  "2-M|1-5":   { patternImage: "2-M.jpeg", patternName: "Spinning Pinwheel", funName: "The Spinning Pinwheel 🌸" },
  "1-E|4-4":   { patternImage: "1-E.jpeg", patternName: "Oval Bubble Ring", funName: "The Oval Bubble Ring 🫧" },
  "13-R|4-2":   { patternImage: "13-R.jpeg", patternName: "Floppy Oval Ring", funName: "The Floppy Oval Ring 🫧" },
  "3-C|3-3":   { patternImage: "3-C.jpeg", patternName: "Overlapping Circles Flower", funName: "The Circle Puzzle Flower 🌸" },
  "1-I|4-3":   { patternImage: "1-I.jpeg", patternName: "Feather Grid Ring", funName: "The Feather Grid Ring 🌿" },
  "7-A|1-2":   { patternImage: "7-A.jpeg", patternName: "Spiky Sea Urchin", funName: "The Spiky Sea Urchin 🦔" },
  "11-L|1-1":   { patternImage: "11-L.jpeg", patternName: "Thin Circle Ring", funName: "The Thin Magic Ring 💍" },
  "10-R|1-1":   { patternImage: "10-R.jpeg", patternName: "Simple Polygon Star", funName: "The Simple Magic Window ⭐" },
  "10-Q|1-1":   { patternImage: "10-Q-1.jpeg", patternName: "Triangle Star", funName: "The Triangle Magic Star 🔯" },
  "6-L|1-1":   { patternImage: "6-L.jpeg", patternName: "Dense Tangle Star", funName: "The Dense Tangle Star ⭐" },
  "10-Q|1-2":   { patternImage: "10-Q-2.jpeg", patternName: "Spiky Chaos Star", funName: "The Spiky Chaos Star 💥" },
  "9-B|2-1":   { patternImage: "9-B-2.jpeg", patternName: "Compact Loopy Star", funName: "The Compact Loopy Star 🌟" },
  "4-H|3-6":   { patternImage: "4-H.jpeg", patternName: "Tiny Loop Lace Ring", funName: "The Lacy Loop Ring 🍩" },
  "G-6|1-2":   { patternImage: "G-6.jpeg", patternName: "Tiny Sun Badge", funName: "The Tiny Sun Badge 🪙" },
  "15-Q|2-1":   { patternImage: "15-Q.jpeg", patternName: "Criss-Cross Loop Ring", funName: "The Criss-Cross Magic Ring 💫" },
};

// Lookup by board only (for cases where only board is known, no stick)
// Returns the entry if that board has exactly ONE configuration, else null.
function lookupByBoard(board) {
  const upper = board.toUpperCase();
  const matches = Object.entries(PATTERN_LOOKUP).filter(([k]) => k.startsWith(upper + "|"));
  return matches.length === 1 ? matches[0][1] : null;
}

// Lookup by board+stick — always exact, never approximate
function lookupExact(board, stick) {
  const key = `${board.toUpperCase()}|${stick}`;
  return PATTERN_LOOKUP[key] || null;
}

// =============================================================================
// Vision Pattern Classifier — GPT-4o + precise fingerprints for all 50 patterns
// Upgraded from gpt-4o-mini to gpt-4o for superior fine geometric detail vision.
// Precise distinguishing fingerprints added for every visually similar pair.
// Returns boardPosition + stickPosition. Image filename resolved via PATTERN_LOOKUP.
// =============================================================================
const VISION_CLASSIFIER_PROMPT = `You are an expert Spin Genius pattern classifier. You will be shown a GRAYSCALE spirograph pattern drawn on paper.
The image is grayscale — identify by GEOMETRY AND STRUCTURAL DETAILS ONLY. Color does not exist.

YOUR JOB: Identify the exact board position and stick position from the descriptions below.
Study carefully before deciding. Use the distinguishing fingerprints.

════════════════════════════════════════════════════════
STEP 1 — SIZE AND POSITION OF THE PATTERN ON THE DISC
════════════════════════════════════════════════════════

TINY / SMALL pattern concentrated near center of disc (lots of empty white space around it):
  → Could be: 10-R, 11-L, 9-B(2-1), 10-Q(1-1), 10-Q(1-2), 6-L, 8-A, 9-B(4-2), 7-J, 4-C, G-6, 7-A, 2-C

MEDIUM pattern filling roughly half the disc:
  → Could be: 7-B, 7-H, 7-C, 3-C, 1-I, 13-R, 1-E, 2-F, 3-D

LARGE pattern filling most or all of the disc:
  → Could be: 11-J, 14-I, 2-P, 12-J, 3-R, 6-N, 11-A, 9-M, 10-A, 3-L, all ring patterns

════════════════════════════════════════════════════════
STEP 2 — OVERALL STRUCTURE TYPE
════════════════════════════════════════════════════════

A) GRID/NET — Lines cross making square/rectangular gaps in a ring. Look for distinctive inner border.
B) CROSSING-ARC RING — Lines sweep and cross in a ring making diamond gaps. Smooth edges.
C) SCALLOPED/WOVEN DISC — Large disc-filling overlapping arcs making petal/scallop shapes.
D) PARALLEL-ARC RING — Lines are PARALLEL (NOT crossing) curved arcs stacked in a ring.
E) FEATHER/SCALE RING — Feather, leaf, or scale shapes all tilted in same swirling direction.
F) FLOWER/PETAL — Curved petal or oval shapes meeting or overlapping at/near center.
G) OVERLAPPING CIRCLES — Full round circles overlapping each other (not petals — full circles).
H) STAR/SPOKE — Straight lines radiating outward forming a star or spoke pattern.
I) FAN/PINWHEEL — Curved arms all sweeping in ONE direction like a fan or pinwheel.
J) OUTER-LOOP RING — Shapes only on outer edge, large empty center.
K) BUBBLE/OVAL CHAIN — Separate oval or bubble shapes linked in a ring.
L) DENSE/SOLID RING — Ring looks very thick and dense, arcs nearly filling the ring.
M) TINY DENSE CENTER — Very small dense pattern concentrated at center of disc.

════════════════════════════════════════════════════════
STEP 3 — EXACT BOARD FROM OBSERVED FINGERPRINTS
════════════════════════════════════════════════════════

── TYPE A: GRID/NET — CRITICAL DISAMBIGUATION ─────────────
  These 5 patterns all have crossing lines. Use this STRICT DECISION TREE:

  QUESTION 1: Does the center hole have JAGGED/SPIKY/ZIGZAG points?
    YES → 12-J sticks 5-5
      (Large grid ring. Inner border has sharp notched teeth pointing inward all around the hole.)
    NO → continue to Q2

  QUESTION 2: Are there short spiky or feather shapes pointing OUTWARD on the outer edge?
    YES → 1-I sticks 4-3
      (Medium ring. Grid inside + distinctive spikes/feathers on outer edge. Two features combined.)
    NO → continue to Q3

  QUESTION 3: Are there tiny loop or knot shapes on the outer edge?
    YES → 2-J sticks 6-6
      (Ring with diamond grid inside. Outer edge has tiny curl/knot decorations. Ring is moderately sized, large open center.)
    NO → continue to Q4

  QUESTION 4: Can you see 3 or 4 large sweeping S-CURVE or diagonal wave lines cutting through the grid?
    YES → 2-P sticks 6-6
      (Large ring/disc. The most distinctive feature: 3-4 bold curved lines sweep diagonally across the whole pattern making an S or wave shape through the grid. Fills large area.)
    NO → 11-A sticks 6-6
      (Clean thin ring of crossing lines only. No spikes, no knots, no S-curves, no jagged border. Just a plain neat woven thread ring. Large open center. This is the simplest of the group.)

── TYPE B: CROSSING-ARC RING ─────────────────────────────
  10-A sticks 4-4  — Ring where lines sweep and cross making a woven band with diamond/square gaps. Both inner and outer edges are smooth curves. Large round open center.
  2-J  sticks 6-6  — Similar ring but with tiny loop or knot shapes decorating the outer edge. Inner crossing lines make diamond gaps. Distinctive outer knot decoration.

── TYPE C: SCALLOPED/WOVEN DISC ──────────────────────────
  KEY QUESTION: How much of the disc is filled, and what is the gap/center size?

  11-J sticks 4-4  — ENTIRE DISC filled. Overlapping curved arcs make a beautiful scalloped petal grid covering the WHOLE disc from edge to edge. The arcs create a regular repeating petal pattern like fish scales. Center hole is ROUND and clearly defined.
  3-R  sticks 4-2  — LARGE RING but NOT filling full disc — wide thick ring with medium center hole. The overlapping scalloped arcs form a wide band. You can clearly see individual arc shapes. The ring band is VERY WIDE and thick, almost but not quite filling the disc.
  3-D  sticks 3-3  — MEDIUM RING, smaller than 3-R, more concentrated. The overlapping loops/arcs are packed into a ring that does NOT fill the full disc. Has a clear medium center hole. Less grand than 3-R. More like a thick donut ring than a full disc pattern.

── TYPE D: PARALLEL-ARC RING ─────────────────────────────
  9-M  sticks 2-7  — Thick ring of PARALLEL (non-crossing) curved arcs all stacked together. Very clean and regular. Lines do NOT cross each other — they are parallel. Large open center.

── TYPE E: FEATHER/SCALE RING ────────────────────────────
  KEY QUESTION: Do the individual shapes have visible lines INSIDE them (internal texture)?

  6-N  sticks 5-6  — YES internal texture. Each feather/leaf shape has many tiny lines INSIDE it making it look like a real feather with detail. The feathers also CROSS each other with crossing lines visible between them. Large open center. Ring fills large area of disc.
  5-L  sticks 6-5  — NO internal texture. Each shape is a plain clean oval or leaf with NO lines inside. Just simple clean oval outlines all tilted the same direction. Large open center. Simpler, cleaner look than 6-N.
  1-H  sticks 3-5  — Scale shapes like pinecone or fish scales. Each scale leans over neighbor in ONE consistent direction. Scales are more rounded/compact than 5-L's ovals. Open center.

── TYPE F: FLOWER/PETAL ──────────────────────────────────
  Size and petal shape are key. Compare carefully:

  7-J  sticks 1-1  — SMALL compact. 7-8 round fat oval petals overlapping at center. Centered on disc with lots of white space around. Looks like a cheerful doodle flower.
  7-C  sticks 2-2  — SMALL. Exactly 8 equal round petals perfectly symmetrical, like a stamp. Clean and neat. Small.
  7-H  sticks 3-3  — MEDIUM-SMALL. Dense circle wreath — many overlapping circular loops packed into a thick ring. More like a wreath than a flower. Small center hole.
  3-C  sticks 3-3  — MEDIUM. 8-10 large full CIRCLES (not petals) overlapping each other filling an area. Like overlapping rings. NO clear center hole.
  2-C  sticks 2-2  — TINY. A very small 4-petal flower ENCLOSED INSIDE a circular outer border. Has a visible circle outline surrounding it. Very small, centered.
  1-P  sticks 4-4  — LARGE. Big bold wide petals like a rose. Bigger and bolder than 7-J. Fills more of disc.
  14-I sticks 4-4  — LARGE. Grand rose — many large oval petals slightly rotated, spreading across most of disc. Elegant, open. Small center.
  3-L  sticks 1-5  — LARGE. 20-25 big bold oval petals all swirling in ONE direction. Strong rotation sense. Polygon-shaped center hole.
  12-C sticks 3-1  — MEDIUM-SMALL. ALL lines meet at ONE SOLID CENTER POINT. No hole at all. Like an exploding sun or firework from center. Starburst with solid center.

── TYPE G: OVERLAPPING CIRCLES ───────────────────────────
  3-C  sticks 3-3  — 8-10 large full circles overlapping. (Also listed in Type F above)

── TYPE H: STAR/SPOKE ────────────────────────────────────
  Compare by size, number of arms, and arm style:

  10-R sticks 1-1  — VERY SMALL. Just a few straight lines making a clean simple polygon star or octagon. Minimal lines. Very tiny on disc.
  8-A  sticks 3-2  — SMALL-MEDIUM. 8 very long thin wispy arms, each ending in tiny loop. Very sparse — lots of empty space between arms. Larger than 9-B(4-2). Like a wispy asterisk.
  9-B  sticks 4-2  — SMALL-MEDIUM. 8 long thin arms each ending in small rounded loop. More defined than 8-A but similar. Like a loopy asterisk star. Slightly smaller than 8-A.
  9-B  sticks 2-1  — SMALL. 8 shorter FATTER arms with BIGGER oval loops. More compact and concentrated. Looks like a compact badge star.
  7-B  sticks 7-4  — MEDIUM-LARGE. 5-6 LARGE bold oval loop petals all crossing each other at center. Very open and airy. Each loop is big. Open star shape.
  6-L  sticks 1-1  — SMALL-MEDIUM. Many straight lines crossing in a DENSE TANGLE. Busier and more complex than other stars. Small open center barely visible.
  10-Q sticks 1-1  — SMALL. Clean geometric star of neat overlapping TRIANGLES. Polygon center hole clearly visible. Organized and mathematical-looking.
  10-Q sticks 1-2  — SMALL. Many thin straight lines shooting in all DIRECTIONS making a chaotic spiky explosion. Messy and energetic. Small center.
  4-P  sticks 4-5  — MEDIUM-LARGE. Sharp SHARK FIN star points (triangular, straight one side / diagonal other). Points rotate around open center. Fills disc well.

── TYPE I: FAN/PINWHEEL ──────────────────────────────────
  11-C sticks 5-1  — 20-25 curved blades sweeping in ONE direction. Open round center. Medium disc coverage. Classic fan blade look.
  2-M  sticks 1-5  — ~25 long smooth curved arms sweeping in one direction with pointed tips. More open spacing between arms than 11-C. Round open center.
  3-L  sticks 1-5  — 20-25 large bold oval petals swirling one direction. (Also listed in Type F)
  4-N  sticks 5-5  — Fan-shaped triangular blades with tiny loop tip on each one. Distinctive loop at tip of every blade. Rotation direction visible.
  16-L sticks 4-6  — Large smooth sweeping snowflake/pinwheel curves. Very large, open, graceful. Much bigger than 11-C.

── TYPE J: OUTER-LOOP RING ───────────────────────────────
  6-Q  sticks 5-4  — Many thin radiating lines ALL ending in tiny TEARDROP/HEART loops at outer tips. Like a starburst where every tip has a tiny heart. Open center.
  4-H  sticks 3-6  — Ring of HUNDREDS of tiny loops. Very dense lacy ring. Large open center. Looks like detailed lacework.
  15-Q sticks 2-1  — Ring of HUNDREDS of tiny CRISS-CROSSING loops. Busier and more tangled than 4-H. Loops cross each other.

── TYPE K: BUBBLE/OVAL CHAIN ─────────────────────────────
  CRITICAL GROUP — compare carefully by oval COUNT and SIZE:

  13-R sticks 4-2  — 6-8 LARGE FLOPPY ovals loosely arranged in a ring. Each oval is big and casual. Very open. Large open center. Ovals are the biggest of the group.
  1-E  sticks 4-4  — ~12 THIN SMALLER ovals linked side by side in a chain ring. More numerous and thinner than 13-R. Light and delicate. Large open center.
  2-F  sticks 3-3  — ~10 VERY THIN DELICATE ovals. Even lighter/fainter/smaller than 1-E. Very airy and ghostly. Large open center.
  9-O  sticks 5-5  — Many thin loops slightly overlapping and dancing. More numerous than others. Loops touch and overlap neighbors. Ring looks more continuous than a chain.

  DISAMBIGUATION: Count ovals and check thickness.
  6-8 large floppy → 13-R
  ~12 thin medium  → 1-E
  ~10 very faint   → 2-F
  many overlapping → 9-O

── TYPE L: DENSE/SOLID RING ──────────────────────────────
  11-L sticks 1-1  — TINY NARROW RING. Looks like a thin simple circle/wedding ring. Very small, minimal lines, very thin band. Sits in center of disc. The simplest ring.
  15-A sticks 4-6  — Thick messy SCRIBBLY ring. Lines look rough, tangled, and random. Like a tire drawn by scribbling. Small center hole.
  16-O sticks 6-3  — Large grid filling most of disc. Distinctive SMALL CURL/LOOP shapes on the INNER EDGE pointing inward. Main body is a crossing-line grid.
  G-6  sticks 1-2  — TINY SMALL dense circular scribble concentrated at center of disc. Very small. Like a tiny badge or coin. Lots of empty disc space around it.
  7-A  sticks 1-2  — TINY SMALL dense SPIKY pattern at center. Many very short spiky lines shooting outward. Like a tiny sea urchin or spiky ball. Very small.
  4-C  sticks 2-2  — SMALL messy scribble donut. Small overlapping circular scribbles making a tiny donut shape. Small center hole.

── TYPE M: FUZZY/THREAD TUNNEL ───────────────────────────
  1-I  sticks 4-3  — Ring with GRID inside + SHORT SPIKES/FEATHERS on outer edge. Two combined features. (Also in Type A above)

════════════════════════════════════════════════════════
CRITICAL CONFUSION RULES
════════════════════════════════════════════════════════

12-J vs 2-P: 12-J has JAGGED inner border. 2-P has smooth inner border and fills full disc.
3-R vs 11-J vs 3-D: 11-J fills ENTIRE disc with scalloped grid. 3-R is a thick wide ring with visible arcs. 3-D is smaller/more concentrated.
1-E vs 13-R vs 2-F: Count ovals. 6-8 big floppy=13-R. ~12 thin=1-E. ~10 very faint=2-F.
9-B(4-2) vs 8-A: Both 8 arms with loops. 8-A arms are LONGER and THINNER. 9-B(4-2) slightly more defined/compact.
9-B(2-1) vs 9-B(4-2): 9-B(2-1) has SHORTER FATTER arms with BIGGER loops, more compact overall.
10-Q(1-1) vs 10-Q(1-2): 10-Q(1-1) has neat overlapping TRIANGLES and is organized. 10-Q(1-2) has chaotic spiky lines going everywhere.
11-L vs G-6 vs 7-A vs 4-C: 11-L=thin ring. G-6=tiny dense scribble badge. 7-A=tiny spiky sea urchin. 4-C=tiny scribble donut with hole.
11-C vs 2-M vs 4-N: 11-C=clean curved blades. 2-M=longer arms pointed tips. 4-N=blades with tiny loop at each tip (distinctive loop tip).
NEVER output 12-J unless inner border is clearly JAGGED/ZIGZAG/SPIKY.
11-A vs 2-P vs 1-I vs 2-J vs 12-J: Use the STRICT DECISION TREE in Type A above. Ask each question in order.
  11-A = plain woven ring, no extra features. 2-P = S-curves through grid. 2-J = outer knots. 1-I = outer spikes. 12-J = jagged inner border.
5-L vs 6-N: 6-N has lines INSIDE each feather (internal texture + crossing between feathers). 5-L shapes are PLAIN clean ovals with nothing inside.
11-C vs 2-M vs 4-N: 4-N has LOOPS at blade tips. 11-C has bare pointed tips, ~20-25 blades. 2-M has bare pointed tips, longer arms, more open spacing.
11-J vs 3-R vs 3-D: 11-J fills ENTIRE disc. 3-R is a very wide thick ring, NOT full disc. 3-D is a smaller medium ring.
NEVER output 3-R unless arcs are clearly visible in a scalloped wide ring.
ALWAYS output a board from the lists above.

Respond ONLY with valid JSON, no other text:
{"boardPosition":"1-E","stickPosition":"4-4","patternImage":"1-E.jpeg"}`;

// =============================================================================
// THREE-STEP CLASSIFIER — Grounded in direct observation of all 50 real patterns
//
// CONFIRMED FAILURES FIXED IN THIS VERSION:
//   11-J → 11-A  (scalloped disc misrouted to grid group)
//   13-R → 3-L   (floppy oval loops misrouted to flower group)
//   5-L  → 6-N   (pen line width confused with interior hatching)
//   16-L → 7-H   (large sweeping loops misrouted to flower/wreath group)
//   6-Q  → 9-M   (teardrop-tip starburst misrouted to large ring group)
//
// APPROACH:
//   Step 1 — Sequential YES/NO questions route image to correct group (A-L)
//   Step 2 — Within that group, identify the exact board
//   Step 3 — Verification pass for the most visually similar pairs
// =============================================================================

// ─── STEP 1 ──────────────────────────────────────────────────────────────────
const STEP1_CATEGORY_PROMPT = `You are a Spin Genius spirograph pattern classifier.
You will see a GRAYSCALE image of a spirograph drawn on a circular white disc.
IGNORE ALL COLOR. Judge only by SHAPE, STRUCTURE, SIZE, and LINE GEOMETRY.

Go through these questions IN ORDER. Stop at the FIRST question that matches. Output ONLY the single letter.

Q1 — TINY CENTERED PATTERN:
Is the ENTIRE pattern SMALL and sitting only at the CENTER of the disc, with large empty white area all around it? (Pattern fills less than 1/4 of the disc)
  YES → G

Q2 — TEARDROP/HEART LOOPS AT OUTER TIPS:
Do you see many thin lines or spokes radiating outward, where EACH LINE ENDS IN A SMALL TEARDROP, HEART, or LOOP SHAPE at the outer tip? Loops are only at the outer edge. Large open center.
  YES → J

Q3 — SEPARATE ENCLOSED BUBBLE/OVAL SHAPES IN A RING:
Can you see clearly SEPARATE enclosed oval or loop shapes arranged in a ring, with open space between each shape? Each shape is its own distinct closed loop — they do NOT all meet at a central point.
  YES → F

Q4 — LARGE SWEEPING PETAL LOOPS (few very large loops crossing each other):
Does the pattern have a SMALL NUMBER (5-8) of VERY LARGE, WIDE, SWEEPING oval or petal loops that CROSS THROUGH EACH OTHER, forming a ring? The loops are big and bold — each one curves widely across the disc. Has a polygon-shaped open center hole. Fills most of disc.
  YES → L

Q5 — FULL DISC SCALLOP/PETAL GRID:
Does the pattern cover MOST or ALL of the disc AND consist of overlapping arc shapes creating a repeating PETAL or SCALLOP texture like fish scales tiling across the surface?
  YES → H

Q6 — GRID WITH SPECIAL EDGE OR BODY FEATURE:
Is there a crosshatch/grid ring AND you can see ANY of: (a) jagged/spiky teeth on inner rim, (b) short spiky lines on outer edge, (c) tiny squiggly curls on outer edge, (d) large sweeping diagonal S-curves through the grid body?
  YES → A

Q7 — STRAIGHT RADIATING LINES (arms are straight, not curved petal loops):
Do you see STRAIGHT or nearly-straight lines radiating from a center point, forming a star or asterisk? The arms are straight lines — not curved petals or wide oval loops.
  YES → E

Q8 — LARGE FAN/PINWHEEL (long narrow curved blades, same sweep direction):
Do you see a large ring of LONG NARROW CURVED BLADE shapes, all sweeping in the SAME rotational direction like a spinning fan? Each blade is a long narrow individual curve.
  YES → C

Q9 — LEAF/FEATHER/SCALE RING (distinct closed leaf shapes in a ring):
Do you see a ring of LEAF, FEATHER, or SCALE shapes — each one a distinct closed outline shape, all tilted the same direction?
  YES → B

Q10 — FLOWER/PETAL (shapes meet at or near center):
Do petal or oval shapes MEET or OVERLAP at or near the CENTER of the pattern, forming a flower where petals share a center point?
  YES → D

Q11 — PLAIN CROSSING-LINE RING (smooth woven band, no extra features):
Is there a clean ring of thin lines crossing each other, with smooth inner and outer edges and no extra features?
  YES → I

Default → A

Output ONLY the single letter. Nothing else.`;

// ─── STEP 2 PROMPTS ──────────────────────────────────────────────────────────
const STEP2_PROMPTS = {

  // GROUP A: GRID / CROSSING-LINE RINGS
  A: `GRAYSCALE spirograph. Identify the GRID or CROSSING-LINE RING.
Answer each question in order. Stop at first YES.

Q1: Does the CENTER HOLE have a JAGGED/SPIKY/ZIGZAG border — sharp teeth pointing inward all around the hole rim?
  YES → {"boardPosition":"12-J","stickPosition":"5-5"}

Q2: Are there SHORT SPIKY or FEATHER-LIKE lines sticking OUTWARD beyond the ring outer edge — pointing away from center like short spears?
  YES → {"boardPosition":"1-I","stickPosition":"4-3"}

Q3: Are there tiny SQUIGGLY CURLS or SCRIBBLY LOOP decorations all around the OUTER RIM of the ring?
  YES → {"boardPosition":"2-J","stickPosition":"6-6"}

Q4: Can you see 4-5 large bold SMOOTH DIAGONAL CURVES or S-SHAPES sweeping across the ENTIRE pattern, filling most of the disc?
  YES → {"boardPosition":"2-P","stickPosition":"6-6"}

Q5: Plain ring of crossing lines — smooth edges both sides, no extra features?
  YES → {"boardPosition":"11-A","stickPosition":"6-6"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP B: LEAF / FEATHER / SCALE RING
  B: `GRAYSCALE spirograph. Identify the LEAF, FEATHER, or SCALE RING.

CRITICAL — Look carefully INSIDE one leaf/feather shape:
Can you see MULTIPLE FINE LINES running across the INTERIOR of each shape — like veins or hatching lines drawn INSIDE the body (not just the outline)?

  YES, fine interior lines inside each shape → {"boardPosition":"6-N","stickPosition":"5-6"}

  NO, shapes are plain hollow outlines — nothing drawn inside:
    Shapes LEAN OVER their neighbor like pinecone or fish scales (each one overlaps the next in one direction)?
      YES → {"boardPosition":"1-H","stickPosition":"3-5"}
    Plain clean oval or leaf outlines, all tilted same direction, hollow inside?
      YES → {"boardPosition":"5-L","stickPosition":"6-5"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP C: LARGE FAN / PINWHEEL BLADES
  C: `GRAYSCALE spirograph. Identify the LARGE FAN or PINWHEEL BLADE pattern.
Long narrow curved blades all sweeping in the same direction.

Look at the TIP of each blade:

  Each tip ends in a tiny LOOP or CURL (a small closed loop at the tip):
    → {"boardPosition":"4-N","stickPosition":"5-5"}

  Each tip ends in a small OVAL, blades WIDELY SPACED with clear gaps between them:
    → {"boardPosition":"2-M","stickPosition":"1-5"}

  Each tip ends in a small OVAL, blades CLOSER TOGETHER (denser, ~20-25 blades):
    → {"boardPosition":"11-C","stickPosition":"5-1"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP D: FLOWER / PETAL
  D: `GRAYSCALE spirograph. Identify the FLOWER or PETAL pattern.

TINY/SMALL (less than 1/3 of disc):
  Visible CIRCULAR OUTER BORDER surrounding the flower? → {"boardPosition":"2-C","stickPosition":"2-2"}
  Exactly 8 neat equal petals, no outer frame? → {"boardPosition":"7-C","stickPosition":"2-2"}
  7-8 round fatter overlapping petals, casual doodle flower? → {"boardPosition":"7-J","stickPosition":"1-1"}

MEDIUM (1/3 to 2/3 of disc):
  Dense WREATH/RING of many overlapping circular loops — looks like a donut with a center hole? → {"boardPosition":"7-H","stickPosition":"3-3"}
  Separate full ROUND CIRCLES overlapping (like soap bubbles — no clear ring hole)? → {"boardPosition":"3-C","stickPosition":"3-3"}
  ALL lines meet at ONE SOLID CENTER POINT — no hole at all (starburst/firework)? → {"boardPosition":"12-C","stickPosition":"3-1"}

LARGE (fills most of disc):
  ~20-25 WIDE BOLD oval petals ALL swirling strongly in ONE direction, polygon center hole? → {"boardPosition":"3-L","stickPosition":"1-5"}
  MANY large oval petals rotating around a small polygon center hole, not strongly directional? → {"boardPosition":"14-I","stickPosition":"4-4"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP E: STAR / SPOKE / ASTERISK
  E: `GRAYSCALE spirograph. Identify the STAR, SPOKE, or ASTERISK pattern.

NO LOOPS — plain straight lines or sharp points at arm tips:
  Very small — just a few lines making a neat OCTAGON or simple polygon ring? → {"boardPosition":"10-R","stickPosition":"1-1"}
  Small neat overlapping TRIANGLES, organized geometric star, clear polygon center hole? → {"boardPosition":"10-Q","stickPosition":"1-1"}
  Small CHAOTIC — many thin lines going all directions, messy explosion? → {"boardPosition":"10-Q","stickPosition":"1-2"}
  Small-medium — many lines in a DENSE TANGLE, very complex and busy? → {"boardPosition":"6-L","stickPosition":"1-1"}
  SHARP TRIANGULAR POINTS (shark fins) rotating around open center, fills disc well? → {"boardPosition":"4-P","stickPosition":"4-5"}

YES LOOPS at arm tips:
  5-6 VERY LARGE BOLD OVAL LOOPS crossing each other, fills most of disc, polygon center hole? → {"boardPosition":"7-B","stickPosition":"7-4"}
  8 COMPACT FAT plump oval loops, short arms, dense star? → {"boardPosition":"9-B","stickPosition":"2-1"}
  8 MEDIUM-LENGTH arms each ending in a small rounded loop, open/sparse? → {"boardPosition":"9-B","stickPosition":"4-2"}
  8 VERY LONG WISPY barely-visible arms with tiny loops, huge white space? → {"boardPosition":"8-A","stickPosition":"3-2"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP F: SEPARATE OVAL/BUBBLE SHAPES IN A RING
  F: `GRAYSCALE spirograph. Identify the SEPARATE BUBBLE/OVAL CHAIN pattern.

~7-8 LARGE FLOPPY ovals, each drawn with MULTIPLE OVERLAPPING LINES making them thick and messy. Big and casual-looking. Large open center.
  → {"boardPosition":"13-R","stickPosition":"4-2"}

~12-14 CLEAN SINGLE-LINE neat oval shapes in a tidy ring. Each oval is a crisp single-stroke outline. Very large open center.
  → {"boardPosition":"1-E","stickPosition":"4-4"}

~8-10 MEDIUM oval cluster shapes — small bundles of a few overlapping lines each. Not as large as 13-R, not as neat as 1-E.
  → {"boardPosition":"2-F","stickPosition":"3-3"}

MANY small loops OVERLAPPING continuously — ring looks nearly continuous, loops blend into each other.
  → {"boardPosition":"9-O","stickPosition":"5-5"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP G: TINY CENTERED PATTERN
  G: `GRAYSCALE spirograph. Identify the TINY CENTERED pattern.
Entire pattern is small at the disc center with large white space around it.

OUTWARD SPIKES radiating in all directions from center — spiky ball or sea urchin look. Has a tiny center hole.
  → {"boardPosition":"7-A","stickPosition":"1-2"}

Just ONE plain simple RING or CIRCLE — like a single drawn ring band. Very minimal, no complexity at all.
  → {"boardPosition":"11-L","stickPosition":"1-1"}

NEAT ORDERLY tightly-wound coil — concentric arcs wound around each other precisely. NOT messy. Small center hole. Looks like a neatly coiled spring from above.
  → {"boardPosition":"G-6","stickPosition":"1-2"}

MESSY CHAOTIC scribble donut — overlapping circular scribble lines forming a small messy donut. Lines are RANDOM. Has a center hole.
  → {"boardPosition":"4-C","stickPosition":"2-2"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP H: FULL DISC SCALLOP / PETAL GRID
  H: `GRAYSCALE spirograph. Identify the LARGE SCALLOPED or PETAL DISC pattern.

ENTIRE DISC filled from edge to edge — petal/scallop texture covers ALL space, no background gap outside. Clear ROUND center hole.
  → {"boardPosition":"11-J","stickPosition":"4-4"}

VERY WIDE RING covering most of disc — VERY SMALL center hole. Thin gap of background at outer disc edge.
  → {"boardPosition":"3-R","stickPosition":"4-2"}

MEDIUM-LARGE RING — visible white gap outside ring, MEDIUM-SIZED center hole. Overlapping loops/petals form the ring.
  → {"boardPosition":"3-D","stickPosition":"3-3"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP I: PLAIN CROSSING-LINE RING
  I: `GRAYSCALE spirograph. Identify the PLAIN CROSSING-LINE RING.

Crossing lines making diamond-shaped gaps — BOTH inner and outer edges are perfectly smooth curves. Elegant and clean. Large round open center.
  → {"boardPosition":"10-A","stickPosition":"4-4"}

Thick ring of large CHUNKY ROUNDED BLOB-LIKE sections — sections look thick and rounded/lens-shaped. Very large open center.
  → {"boardPosition":"9-M","stickPosition":"2-7"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP J: TEARDROP/HEART LOOP TIPS AT OUTER EDGE
  J: `GRAYSCALE spirograph. Identify the OUTER-LOOP TIP pattern.

Many thin spokes where EACH TIP ends in a TEARDROP or HEART-SHAPED LOOP at the outer edge. Loops are clearly teardrop/heart shaped. Fills most of disc. Large open center.
  → {"boardPosition":"6-Q","stickPosition":"5-4"}

A ring of HUNDREDS of tiny small loops — very dense lacy ring. Loops are side by side without crossing. Large open center.
  → {"boardPosition":"4-H","stickPosition":"3-6"}

A ring of HUNDREDS of tiny loops that CRISS-CROSS and interweave with each other. Busier and more tangled than 4-H.
  → {"boardPosition":"15-Q","stickPosition":"2-1"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,

  // GROUP L: LARGE SWEEPING PETAL LOOPS (16-L type)
  L: `GRAYSCALE spirograph. Identify the LARGE SWEEPING LOOP pattern.

5-6 VERY LARGE GRACEFUL SMOOTH CURVES sweeping across the disc. Very few curves, very open. Polygon-shaped center hole. Fills most of disc.
  → {"boardPosition":"16-L","stickPosition":"4-6"}

~20-25 wide bold oval petals ALL swirling strongly in ONE direction — polygon center hole, pinwheel-like rotation.
  → {"boardPosition":"3-L","stickPosition":"1-5"}

Respond ONLY with JSON: {"boardPosition":"...","stickPosition":"..."}`,
};

// ─── STEP 3: Verification prompts for the most visually similar pairs ─────────
const VERIFY_PROMPTS = {

  "11-J": `GRAYSCALE spirograph verification. Classifier said 11-J (full disc scallop pattern).
Does the petal/arc texture reach ALL THE WAY to the disc rim with NO white gap outside?
  YES fills entire disc → {"boardPosition":"11-J","stickPosition":"4-4"}
  NO gap outside, very small center hole → {"boardPosition":"3-R","stickPosition":"4-2"}
  NO gap outside, medium center hole → {"boardPosition":"3-D","stickPosition":"3-3"}
Respond ONLY with JSON.`,

  "3-D": `GRAYSCALE spirograph verification. Classifier said 3-D (medium ring, does not fill full disc).
Is there a visible white background gap between the outer ring edge and disc rim?
  YES clear gap, medium center hole → {"boardPosition":"3-D","stickPosition":"3-3"}
  YES clear gap, very small center hole → {"boardPosition":"3-R","stickPosition":"4-2"}
  NO fills entire disc edge to edge → {"boardPosition":"11-J","stickPosition":"4-4"}
Respond ONLY with JSON.`,

  "3-R": `GRAYSCALE spirograph verification. Classifier said 3-R (nearly full disc, very small center hole).
Is the center hole VERY SMALL, and is there a thin white gap at the outer disc edge?
  YES very small hole + thin outer gap → {"boardPosition":"3-R","stickPosition":"4-2"}
  NO medium-sized hole → {"boardPosition":"3-D","stickPosition":"3-3"}
  NO fills entire disc → {"boardPosition":"11-J","stickPosition":"4-4"}
Respond ONLY with JSON.`,

  "5-L": `GRAYSCALE spirograph verification. Classifier said 5-L (plain hollow oval outlines, nothing inside).
Look very carefully INSIDE one oval/leaf shape. Can you see FINE PARALLEL LINES drawn across the INTERIOR of the shape — actual additional lines inside the body (not just the outline)?
  YES fine interior lines → {"boardPosition":"6-N","stickPosition":"5-6"}
  NO plain hollow outlines → {"boardPosition":"5-L","stickPosition":"6-5"}
Respond ONLY with JSON.`,

  "6-N": `GRAYSCALE spirograph verification. Classifier said 6-N (feather shapes with interior hatching lines).
Can you clearly see fine hatching/vein lines drawn INSIDE the shape body?
  YES interior lines clearly visible → {"boardPosition":"6-N","stickPosition":"5-6"}
  NO plain hollow outlines → {"boardPosition":"5-L","stickPosition":"6-5"}
Respond ONLY with JSON.`,

  "11-A": `GRAYSCALE spirograph verification. Classifier said 11-A (plain crossing-line ring, no features).
Do the crossing lines form clear DIAMOND-SHAPED GAPS and are both edges smooth curves?
  YES diamond gaps, smooth edges → {"boardPosition":"10-A","stickPosition":"4-4"}
  Tiny SQUIGGLY CURLS on outer rim? → {"boardPosition":"2-J","stickPosition":"6-6"}
  Plain ring, no distinctive features → {"boardPosition":"11-A","stickPosition":"6-6"}
Respond ONLY with JSON.`,

  "13-R": `GRAYSCALE spirograph verification. Classifier said 13-R (7-8 large floppy multi-stroked ovals).
Are there MORE than 8 ovals with clean single-line outlines?
  YES more than 8, clean single-line → {"boardPosition":"1-E","stickPosition":"4-4"}
  YES more than 8, small oval clusters → {"boardPosition":"2-F","stickPosition":"3-3"}
  NO clearly 7-8 large floppy multi-stroked ovals → {"boardPosition":"13-R","stickPosition":"4-2"}
Respond ONLY with JSON.`,

  "1-E": `GRAYSCALE spirograph verification. Classifier said 1-E (~12-14 clean single-line ovals).
Are there only 7-8 ovals, each with multiple messy overlapping lines (floppy, thick)?
  YES only 7-8 large messy multi-stroked ovals → {"boardPosition":"13-R","stickPosition":"4-2"}
  NO about 12-14 clean neat single-line ovals → {"boardPosition":"1-E","stickPosition":"4-4"}
Respond ONLY with JSON.`,

  "9-B_4-2": `GRAYSCALE spirograph verification. Classifier said 9-B sticks 4-2 (8 medium arms, small loops).
Are the arms EXTREMELY long and wispy, reaching more than 2/3 to disc edge, with barely-visible loops?
  YES extremely long wispy arms → {"boardPosition":"8-A","stickPosition":"3-2"}
  NO medium-length arms, loops clearly visible → {"boardPosition":"9-B","stickPosition":"4-2"}
Respond ONLY with JSON.`,

  "10-Q_1-1": `GRAYSCALE spirograph verification. Classifier said 10-Q sticks 1-1 (neat overlapping triangles).
Is the pattern actually CHAOTIC — many thin lines going in all directions, messy explosion?
  YES chaotic lines everywhere → {"boardPosition":"10-Q","stickPosition":"1-2"}
  NO neat organized triangles, polygon center hole → {"boardPosition":"10-Q","stickPosition":"1-1"}
Respond ONLY with JSON.`,

  "G-6": `GRAYSCALE spirograph verification. Classifier said G-6 (neat tightly-wound coil donut).
  OUTWARD SPIKES radiating in all directions (sea urchin)? → {"boardPosition":"7-A","stickPosition":"1-2"}
  Lines are MESSY and CHAOTIC? → {"boardPosition":"4-C","stickPosition":"2-2"}
  Just a plain single ring/circle? → {"boardPosition":"11-L","stickPosition":"1-1"}
  Neat orderly coiled arcs → {"boardPosition":"G-6","stickPosition":"1-2"}
Respond ONLY with JSON.`,

  "4-C": `GRAYSCALE spirograph verification. Classifier said 4-C (messy chaotic scribble donut).
  OUTWARD SPIKES radiating in all directions? → {"boardPosition":"7-A","stickPosition":"1-2"}
  Lines are NEAT and ORDERLY? → {"boardPosition":"G-6","stickPosition":"1-2"}
  Just a plain single ring/circle? → {"boardPosition":"11-L","stickPosition":"1-1"}
  Messy chaotic scribble with center hole → {"boardPosition":"4-C","stickPosition":"2-2"}
Respond ONLY with JSON.`,

  "4-H": `GRAYSCALE spirograph verification. Classifier said 4-H (dense lacy ring of small loops).
Do the loops visibly CRISS-CROSS and interweave with each other, making it look busier and more tangled?
  YES loops criss-cross and interweave → {"boardPosition":"15-Q","stickPosition":"2-1"}
  NO loops are side by side, not crossing → {"boardPosition":"4-H","stickPosition":"3-6"}
Respond ONLY with JSON.`,

  "6-Q": `GRAYSCALE spirograph verification. Classifier said 6-Q (spokes with teardrop/heart loops at outer tips).
Are the tip-loops clearly TEARDROP or HEART SHAPED — distinct shaped loops at end of each spoke (not tiny lacy loops)?
  YES distinct teardrop/heart loops at spoke tips → {"boardPosition":"6-Q","stickPosition":"5-4"}
  NO looks like hundreds of tiny lacy loops forming a dense ring → {"boardPosition":"4-H","stickPosition":"3-6"}
Respond ONLY with JSON.`,

  "16-L": `GRAYSCALE spirograph verification. Classifier said 16-L (5-6 very large graceful sweeping curves).
Are there only 5-6 very large smooth curves (very few, very grand scale)?
  YES just 5-6 giant graceful curves, very open → {"boardPosition":"16-L","stickPosition":"4-6"}
  NO many more petals (~20+) swirling strongly in one direction → {"boardPosition":"3-L","stickPosition":"1-5"}
Respond ONLY with JSON.`,

  "11-C": `GRAYSCALE spirograph verification. Classifier said 11-C (~20-25 fan blades, closer together).
Are the blades WIDELY SPACED with clear open gaps between each blade?
  YES wide spacing → {"boardPosition":"2-M","stickPosition":"1-5"}
  NO blades are closer/denser → {"boardPosition":"11-C","stickPosition":"5-1"}
Respond ONLY with JSON.`,

  "7-H": `GRAYSCALE spirograph verification. Classifier said 7-H (wreath/donut ring of overlapping loops, has center hole).
Are the shapes actually FULL SEPARATE CIRCLES overlapping (like soap bubble rings), with no clear donut hole?
  YES full overlapping circles, no donut hole → {"boardPosition":"3-C","stickPosition":"3-3"}
  NO clearly a wreath/donut ring with a visible center hole → {"boardPosition":"7-H","stickPosition":"3-3"}
Respond ONLY with JSON.`,

  "1-P": `GRAYSCALE spirograph verification. Classifier said 1-P (6-7 giant wide bold oval loops like huge rose petals).
Are there actually MANY MORE petals (20+) swirling strongly in one direction?
  YES many petals swirling same direction, polygon center → {"boardPosition":"3-L","stickPosition":"1-5"}
  YES many rotating petals, small polygon hole, not strongly directional → {"boardPosition":"14-I","stickPosition":"4-4"}
  NO just 6-7 giant wide bold oval loops crossing → {"boardPosition":"1-P","stickPosition":"4-4"}
Respond ONLY with JSON.`,
};

// Boards that get a verification pass
const VERIFY_BOARDS = new Set([
  "11-J","3-D","3-R","5-L","6-N","11-A","13-R","1-E",
  "4-H","11-C","1-P","7-H","G-6","4-C","6-Q","16-L"
]);

function getVerifyKey(board, stick) {
  if (board === "9-B" && stick === "4-2") return "9-B_4-2";
  if (board === "10-Q" && stick === "1-1") return "10-Q_1-1";
  return board;
}

async function gptVisionCall(prompt, imageUrl, apiKey, maxTokens = 10) {
  const r = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-4o",
      temperature: 0,
      max_tokens: maxTokens,
      messages: [{ role: "user", content: [
        { type: "text", text: prompt },
        { type: "image_url", image_url: { url: imageUrl } }
      ]}]
    }),
  });
  if (!r.ok) return null;
  const data = await r.json();
  return data?.choices?.[0]?.message?.content?.trim() || null;
}

async function classifyPatternFromImage(grayscaleImageUrl, apiKey) {
  try {
    // ── STEP 1: Route to visual group ────────────────────────────────────────
    const cat = await gptVisionCall(STEP1_CATEGORY_PROMPT, grayscaleImageUrl, apiKey, 5);
    if (!cat) return null;
    const category = cat.trim().toUpperCase().charAt(0);
    console.log("Step 1 category:", category);

    const step2Prompt = STEP2_PROMPTS[category];
    if (!step2Prompt) {
      console.warn("Unknown Step 1 category:", category);
      return null;
    }

    // ── STEP 2: Identify exact board within group ────────────────────────────
    const raw = await gptVisionCall(step2Prompt, grayscaleImageUrl, apiKey, 80);
    if (!raw) return null;
    console.log("Step 2 raw:", raw);

    const clean2 = raw.replace(/```json|```/g, "").trim();
    const parsed2 = JSON.parse(clean2);
    if (!parsed2?.boardPosition) return null;

    let board = parsed2.boardPosition;
    let stick = parsed2.stickPosition || "";

    // ── STEP 3: Verification for known confusion pairs ───────────────────────
    const verifyKey = getVerifyKey(board, stick);
    if (VERIFY_BOARDS.has(board) && VERIFY_PROMPTS[verifyKey]) {
      const rawV = await gptVisionCall(VERIFY_PROMPTS[verifyKey], grayscaleImageUrl, apiKey, 60);
      if (rawV) {
        try {
          const cleanV = rawV.replace(/```json|```/g, "").trim();
          const parsedV = JSON.parse(cleanV);
          if (parsedV?.boardPosition) {
            console.log(`Step 3 verify: ${board} → ${parsedV.boardPosition} (stick: ${parsedV.stickPosition})`);
            board = parsedV.boardPosition;
            stick = parsedV.stickPosition || stick;
          }
        } catch (_) { /* keep Step 2 result */ }
      }
    }

    // ── PATTERN_LOOKUP as single source of truth ─────────────────────────────
    let entry = lookupExact(board, stick);
    if (!entry) entry = lookupByBoard(board);
    if (!entry) return null;

    return {
      boardPosition: board,
      stickPosition: stick || entry.stickPosition || "",
      patternImage: entry.patternImage,
      patternName: entry.patternName,
      funName: entry.funName,
    };
  } catch (err) {
    console.error("Classifier error:", err.message);
    return null;
  }
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

      // ── STEP 1: Convert image to grayscale ONCE ───────────────────────────
      // This physically strips color so GPT cannot use it as a shortcut.
      // The same grayscale image is used for both the classifier and main call.
      if (attachment) {
        grayscaleImageUrl = await toGrayscaleBase64(attachment);
      }

      try {
        // ── STEP 2A: IMAGE PATH — classify first, then RAG with board position ─
        // CRITICAL FIX: RAG was previously skipped when image was present.
        // Now: classify → get board/stick → query Pinecone with board name
        // so the full KB description is always available to GPT.
        if (grayscaleImageUrl) {
          try {
            const classified = await classifyPatternFromImage(grayscaleImageUrl, apiKey);
            console.log("Vision classifier result:", classified);
            if (classified?.patternImage) {
              patternImages = [classified.patternImage];
              classifiedPattern = classified;
              // Query Pinecone using pattern name + board+stick for best semantic match
              const searchQuery = `${classified.patternName || ""} board ${classified.boardPosition} sticks ${classified.stickPosition} pattern`;
              const queryEmbedding = await embedText(searchQuery, apiKey);
              ragChunks = await queryPinecone(queryEmbedding, "spingenius", 6);
            }
          } catch (classifyErr) {
            console.error("Vision classifier error:", classifyErr.message);
            // Fallback: if classifier fails, run RAG on user text
            try {
              const queryEmbedding = await embedText(plannedUserText || rawUserText, apiKey);
              ragChunks = await queryPinecone(queryEmbedding, "spingenius", 6);
            } catch (_) {}
          }
        }

        // ── STEP 2B: TEXT PATH — RAG on user text ────────────────────────────
        if (!grayscaleImageUrl) {
          const queryEmbedding = await embedText(plannedUserText || rawUserText, apiKey);
          ragChunks = await queryPinecone(queryEmbedding, "spingenius", 6);
        }

        // ── STEP 3: Build RAG context from retrieved chunks ───────────────────
        if (ragChunks.length > 0) {
          ragContext = "=== RETRIEVED KNOWLEDGE ===\n" +
            ragChunks.map((c, i) => `[Chunk ${i + 1} | type: ${c.type}]\n${c.text}`).join("\n\n---\n\n");
        }

        // ── STEP 4: Text-path image matching (no image or classifier failed) ──
        if (!grayscaleImageUrl || patternImages.length === 0) {
          const configChunks = ragChunks.filter(c => c.type === "configuration" && c.patternImage);
          const queryText = (plannedUserText || rawUserText).toLowerCase();

          const boardMatch = queryText.match(/\b([0-9]{1,2}-[a-z])\b/i)
            || queryText.match(/\b(g-[0-9])\b/i);
          const askedBoard = boardMatch?.[1]?.toUpperCase();
          const stickMatch = queryText.match(/stick[s]?\s*[:\-\s]*([0-9]+-[0-9]+)/i)
            || queryText.match(/position[s]?\s*[:\-\s]*([0-9]+-[0-9]+)/i)
            || queryText.match(/\b([0-9]+-[0-9]+)\b/);
          const askedStick = stickMatch?.[1];

          if (askedBoard || askedStick) {
            // Use PATTERN_LOOKUP for exact match — never Pinecone rank
            let entry = null;
            if (askedBoard && askedStick) {
              entry = lookupExact(askedBoard, askedStick);
            }
            if (!entry && askedBoard) {
              // Board only — safe if that board has exactly ONE config in lookup
              entry = lookupByBoard(askedBoard);
            }
            if (!entry && askedStick && !askedBoard) {
              // Stick only — find the one config with that stick position
              const stickMatch2 = Object.entries(PATTERN_LOOKUP).find(([k]) => k.endsWith("|" + askedStick));
              entry = stickMatch2?.[1] || null;
            }
            patternImages = entry?.patternImage ? [entry.patternImage] : [];
          } else {
            // Name-based lookup — PATTERN_NAME_MAP is the first gate
            const nameMatchedImage = findImageByPatternName(queryText);
            if (nameMatchedImage) {
              patternImages = [nameMatchedImage];
            } else {
              // RAG fallback: only use top config chunk if score is high enough
              // (it's the closest semantic match — not guaranteed exact)
              // We allow this only when no other signal is available
              const topConfig = configChunks[0];
              patternImages = topConfig?.patternImage ? [topConfig.patternImage] : [];
            }
          }
        }

      } catch (ragErr) {
        console.error("Spin Genius RAG error:", ragErr.message);
      }

      // ── STEP 5: Build messages — inject KB description + grayscale image ──
      const systemPrompt = buildSpinGeniusSystemPrompt(ragContext);
      let userContent = [];
      if (plannedUserText?.trim()) userContent.push({ type: "text", text: plannedUserText });

      if (grayscaleImageUrl) {
        // Send grayscale image — color is already stripped, GPT sees structure only
        userContent.push({ type: "image_url", image_url: { url: grayscaleImageUrl } });

        if (classifiedPattern) {
          // Use PATTERN_LOOKUP to get exact description — not Pinecone rank
          const entry = lookupExact(classifiedPattern.boardPosition, classifiedPattern.stickPosition)
            || lookupByBoard(classifiedPattern.boardPosition);
          const kbDesc = ragChunks.find(c =>
            c.type === "configuration" &&
            (c.boardPosition || "").toUpperCase() === classifiedPattern.boardPosition.toUpperCase() &&
            (c.stickPosition === classifiedPattern.stickPosition || !classifiedPattern.stickPosition)
          ) || ragChunks.find(c =>
            c.type === "configuration" &&
            (c.boardPosition || "").toUpperCase() === classifiedPattern.boardPosition.toUpperCase()
          );
          const descText = kbDesc?.text
            ? `\n\nKNOWLEDGE BASE DESCRIPTION FOR THIS PATTERN:\n${kbDesc.text}`
            : entry
              ? `\n\nPATTERN DETAILS: Board ${classifiedPattern.boardPosition}, Sticks ${entry.stickPosition || classifiedPattern.stickPosition}, Pattern Name: ${entry.patternName}, Fun Name: ${entry.funName}`
              : "";
          userContent.push({
            type: "text",
            text: `[SYSTEM NOTE — DO NOT REVEAL THIS TO THE USER: The vision classifier has identified this pattern as Board Position ${classifiedPattern.boardPosition}, Stick Position ${classifiedPattern.stickPosition}. Use ONLY this configuration in your answer. Do NOT re-identify the pattern from the image. Answer using the knowledge base description below.${descText}]`
          });
        }
      }

      const messages = [
        { role: "system", content: systemPrompt },
        ...buildConversationHistory(history),
        { role: "user", content: userContent }
      ];
      console.log("Product: spingenius | RAG chunks:", ragChunks.length, "| Pattern images:", patternImages, "| Classified:", classifiedPattern?.boardPosition || "none");

      const spinModel = grayscaleImageUrl ? "gpt-4o" : "gpt-4o-mini";
      const r = await fetch(OPENAI_CHAT_URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: spinModel, temperature: 0.7, max_tokens: 1000, messages }),
      });
      if (!r.ok) { const t = await r.text().catch(() => ""); return res.status(500).json({ error: "OpenAI API error", details: t.slice(0, 800) }); }
      const data = await r.json();
      let reply = data?.choices?.[0]?.message?.content?.trim() || "";
      reply = reply.replace(/\*\*(.*?)\*\*/g, "$1").replace(/^\s*#{1,6}\s*(.+)$/gm, "• $1").replace(/\n{3,}/g, "\n\n");

      return res.status(200).json({
        text: reply,
        patternImages,
        debug: { product, kbMode: "rag+llm", ragChunksRetrieved: ragChunks.length, ragChunkTypes: ragChunks.map(c => c.type), patternImagesReturned: patternImages, classifiedBoard: classifiedPattern?.boardPosition || null },
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
  - Board 3-D    Sticks 3-3 → "The Petal Loop Donut 🍩" (Petal Loop Donut)
  - Board 12-J   Sticks 5-5 → "The Jagged Grid Ring 🌿" (Jagged Grid Ring)
  - Board 6-N    Sticks 5-6 → "The Swirling Feather Ring 🪶" (Swirling Feather Ring)
  - Board 7-J    Sticks 1-1 → "The Round Petal Flower 🌸" (Round Petal Flower)
  - Board 3-R    Sticks 4-2 → "The Woven Scallop Ring 🍩" (Woven Scallop Ring)
  - Board 10-A   Sticks 4-4 → "The Crossing Arc Ring 🌟" (Crossing Arc Ring)
  - Board 9-M    Sticks 2-7 → "The Thick Arc Donut 🍩" (Arc Donut)
  - Board 2-P    Sticks 6-6 → "The Magic Grid Globe 🌐" (Magic Grid Globe)
  - Board 4-P    Sticks 4-5 → "The Spinning Shark Fin Star 🦈" (Shark Fin Star)
  - Board 4-N    Sticks 5-5 → "The Spinning Fan Crown 🌀" (Spinning Fan Crown)
  - Board 1-P    Sticks 4-4 → "The Big Bold Rose 🌹" (Bold Rose Flower)
  - Board 15-A   Sticks 4-6 → "The Scribble Tire 🛞" (Scribble Tire)
  - Board 2-J    Sticks 6-6 → "The Diamond Lace Ring 💎" (Diamond Lace Ring)
  - Board 16-L   Sticks 4-6 → "The Swooping Snowflake ❄️" (Swooping Snowflake)
  - Board 9-B    Sticks 4-2 → "The Loopy Asterisk Star ⭐" (Loopy Asterisk Star)
  - Board 2-F    Sticks 3-3 → "The Soap Bubble Garland 🫧" (Soap Bubble Garland)
  - Board 2-C    Sticks 2-2 → "The Secret Flower Globe 🔮" (Flower in a Globe)
  - Board 4-C    Sticks 2-2 → "The Little Scribble Donut 🍩" (Scribble Donut)
  - Board 5-L    Sticks 6-5 → "The Blue Feather Wreath 🪶" (Feather Wreath)
  - Board 6-Q    Sticks 5-4 → "The Heart Tip Star 💝" (Heart Tip Starburst)
  - Board 11-C   Sticks 5-1 → "The Spinning Fan 🌀" (Spinning Fan Blades)
  - Board 16-O   Sticks 6-3 → "The Curly Grid Ring 🌊" (Grid with Curly Edge)
  - Board 7-B    Sticks 7-4 → "The Bold Petal Star 🌟" (Petal Star)
  - Board 14-H   Sticks 4-3 → "The Dandelion Wish Ring 🌼" (Dandelion Petal Ring)
  - Board 15-F   Sticks 5-3 → "The Zigzag Gear ⚙️" (Zigzag Gear)
  - Board 1-H    Sticks 3-5 → "The Lucky Pinecone Ring 🌲" (Pinecone Scale Ring)
  - Board 11-A   Sticks 6-6 → "The Golden Thread Ring ✨" (Thread Ring)
  - Board 9-O    Sticks 5-5 → "The Delicate Loop Ring 🐦" (Delicate Loop Ring)
  - Board 11-J   Sticks 4-4 → "The Scalloped Wonder Disc 🌀" (Scalloped Woven Disc)
  - Board 14-I   Sticks 4-4 → "The Grand Rose 🌹" (Grand Rose)
  - Board 7-C    Sticks 2-2 → "The Eight Petal Stamp 🌼" (Eight Round Petals)
  - Board 8-A    Sticks 3-2 → "The Wispy Magic Star ✨" (Wispy Asterisk)
  - Board 12-C   Sticks 3-1 → "The Exploding Sun ☀️" (Exploding Sun)
  - Board 7-H    Sticks 3-3 → "The Cozy Circle Wreath 🎄" (Circle Wreath)
  - Board 3-L    Sticks 1-5 → "The Bold Swirling Petals 🌸" (Bold Swirling Petals)
  - Board 2-M    Sticks 1-5 → "The Spinning Pinwheel 🌸" (Spinning Pinwheel)
  - Board 1-E    Sticks 4-4 → "The Oval Bubble Ring 🫧" (Oval Bubble Ring)
  - Board 13-R   Sticks 4-2 → "The Floppy Oval Ring 🫧" (Floppy Oval Ring)
  - Board 3-C    Sticks 3-3 → "The Circle Puzzle Flower 🌸" (Overlapping Circles Flower)
  - Board 1-I    Sticks 4-3 → "The Feather Grid Ring 🌿" (Feather Grid Ring)
  - Board 7-A    Sticks 1-2 → "The Spiky Sea Urchin 🦔" (Spiky Sea Urchin)
  - Board 11-L   Sticks 1-1 → "The Thin Magic Ring 💍" (Thin Circle Ring)
  - Board 10-R   Sticks 1-1 → "The Simple Magic Window ⭐" (Simple Polygon Star)
  - Board 10-Q   Sticks 1-1 → "The Triangle Magic Star 🔯" (Triangle Star)
  - Board 6-L    Sticks 1-1 → "The Dense Tangle Star ⭐" (Dense Tangle Star)
  - Board 10-Q   Sticks 1-2 → "The Spiky Chaos Star 💥" (Spiky Chaos Star)
  - Board 9-B    Sticks 2-1 → "The Compact Loopy Star 🌟" (Compact Loopy Star)
  - Board 4-H    Sticks 3-6 → "The Lacy Loop Ring 🍩" (Tiny Loop Lace Ring)
  - Board G-6    Sticks 1-2 → "The Tiny Sun Badge 🪙" (Tiny Sun Badge)
  - Board 15-Q   Sticks 2-1 → "The Criss-Cross Magic Ring 💫" (Criss-Cross Loop Ring)
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
