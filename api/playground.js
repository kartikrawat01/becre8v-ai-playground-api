// /api/playground.js
// Robocoders AI Playground (chat endpoint) — with CORS + OPTIONS fixed
// Keeps deterministic logic + grounded OpenAI call
// Supports BOTH payload formats:
// - { input, messages, attachment } (current frontend)
// - { message, history } (older format)
//
// Update in this version:
// - Adds a lightweight "Prompt Planner" (gpt-4o-mini) to improve normal chat quality
// - Does NOT break deterministic flows (LIST_PROJECTS, PROJECT_VIDEOS)
// - Keeps your existing KB grounding rules intact
const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"; function origins() {
return (process.env.ALLOWED_ORIGIN || "")
.split(",")
.map((s) => s.trim())
.filter(Boolean);
}

function allow(res, origin) { res.setHeader("Vary", "Origin");
res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization"); if (!origin) return false;
const list = origins();
if (!list.length || list.some((a) => origin.startsWith(a))) { res.setHeader("Access-Control-Allow-Origin", origin); return true;
}
return false;
}

/*	Chat Prompt Planner	*/
/*
Goal:
-	Make the user message clearer + more answerable (like ChatGPT feel)
-	Only used for GENERAL chats (not deterministic LIST_PROJECTS / PROJECT_VIDEOS)
-	Output is ONLY the rewritten prompt (no extra commentary)
*/
const CHAT_PLANNER_PROMPT = `
You are Be Cre8v AI Conversation Planner.

Rewrite the user's message into a clearer, more intelligent version BEFORE it is answered.
 
Rules:
-	Do NOT answer the user.
-	Output ONLY the rewritten prompt text.
-	Preserve intent and key details.
-	Make it easier to answer with steps, structure, and the right questions.
-	Keep child-friendly, encouraging tone.
-	If the user asks something project-specific but doesn't specify the project/module name, include a short clarification question in the rewritten prompt.
-	Do not add adult/unsafe content.
`.trim();

async function planChatPrompt(userText, apiKey) { const r = await fetch(OPENAI_CHAT_URL, {
method: "POST", headers: {
Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json",
},
body: JSON.stringify({ model: "gpt-4o-mini", temperature: 0.4, messages: [
{ role: "system", content: CHAT_PLANNER_PROMPT },
{ role: "user", content: String(userText || "").trim() },
],
}),
});

if (!r.ok) {
const t = await r.text().catch(() => "");
throw new Error("Planner error: " + t.slice(0, 800));
}

const data = await r.json();
const out = data?.choices?.[0]?.message?.content?.trim(); return out || String(userText || "").trim();
}

/*	Handler	*/ export default async function handler(req, res) { const origin = req.headers.origin || "";

// 1) Preflight for CORS
if (req.method === "OPTIONS") { allow(res, origin);
return res.status(204).end();
}
 
// 2) Allowlist origin
if (!allow(res, origin)) { return res
.status(403)
.json({ error: "Forbidden origin", origin, allowed: origins() });
}

// 3) Only POST
if (req.method !== "POST") {
return res.status(405).json({ error: "Method not allowed. Use POST." });
}

try {
// ---- Accept both payload shapes (do NOT break your frontend) ---- const body = req.body || {};

// Your frontend: { input, messages, attachment }
// Old backend: { message, history } const message =
typeof body.message === "string"
? body.message
: typeof body.input === "string"
? body.input
: "";

const history = Array.isArray(body.history)
? body.history
: Array.isArray(body.messages)
? body.messages
: [];

// Attachment is currently sent by frontend; keep it for future use const attachment = body.attachment || null;

if (!message || typeof message !== "string") {
return res.status(400).json({ error: "Missing message/input string." });
}

// Raw user text (unchanged)
const rawUserText = String(message || "").trim();

//	Load KB
const knowledgeUrl = process.env.KNOWLEDGE_URL; if (!knowledgeUrl) {
return res
.status(500)
.json({ error: "KNOWLEDGE_URL is not set in env." });
 
}

const kbResp = await fetch(knowledgeUrl, { cache: "no-store" }); if (!kbResp.ok) {
return res.status(500).json({
error: `Failed to fetch knowledge JSON. status=${kbResp.status}`,
});
}

const kb = await kbResp.json();

// --------- Normalize KB into searchable structures ---------- const {
projectNames, projectsByName, lessonsByProject,
canonicalPinsText, safetyText,
} = buildIndexes(kb);

// --------- Intent detection (deterministic, using RAW text) ---------- const rawIntent = detectIntent(rawUserText);

// --------- Project detection (deterministic, using RAW text) ---------- const rawDetectedProject = detectProject(rawUserText, projectNames, kb);

// If user asks for project list: answer deterministically (no model) if (rawIntent.type === "LIST_PROJECTS") {
return res.status(200).json({ text:
"Robocoders Kit projects/modules:\n\n" +
projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n"), debug: {
detectedProject: rawDetectedProject || null, intent: rawIntent,
kbMode: "deterministic",
},
});
}

// If user asks for videos/lessons OR “not working” troubleshooting:
// return ALL relevant videos for that project, in the canonical preference order. if (rawIntent.type === "PROJECT_VIDEOS") {
if (!rawDetectedProject) { return res.status(200).json({ text:
"Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I’ll share all the relevant lesson videos for that project.",
 
debug: { detectedProject: null, intent: rawIntent },
});
}

const videos = lessonsByProject[rawDetectedProject] || []; if (!videos.length) {
return res.status(200).json({ text:
`I found the project "${rawDetectedProject}", but no lesson videos are mapped for it in the knowledge file.\n` +
"If you share the lesson links for this project, I’ll add them into the KB mapping.", debug: { detectedProject: rawDetectedProject, intent: rawIntent, videosFound: 0 },
});
}

const out =
`Lesson videos for ${rawDetectedProject}:\n\n` + videos
.map((v, idx) => {
const links = (v.videoLinks || []).map((u) => `- ${u}`).join("\n"); return (
`${idx + 1}. ${v.lessonName}\n` +
`${v.explainLine ? `Why this helps: ${v.explainLine}\n` : ""}` +
`Links:\n${links}`
);
})
.join("\n\n");

return res.status(200).json({ text: out,
debug: {
detectedProject: rawDetectedProject, intent: rawIntent,
lessonsReturned: videos.length, kbMode: "deterministic",
},
});
}

// --------- OpenAI key (needed for planner + answer model) ---------- const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) { return res
.status(500)
.json({ error: "OPENAI_API_KEY is not set in env." });
}

// --------- Planner (ONLY for GENERAL) ----------
 
// We do NOT change deterministic behavior above. 
let plannedUserText = rawUserText;

try {
plannedUserText = await planChatPrompt(rawUserText, apiKey);
} catch (plannerErr) {
// Planner is "best-effort" only; never fail the chat because planner failed. plannedUserText = rawUserText;
}

// Re-run project detection using planned text if raw didn’t detect const detectedProject =
rawDetectedProject || detectProject(plannedUserText, projectNames, kb);

// Use the existing intent (raw deterministic intent) for routing const intent = rawIntent;

// --------- Build grounded context for model ---------- const projectContext = detectedProject
? projectsByName[detectedProject] || null
: null;

const groundedContext = buildGroundedContext({ detectedProject,
projectContext, canonicalPinsText, safetyText,
lessonsForProject: detectedProject
? lessonsByProject[detectedProject] || []
: [],
intent, attachment,
});

const system = buildSystemPrompt();

// Keep short history to reduce drift
const trimmedHistory = Array.isArray(history)
? history
.slice(-10)
.filter(
  (x) =>
    x &&
    typeof x.content === "string" &&
    typeof x.role === "string"
)

: [];

// NOTE:
// - We pass plannedUserText as the final user message for better answer quality
// - We keep original history untouched to avoid breaking UI behavior const payload = {
model: "gpt-4o-mini", temperature: 0.1,
 
messages: [
{ role: "system", content: system },
{ role: "system", content: groundedContext },
...trimmedHistory,
{ role: "user", content: plannedUserText },
],
};

const resp = await fetch(OPENAI_CHAT_URL, { method: "POST",
headers: {
Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json",
},
body: JSON.stringify(payload),
});

if (!resp.ok) {
const errText = await resp.text().catch(() => ""); return res.status(500).json({
error: `OpenAI error status=${resp.status}`, details: errText.slice(0, 2000),
});
}

const data = await resp.json(); const answer =
data?.choices?.[0]?.message?.content?.trim() || "I couldn’t generate a response.";

return res.status(200).json({ text: answer,
debug: {
detectedProject: detectedProject || null, intent,
kbMode: "grounded-context",
plannerUsed: plannedUserText !== rawUserText,
},
});
} catch (e) {
return res.status(500).json({ error: "Server error",
details: String(e?.message || e),
});
}
}

/*	Helpers	*/
 
function buildSystemPrompt() { return `
You are the Robocoders Kit assistant for children aged 8–14 and parents.

Hard rules:
-	Only answer from the provided Grounded Context. If info is missing, say what is missing and ask for the exact project/module name or the missing detail.
-	Never mix projects. If the user asks Candle Lamp, do not give Mood Lamp content (and vice versa).
-	Never invent pins. If pin info is not present in Grounded Context, say you don’t have it.
-	If user asks for materials/components of a project, provide only that project's Components Used (do not dump the full kit list).
-	If user asks for lesson videos or “not working” troubleshooting: do NOT guess random links. Only share links listed in Grounded Context.
-	Safety: Do not discuss sexual content, pregnancy, “how babies are made”,
romantic/valentines content, horror movies, or any adult topics. Politely refuse and redirect to Robocoders help.

Style:
-	Simple, calm, step-by-step.
-	If it’s a troubleshooting question: suggest order = Connections first, then Build, then Coding, then Working, then Introduction.
`.trim();
}

function buildIndexes(kb) { 
if (!kb.pages && Array.isArray(kb.contents)) {
  kb.pages = kb.contents.map((c) => ({
    text: `${c.name}: ${c.description}`,
  }));
}
const projectNames = [];

const candidates =
kb?.canonical?.projects || kb?.projects ||
kb?.modules ||
kb?.glossary?.projects || [];

if (Array.isArray(candidates) && candidates.length) { for (const p of candidates) {
const name = typeof p === "string" ? p : p?.name;
if (name && !projectNames.includes(name)) projectNames.push(name);
}
}

if (!projectNames.length && Array.isArray(kb?.pages)) { for (const pg of kb.pages) {
const t = (pg?.text || "").trim();
const m = t.match(/Project Name\s*[\:\-]?\s*([^\n]+)/i); if (m && m[1]) {
 
const name = m[1].trim();
if (name && !projectNames.includes(name)) projectNames.push(name);
}
}
}

if (!projectNames.length) { projectNames.push(
"Hello World!", "Mood Lamp", "Game Controller", "Coin Counter", "Smart Box",
"Musical Instrument", "Toll Booth",
"Analog Meter", "DJ Nights",
"Roll The Dice", "Table Fan",
"Disco Lights",
"Motion Activated Wave Sensor", "RGB Color Mixer",
"The Fruit Game",
"The Ping Pong Game",
"The UFO Shooter Game", "The Extension Wire", "Light Intensity Meter", "Pulley LED",
"Candle Lamp"
);
}

const projectsByName = {};
for (const name of projectNames) {
projectsByName[name] = extractProjectBlock(kb, name) || "";

}

const lessonsByProject = {};
for (const name of projectNames) {
lessonsByProject[name] = extractLessons(kb, name); lessonsByProject[name].sort(
(a, b) => lessonRank(a.lessonName) - lessonRank(b.lessonName)
);
}

const canonicalPinsText =
extractCanonicalPins(kb) || "Canonical pin rules not found."; const safetyText =
 
extractSafety(kb) ||
"Robocoders is low-voltage and kid-safe. No wall power. No cutting or soldering. Adult help allowed when needed.";

return {
projectNames, projectsByName, lessonsByProject,
canonicalPinsText, safetyText,
};
}

function detectIntent(text) { const t = text.toLowerCase();

const wantsList = t.includes("projects") ||
t.includes("modules") || t.includes("all project") || t.includes("all module");

if (
wantsList &&
(t.includes("what are") || t.includes("list") || t.includes("give"))
) {
return { type: "LIST_PROJECTS" };
}

const videoWords = ["video", "videos", "lesson", "lessons", "link", "youtube"]; const troubleWords = [
"not working", "doesn't work",
"not responding", "stuck",
"problem", "issue",
"error",
];

if (
videoWords.some((w) => t.includes(w)) || troubleWords.some((w) => t.includes(w))
) {
return { type: "PROJECT_VIDEOS" };
}

return { type: "GENERAL" };
 
}

function detectProject(text, projectNames, kb) {
  const t = text.toLowerCase();

  // 1️⃣ Exact project name match
  for (const p of projectNames) {
    if (t.includes(p.toLowerCase())) return p;
  }

  // 2️⃣ Alias match (NEW)
  const projects = Array.isArray(kb?.projects) ? kb.projects : [];
  for (const p of projects) {
    if (Array.isArray(p.aliases)) {
      for (const a of p.aliases) {
        if (t.includes(a.toLowerCase())) return p.name;
      }
    }
  }

  return null;
}

function buildGroundedContext({ detectedProject,
projectContext, canonicalPinsText, safetyText,
lessonsForProject,
 
intent, attachment,
}) {
const lines = [];

lines.push("Grounded Context (authoritative):"); lines.push("");
lines.push("Safety:"); lines.push(safetyText); lines.push("");
lines.push("Canonical Brain / Pins / Ports rules:"); lines.push(canonicalPinsText);
lines.push("");

if (attachment && attachment.kind === "image") { lines.push("User attached an image.");
lines.push("If the user asks about the image, ask what project this photo is from and what is not working.");
lines.push("");
}

if (detectedProject && projectContext) { lines.push(`Project in focus: ${detectedProject}`); lines.push("");
lines.push("Project details (use only this for this project):"); lines.push(projectContext);
lines.push("");
} else {
lines.push("No project selected yet."); lines.push(
"If the user question is project-specific, ask for the exact project/module name."
);
lines.push("");
}

if (intent?.type === "PROJECT_VIDEOS" && detectedProject) {
lines.push("Lesson videos for this project (only these links are allowed):"); if (!lessonsForProject.length) {
lines.push("(No lesson videos found in KB for this project.)");
} else {
for (const l of lessonsForProject) { lines.push(`- ${l.lessonName}`);
if (l.explainLine) lines.push(` Why: ${l.explainLine}`); for (const u of l.videoLinks || []) lines.push(` Link: ${u}`);
}
}
lines.push("");
}
 
return lines.join("\n");
}

function extractProjectBlock(kb, projectName) {

  const projects = Array.isArray(kb?.projects) ? kb.projects : [];
  const components = Array.isArray(kb?.contents) ? kb.contents : [];

  const componentMap = {};
  for (const c of components) {
    componentMap[c.id] = c.name;
  }

  const project = projects.find(
    (p) =>
      p.name === projectName ||
      (Array.isArray(p.aliases) &&
        p.aliases.some(a => a.toLowerCase() === projectName.toLowerCase()))
  );

  if (!project) return null;

  let text = "";
  text += `Project Name: ${project.name}\n`;
  text += `Difficulty: ${project.difficulty}\n`;
  text += `Estimated Time: ${project.estimated_time}\n`;

  if (Array.isArray(project.components_used)) {

    const readable = project.components_used.map(
      id => componentMap[id] || id
    );

    text += `Components Used: ${readable.join(", ")}\n`;
  }

  return text.trim();
}
function extractLessons(kb, projectName) {
  const projects = Array.isArray(kb?.projects) ? kb.projects : [];

  const project = projects.find(
    (p) =>
      p.name === projectName ||
      (Array.isArray(p.aliases) &&
        p.aliases.some(
          (a) => a.toLowerCase() === projectName.toLowerCase()
        ))
  );

  if (!project || !Array.isArray(project.lessons)) return [];

  return project.lessons.map((l) => ({
    lessonName: l.lesson_name,
    videoLinks: l.videoLinks || [],
    explainLine: null,
  }));
}

function extractCanonicalPins(kb) { if (Array.isArray(kb?.pages)) {
const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : "")); const idx = pages.findIndex((t) =>
t.toLowerCase().includes("fixed port mappings")
);
if (idx >= 0) {
const chunk = pages.slice(idx, idx + 3).join("\n\n"); return sanitizeChunk(chunk);
}
}
return "";
}

function extractSafety(kb) { if (Array.isArray(kb?.pages)) {
const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
const idx = pages.findIndex((t) => t.toLowerCase().includes("global safety")); if (idx >= 0) {
const chunk = pages.slice(idx, idx + 2).join("\n\n"); return sanitizeChunk(chunk);
}
}
return "";
}

function lessonRank(lessonName = "") {
const n = String(lessonName || "").toLowerCase(); if (n.includes("connection")) return 1;
if (n.includes("build")) return 2; if (n.includes("coding")) return 3;
if (n.includes("working")) return 4; if (n.includes("intro")) return 5;
 
return 99;
}

function sanitizeChunk(s) { return String(s || "")
.replace(/\u0000/g, "")
.replace(/[ \t]+\n/g, "\n")
.replace(/\n{3,}/g, "\n\n")
.trim();
}

function matchLine(text, regex) {
const m = String(text || "").match(regex); if (!m) return "";
return (m[1] || "").trim();
}

function uniq(arr) { const out = [];
const seen = new Set(); for (const x of arr || []) { const k = String(x).trim();
if (!k || seen.has(k)) continue; seen.add(k);
out.push(k);
}
return out;
}

function cleanExplain(s) { return String(s || "")
.replace(/\n+/g, " ")
.replace(/\s+/g, " ")
.trim();
}

function dedupeLessons(lessons) { const out = [];
const seen = new Set();
for (const l of lessons || []) {
const key = (l.lessonName || "").toLowerCase(); if (!key || seen.has(key)) continue;
seen.add(key); out.push(l);
}
return out;
}
 

