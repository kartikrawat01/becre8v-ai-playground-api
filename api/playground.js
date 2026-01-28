// /api/playground.js
// Robocoders AI Playground (chat endpoint) — with CORS + OPTIONS fixed
// Keeps your existing deterministic logic + grounded OpenAI call
// IMPORTANT: Frontend currently sends { input, messages, attachment }
// This backend now supports BOTH formats:
// - { input, messages }  (your current frontend)
// - { message, history } (older format)

function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function allow(res, origin) {
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (!origin) return false;

  const list = origins();
  // If no list provided, allow all (not recommended for prod)
  if (!list.length || list.some((a) => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // 1) Preflight for CORS
  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  // 2) Allowlist origin
  if (!allow(res, origin)) {
    return res
      .status(403)
      .json({ error: "Forbidden origin", origin, allowed: origins() });
  }

  // 3) Only POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  try {
    // ---- Accept both payload shapes (do NOT break your frontend) ----
    const body = req.body || {};

    // Your frontend: { input, messages, attachment }
    // Old backend:   { message, history }
    const message =
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

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Missing message/input string." });
    }

    // --------- Load KB ----------
    const knowledgeUrl = process.env.KNOWLEDGE_URL;
    if (!knowledgeUrl) {
      return res
        .status(500)
        .json({ error: "KNOWLEDGE_URL is not set in env." });
    }

    const kbResp = await fetch(knowledgeUrl, { cache: "no-store" });
    if (!kbResp.ok) {
      return res.status(500).json({
        error: `Failed to fetch knowledge JSON. status=${kbResp.status}`,
      });
    }

    const kb = await kbResp.json();

    // --------- Normalize KB into searchable structures ----------
    const {
      projectNames,
      projectsByName,
      lessonsByProject,
      canonicalPinsText,
      safetyText,
    } = buildIndexes(kb);

    // --------- Project detection ----------
    const userText = String(message || "").trim();
    const detectedProject = detectProject(userText, projectNames);

    // --------- Intent detection (deterministic) ----------
    const intent = detectIntent(userText);

    // If user asks for project list: answer deterministically (no model)
    if (intent.type === "LIST_PROJECTS") {
      return res.status(200).json({
        // IMPORTANT: frontend expects data.text (not data.answer)
        text:
          "Robocoders Kit projects/modules:\n\n" +
          projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n"),
        debug: {
          detectedProject: detectedProject || null,
          intent,
          kbMode: "deterministic",
        },
      });
    }

    // If user asks for videos/lessons OR “not working” troubleshooting:
    // return ALL relevant videos for that project, in the canonical preference order.
    if (intent.type === "PROJECT_VIDEOS") {
      if (!detectedProject) {
        return res.status(200).json({
          text:
            "Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I’ll share all the relevant lesson videos for that project.",
          debug: { detectedProject: null, intent },
        });
      }

      const videos = lessonsByProject[detectedProject] || [];
      if (!videos.length) {
        return res.status(200).json({
          text:
            `I found the project "${detectedProject}", but no lesson videos are mapped for it in the knowledge file.\n` +
            "If you share the lesson links for this project, I’ll add them into the KB mapping.",
          debug: { detectedProject, intent, videosFound: 0 },
        });
      }

      // Always return ALL relevant lesson links for that project (not only 3)
      const out =
        `Lesson videos for ${detectedProject}:\n\n` +
        videos
          .map((v, idx) => {
            const links = (v.videoLinks || []).map((u) => `- ${u}`).join("\n");
            return (
              `${idx + 1}. ${v.lessonName}\n` +
              `${v.explainLine ? `Why this helps: ${v.explainLine}\n` : ""}` +
              `Links:\n${links}`
            );
          })
          .join("\n\n");

      return res.status(200).json({
        text: out,
        debug: {
          detectedProject,
          intent,
          lessonsReturned: videos.length,
          kbMode: "deterministic",
        },
      });
    }

    // --------- Build minimal grounded context for model ----------
    const projectContext = detectedProject
      ? projectsByName[detectedProject] || null
      : null;

    const groundedContext = buildGroundedContext({
      detectedProject,
      projectContext,
      canonicalPinsText,
      safetyText,
      lessonsForProject: detectedProject
        ? lessonsByProject[detectedProject] || []
        : [],
      intent,
    });

    // --------- OpenAI call ----------
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res
        .status(500)
        .json({ error: "OPENAI_API_KEY is not set in env." });
    }

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

    const payload = {
      model: "gpt-4o-mini",
      temperature: 0.1,
      messages: [
        { role: "system", content: system },
        { role: "system", content: groundedContext },
        ...trimmedHistory,
        { role: "user", content: userText },
      ],
    };

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return res.status(500).json({
        error: `OpenAI error status=${resp.status}`,
        details: errText.slice(0, 2000),
      });
    }

    const data = await resp.json();
    const answer =
      data?.choices?.[0]?.message?.content?.trim() ||
      "I couldn’t generate a response.";

    // IMPORTANT: frontend expects data.text
    return res.status(200).json({
      text: answer,
      debug: {
        detectedProject: detectedProject || null,
        intent,
        kbMode: "grounded-context",
      },
    });
  } catch (e) {
    return res.status(500).json({
      error: "Server error",
      details: String(e?.message || e),
    });
  }
}

/* ----------------------- Helpers ----------------------- */

function buildSystemPrompt() {
  return `
You are the Robocoders Kit assistant for children aged 8–14 and parents.

Hard rules:
- Only answer from the provided Grounded Context. If info is missing, say what is missing and ask for the exact project/module name or the missing detail.
- Never mix projects. If the user asks Candle Lamp, do not give Mood Lamp content (and vice versa).
- Never invent pins. If pin info is not present in Grounded Context, say you don’t have it.
- If user asks for materials/components of a project, provide only that project's Components Used (do not dump the full kit list).
- If user asks for lesson videos or “not working” troubleshooting: do NOT guess random links. Only share links listed in Grounded Context.
- Safety: Do not discuss sexual content, pregnancy, “how babies are made”, romantic/valentines content, horror movies, or any adult topics. Politely refuse and redirect to Robocoders help.

Style:
- Simple, calm, step-by-step.
- If it’s a troubleshooting question: suggest order = Connections first, then Build, then Coding, then Working, then Introduction.
`;
}

function buildIndexes(kb) {
  const projectNames = [];

  const candidates =
    kb?.canonical?.projects ||
    kb?.projects ||
    kb?.modules ||
    kb?.glossary?.projects ||
    [];

  if (Array.isArray(candidates) && candidates.length) {
    for (const p of candidates) {
      const name = typeof p === "string" ? p : p?.name;
      if (name && !projectNames.includes(name)) projectNames.push(name);
    }
  }

  if (!projectNames.length && Array.isArray(kb?.pages)) {
    for (const pg of kb.pages) {
      const t = (pg?.text || "").trim();
      const m = t.match(/Project Name\s*[\:\-]?\s*([^\n]+)/i);
      if (m && m[1]) {
        const name = m[1].trim();
        if (name && !projectNames.includes(name)) projectNames.push(name);
      }
    }
  }

  if (!projectNames.length) {
    projectNames.push(
      "Hello World!",
      "Mood Lamp",
      "Game Controller",
      "Coin Counter",
      "Smart Box",
      "Musical Instrument",
      "Toll Booth",
      "Analog Meter",
      "DJ Nights",
      "Roll The Dice",
      "Table Fan",
      "Disco Lights",
      "Motion Activated Wave Sensor",
      "RGB Color Mixer",
      "The Fruit Game",
      "The Ping Pong Game",
      "The UFO Shooter Game",
      "The Extension Wire",
      "Light Intensity Meter",
      "Pulley LED",
      "Candle Lamp"
    );
  }

  const projectsByName = {};
  for (const name of projectNames) {
    projectsByName[name] = extractProjectBlock(kb, name);
  }

  const lessonsByProject = {};
  for (const name of projectNames) {
    lessonsByProject[name] = extractLessons(kb, name);
    lessonsByProject[name].sort(
      (a, b) => lessonRank(a.lessonName) - lessonRank(b.lessonName)
    );
  }

  const canonicalPinsText = extractCanonicalPins(kb) || "Canonical pin rules not found.";
  const safetyText =
    extractSafety(kb) ||
    "Robocoders is low-voltage and kid-safe. No wall power. No cutting or soldering. Adult help allowed when needed.";

  return {
    projectNames,
    projectsByName,
    lessonsByProject,
    canonicalPinsText,
    safetyText,
  };
}

function detectIntent(text) {
  const t = text.toLowerCase();

  const wantsList =
    t.includes("projects") ||
    t.includes("modules") ||
    t.includes("all project") ||
    t.includes("all module");

  if (wantsList && (t.includes("what are") || t.includes("list") || t.includes("give"))) {
    return { type: "LIST_PROJECTS" };
  }

  const videoWords = ["video", "videos", "lesson", "lessons", "link", "youtube"];
  const troubleWords = ["not working", "doesn't work", "not responding", "stuck", "problem", "issue", "error"];

  if (videoWords.some((w) => t.includes(w)) || troubleWords.some((w) => t.includes(w))) {
    return { type: "PROJECT_VIDEOS" };
  }

  return { type: "GENERAL" };
}

function detectProject(text, projectNames) {
  const t = text.toLowerCase();

  for (const p of projectNames) {
    if (t.includes(p.toLowerCase())) return p;
  }

  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
  const nt = norm(text);

  let best = null;
  let bestScore = 0;

  for (const p of projectNames) {
    const np = norm(p);
    if (!np) continue;

    let score = 0;
    if (nt.includes(np)) score += 10;

    const chunks = np
      .split(
        /(world|lamp|game|controller|counter|box|instrument|booth|meter|dice|fan|disco|motion|rgb|fruit|pingpong|ufo|extension|pulley|candle)/
      )
      .filter(Boolean);

    for (const c of chunks) {
      if (c.length >= 4 && nt.includes(c)) score += 1;
    }

    if (score > bestScore) {
      bestScore = score;
      best = p;
    }
  }

  return bestScore >= 2 ? best : null;
}

function buildGroundedContext({
  detectedProject,
  projectContext,
  canonicalPinsText,
  safetyText,
  lessonsForProject,
  intent,
}) {
  const lines = [];

  lines.push("Grounded Context (authoritative):");
  lines.push("");
  lines.push("Safety:");
  lines.push(safetyText);
  lines.push("");
  lines.push("Canonical Brain / Pins / Ports rules:");
  lines.push(canonicalPinsText);
  lines.push("");

  if (detectedProject && projectContext) {
    lines.push(`Project in focus: ${detectedProject}`);
    lines.push("");
    lines.push("Project details (use only this for this project):");
    lines.push(projectContext);
    lines.push("");
  } else {
    lines.push("No project selected yet.");
    lines.push("If the user question is project-specific, ask for the exact project/module name.");
    lines.push("");
  }

  if (intent?.type === "PROJECT_VIDEOS" && detectedProject) {
    lines.push("Lesson videos for this project (only these links are allowed):");
    if (!lessonsForProject.length) {
      lines.push("(No lesson videos found in KB for this project.)");
    } else {
      for (const l of lessonsForProject) {
        lines.push(`- ${l.lessonName}`);
        if (l.explainLine) lines.push(`  Why: ${l.explainLine}`);
        for (const u of l.videoLinks || []) lines.push(`  Link: ${u}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}

function extractProjectBlock(kb, projectName) {
  if (Array.isArray(kb?.pages)) {
    const norm = (s) => s.toLowerCase();
    const pNorm = norm(projectName);

    const pages = kb.pages
      .map((pg) => (pg?.text ? String(pg.text) : ""))
      .filter(Boolean);

    let start = -1;
    for (let i = 0; i < pages.length; i++) {
      const txt = pages[i].toLowerCase();
      if (txt.includes("project name") && txt.includes(pNorm)) {
        start = i;
        break;
      }
    }

    if (start >= 0) {
      const chunk = pages.slice(start, start + 6).join("\n\n");
      return sanitizeChunk(chunk);
    }
  }

  const p =
    (kb?.projects && kb.projects[projectName]) ||
    (Array.isArray(kb?.projects) ? kb.projects.find((x) => x?.name === projectName) : null);

  if (p) {
    const parts = [];
    if (p.componentsUsed) parts.push("Components Used:\n" + p.componentsUsed.join("\n"));
    if (p.connections) parts.push("Connections:\n" + p.connections.join("\n"));
    if (p.steps) parts.push("Build Steps:\n" + p.steps.join("\n"));
    return sanitizeChunk(parts.join("\n\n"));
  }

  return "";
}

function extractLessons(kb, projectName) {
  const lessons = [];

  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const p = projectName.toLowerCase();

    for (let i = 0; i < pages.length; i++) {
      const txt = pages[i];
      const low = txt.toLowerCase();

      if (low.includes("project:") || low.includes("project :")) {
        if (!low.includes(p)) continue;

        const blocks = txt.split(/Lesson ID\s*[:\-]/i);
        for (let b = 1; b < blocks.length; b++) {
          const block = "Lesson ID:" + blocks[b];

          const lessonName =
            matchLine(block, /Lesson Name\s*(?:\(Canonical\))?\s*:\s*([^\n]+)/i) ||
            matchLine(block, /Lesson Name\s*:\s*([^\n]+)/i) ||
            "";

          const links = [];
          const linkMatches = block.match(/https?:\/\/[^\s\)]+/gi) || [];
          for (const u of linkMatches) {
            links.push(u.replace(/[\,\)\]]+$/g, ""));
          }

          const explainLine =
            matchLine(block, /What this lesson helps with.*?:\s*([\s\S]*?)When the AI/i) ||
            matchLine(block, /What this lesson helps with\s*\(AI explanation line\)\s*:\s*([\s\S]*?)When the AI/i) ||
            "";

          if (lessonName && links.length) {
            lessons.push({
              lessonName: lessonName.trim(),
              videoLinks: uniq(links),
              explainLine: cleanExplain(explainLine),
            });
          }
        }
      }
    }
  }

  if (!lessons.length && kb?.lessons && kb.lessons[projectName]) {
    for (const l of kb.lessons[projectName]) {
      lessons.push({
        lessonName: l.lessonName,
        videoLinks: Array.isArray(l.videoLinks)
          ? l.videoLinks
          : l.videoLink
          ? [l.videoLink]
          : [],
        explainLine: l.explainLine || "",
      });
    }
  }

  return dedupeLessons(lessons);
}

function extractCanonicalPins(kb) {
  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const idx = pages.findIndex((t) =>
      t.toLowerCase().includes("fixed port mappings")
    );
    if (idx >= 0) {
      const chunk = pages.slice(idx, idx + 3).join("\n\n");
      return sanitizeChunk(chunk);
    }
  }
  return "";
}

function extractSafety(kb) {
  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const idx = pages.findIndex((t) => t.toLowerCase().includes("global safety"));
    if (idx >= 0) {
      const chunk = pages.slice(idx, idx + 2).join("\n\n");
      return sanitizeChunk(chunk);
    }
  }
  return "";
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
  return String(s || "")
    .replace(/\u0000/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function matchLine(text, regex) {
  const m = String(text || "").match(regex);
  if (!m) return "";
  return (m[1] || "").trim();
}

function uniq(arr) {
  const out = [];
  const seen = new Set();
  for (const x of arr || []) {
    const k = String(x).trim();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push(k);
  }
  return out;
}

function cleanExplain(s) {
  return String(s || "")
    .replace(/\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function dedupeLessons(lessons) {
  const out = [];
  const seen = new Set();
  for (const l of lessons || []) {
    const key = (l.lessonName || "").toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(l);
  }
  return out;
}
