// /api/playground.js
// Robocoders AI Playground (chat endpoint) — CORS + deterministic flows + grounded OpenAI call
// Payload formats supported:
// - { input, messages, attachment } (current frontend)
// - { message, history } (older format)

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";

/* ---------------- CORS helpers ---------------- */

function origins() {
  return (process.env.ALLOWED_ORIGIN || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

// Allow request if:
// - Origin is allowed, OR
// - Origin missing but Referer is allowed (some environments show Origin:null)
function isAllowedOrigin(req) {
  const list = origins();
  if (!list.length) return true; // if not set, allow all (not recommended, but avoids lockout)

  const origin = (req.headers.origin || "").trim();
  const referer = (req.headers.referer || "").trim();

  if (origin && list.some((a) => origin.startsWith(a))) return { ok: true, by: "origin", value: origin };
  if (!origin && referer && list.some((a) => referer.startsWith(a))) return { ok: true, by: "referer", value: referer };

  return { ok: false, by: origin ? "origin" : "referer", value: origin || referer || null };
}

function setCors(res, req) {
  const origin = (req.headers.origin || "").trim();
  const allow = isAllowedOrigin(req);

  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  // If origin is present and allowed, echo it back
  if (allow.ok && origin) {
    res.setHeader("Access-Control-Allow-Origin", origin);
  }

  return allow;
}

/* ---------------- Prompt Planner ---------------- */

const CHAT_PLANNER_PROMPT = `
You are Be Cre8v AI Conversation Planner.

Rewrite the user's message into a clearer, more intelligent version BEFORE it is answered.

Rules:
- Do NOT answer the user.
- Output ONLY the rewritten prompt text.
- Preserve intent and key details.
- Make it easier to answer with steps, structure, and the right questions.
- Keep child-friendly, encouraging tone.
- If the user asks something project-specific but doesn't specify the project/module name, include a short clarification question in the rewritten prompt.
- Do not add adult/unsafe content.
`.trim();

async function planChatPrompt(userText, apiKey) {
  const r = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [
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
  const out = data?.choices?.[0]?.message?.content?.trim();
  return out || String(userText || "").trim();
}

/* ---------------- Main handler ---------------- */

module.exports = async function handler(req, res) {
  // 1) CORS preflight
  if (req.method === "OPTIONS") {
    setCors(res, req);
    return res.status(204).end();
  }

  // 2) CORS allowlist
  const allow = setCors(res, req);
  if (!allow.ok) {
    return res.status(403).json({
      error: "Forbidden origin",
      seenOrigin: (req.headers.origin || null),
      seenReferer: (req.headers.referer || null),
      allowedOriginEnv: process.env.ALLOWED_ORIGIN || "",
      allowedOriginsParsed: origins(),
    });
  }

  // 3) Only POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  try {
    const body = req.body || {};

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

    const attachment = body.attachment || null;

    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "Missing message/input string." });
    }

    const rawUserText = String(message || "").trim();

    // Load KB
    const knowledgeUrl = process.env.KNOWLEDGE_URL;
    if (!knowledgeUrl) {
      return res.status(500).json({ error: "KNOWLEDGE_URL is not set in env." });
    }

    const kbResp = await fetch(knowledgeUrl, { cache: "no-store" });
    if (!kbResp.ok) {
      return res.status(500).json({
        error: `Failed to fetch knowledge JSON. status=${kbResp.status}`,
      });
    }
    const kb = await kbResp.json();

    const { projectNames, projectsByName, lessonsByProject, canonicalPinsText, safetyText } =
      buildIndexes(kb);

    const rawIntent = detectIntent(rawUserText);
    const detectedComponent = detectComponent(rawUserText, kb);

if (detectedComponent) {
  return res.status(200).json({
    text: `${detectedComponent.name}:\n${detectedComponent.description}`,
    debug: {
      detectedComponent: detectedComponent.name,
      kbMode: "component-lookup",
    },
  });
}

    const rawDetectedProject = detectProject(rawUserText, projectNames, kb);

    // Deterministic: list projects
    if (rawIntent.type === "LIST_PROJECTS") {
      return res.status(200).json({
        text:
          "Robocoders Kit projects/modules:\n\n" +
          projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n"),
        debug: {
          detectedProject: rawDetectedProject || null,
          intent: rawIntent,
          kbMode: "deterministic",
        },
      });
    }

    // Deterministic: project videos
    if (rawIntent.type === "PROJECT_VIDEOS") {
      if (!rawDetectedProject) {
        return res.status(200).json({
          text:
            "Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I’ll share all the relevant lesson videos for that project.",
          debug: { detectedProject: null, intent: rawIntent },
        });
      }

      const videos = lessonsByProject[rawDetectedProject] || [];
      if (!videos.length) {
        return res.status(200).json({
          text:
            `I found the project "${rawDetectedProject}", but no lesson videos are mapped for it in the knowledge file.\n` +
            "If you share the lesson links for this project, I’ll add them into the KB mapping.",
          debug: { detectedProject: rawDetectedProject, intent: rawIntent, videosFound: 0 },
        });
      }

      const out =
        `Lesson videos for ${rawDetectedProject}:\n\n` +
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
          detectedProject: rawDetectedProject,
          intent: rawIntent,
          lessonsReturned: videos.length,
          kbMode: "deterministic",
        },
      });
    }

    // OpenAI key
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not set in env." });
    }

    // Planner (best effort)
    let plannedUserText = rawUserText;
    try {
      plannedUserText = await planChatPrompt(rawUserText, apiKey);
    } catch (e) {
      plannedUserText = rawUserText;
    }

    const detectedProject = rawDetectedProject || detectProject(plannedUserText, projectNames, kb);
    const intent = rawIntent;

    const projectContext = detectedProject ? projectsByName[detectedProject] || null : null;

    const groundedContext = buildGroundedContext({
      detectedProject,
      projectContext,
      canonicalPinsText,
      safetyText,
      lessonsForProject: detectedProject ? lessonsByProject[detectedProject] || [] : [],
      intent,
      attachment,
    });

    const system = buildSystemPrompt();

    const trimmedHistory = Array.isArray(history)
      ? history
          .slice(-10)
          .filter((x) => x && typeof x.content === "string" && typeof x.role === "string")
      : [];

    const payload = {
      model: "gpt-4o-mini",
      temperature: 0.1,
      messages: [
        { role: "system", content: system },
        { role: "system", content: groundedContext },
        ...trimmedHistory,
        { role: "user", content: plannedUserText },
      ],
    };

    const resp = await fetch(OPENAI_CHAT_URL, {
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
    const answer = data?.choices?.[0]?.message?.content?.trim() || "I couldn’t generate a response.";

    return res.status(200).json({
      text: answer,
      debug: {
        detectedProject: detectedProject || null,
        intent,
        kbMode: "grounded-context",
        plannerUsed: plannedUserText !== rawUserText,
      },
    });
  } catch (e) {
    return res.status(500).json({
      error: "Server error",
      details: String(e?.message || e),
    });
  }
};

/* ---------------- Helpers ---------------- */

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
`.trim();
}

function buildIndexes(kb) {
  if (!kb.pages && Array.isArray(kb.contents)) {
    kb.pages = kb.contents.map((c) => ({ text: `${c.name}: ${c.description}` }));
  }

  const projectNames = [];
  const candidates = kb?.canonical?.projects || kb?.projects || kb?.modules || kb?.glossary?.projects || [];

  if (Array.isArray(candidates) && candidates.length) {
    for (const p of candidates) {
      const name = typeof p === "string"
  ? p
  : p?.name || p?.project_name;
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

  // fallback list (only if KB is missing project list)
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
    projectsByName[name] = extractProjectBlock(kb, name) || "";
  }

  const lessonsByProject = {};
  for (const name of projectNames) {
    lessonsByProject[name] = extractLessons(kb, name);
    lessonsByProject[name].sort((a, b) => lessonRank(a.lessonName) - lessonRank(b.lessonName));
  }

  const canonicalPinsText = extractCanonicalPins(kb) || "Canonical pin rules not found.";
  const safetyText =
    extractSafety(kb) ||
    "Robocoders is low-voltage and kid-safe. No wall power. No cutting or soldering. Adult help allowed when needed.";

  return { projectNames, projectsByName, lessonsByProject, canonicalPinsText, safetyText };
}

function detectIntent(text) {
  const t = text.toLowerCase();

  const wantsList = t.includes("projects") || t.includes("modules") || t.includes("all project") || t.includes("all module");
  if (wantsList && (t.includes("what are") || t.includes("list") || t.includes("give"))) return { type: "LIST_PROJECTS" };

  const videoWords = ["video", "videos", "lesson", "lessons", "link", "youtube"];
  const troubleWords = ["not working", "doesn't work", "not responding", "stuck", "problem", "issue", "error"];

  if (videoWords.some((w) => t.includes(w)) || troubleWords.some((w) => t.includes(w))) {
    return { type: "PROJECT_VIDEOS" };
  }

  return { type: "GENERAL" };
}

function detectProject(text, projectNames, kb) {
  const t = text.toLowerCase();

  // exact project name match
  for (const p of projectNames) {
    if (t.includes(p.toLowerCase())) return p;
  }

  // alias match
  const projects = Array.isArray(kb?.projects) ? kb.projects : [];
  for (const p of projects) {
    if (Array.isArray(p.aliases)) {
      for (const a of p.aliases) {
        if (t.includes(String(a).toLowerCase())) return p.name;
      }
    }
  }

  return null;
}
function detectComponent(text, kb) {
  const t = text.toLowerCase();

  const list =
    Array.isArray(kb?.components)
      ? kb.components
      : Array.isArray(kb?.contents)
      ? kb.contents
      : [];

  for (const c of list) {
    const name = String(c.name || c.id || "").toLowerCase();
    if (name && t.includes(name)) {
      return {
        name: c.name || c.id,
        description: c.description || c.desc || "No description available."
      };
    }
  }

  return null;
}

function buildGroundedContext({ detectedProject, projectContext, canonicalPinsText, safetyText, lessonsForProject, intent, attachment }) {
  const lines = [];

  lines.push("Grounded Context (authoritative):");
  lines.push("");
  lines.push("Safety:");
  lines.push(safetyText);
  lines.push("");
  lines.push("Canonical Brain / Pins / Ports rules:");
  lines.push(canonicalPinsText);
  lines.push("");

  if (attachment && attachment.kind === "image") {
    lines.push("User attached an image.");
    lines.push("If the user asks about the image, ask what project this photo is from and what is not working.");
    lines.push("");
  }

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
        if (l.explainLine) lines.push(` Why: ${l.explainLine}`);
        for (const u of l.videoLinks || []) lines.push(` Link: ${u}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}

function extractProjectBlock(kb, projectName) {
  const projects = Array.isArray(kb?.projects) ? kb.projects : [];
 const components =
  Array.isArray(kb?.components)
    ? kb.components
    : Array.isArray(kb?.contents)
    ? kb.contents
    : [];

const componentMap = {};
for (const c of components) {
  const key = String(c.id || c.name).toLowerCase();
  componentMap[key] = {
    name: c.name || c.id,
    description: c.description || c.desc || ""
  };
}


  const project = projects.find(
    (p) =>
      p.name === projectName ||
      (Array.isArray(p.aliases) && p.aliases.some((a) => String(a).toLowerCase() === projectName.toLowerCase()))
  );

  if (!project) return null;

 let text = "";
text += `Project Name: ${project.name || project.project_name}\n`;
if (project.difficulty) text += `Difficulty: ${project.difficulty}\n`;
if (project.estimated_time) text += `Estimated Time: ${project.estimated_time}\n`;

if (project.description)
  text += `\nProject Description:\n${project.description}\n`;

if (Array.isArray(project.components_used)) {
  const readable = project.components_used.map((id) => {
    const key = String(id).toLowerCase();
    return componentMap[key]?.name || id;
  });
  text += `\nComponents Used:\n- ${readable.join("\n- ")}\n`;
}

if (Array.isArray(project.build_steps)) {
  text += `\nStep-by-Step Build:\n`;
  project.build_steps.forEach((s, i) => {
    text += `${i + 1}. ${s}\n`;
  });
}

if (project.working)
  text += `\nHow it Works:\n${project.working}\n`;

return text.trim();

}

function extractLessons(kb, projectName) {
  const projects = Array.isArray(kb?.projects) ? kb.projects : [];
  const project = projects.find(
    (p) =>
      p.name === projectName ||
      (Array.isArray(p.aliases) && p.aliases.some((a) => String(a).toLowerCase() === projectName.toLowerCase()))
  );

  if (!project || !Array.isArray(project.lessons)) return [];

  return project.lessons.map((l) => ({
  lessonName: l.lesson_name || l.LessonName,
  videoLinks:
    l.videoLinks ||
    l.video_url ||
    l.VideoLinks ||
    [],
  explainLine:
    l.explainLine ||
    l.what_this_helps ||
    l.description ||
    null,
}));

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
