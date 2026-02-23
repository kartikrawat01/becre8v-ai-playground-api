import multer from "multer";

const upload = multer({
  storage: multer.memoryStorage(), // keeps file in memory
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB safety
});

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";

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
  if (!list.length || list.some((a) => origin.startsWith(a))) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    return true;
  }
  return false;
}

/* -------------------- Chat Prompt Planner -------------------- */
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


/* -------------------- Handler -------------------- */
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
  await new Promise((resolve, reject) => {
    upload.single("file")(req, res, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });

  try {
    const body = req.body || {};
    // ---- Accept both payload shapes ----
    let message = "";

    if (typeof body.message === "string") {
      message = body.message.trim();
    } else if (typeof body.input === "string") {
      message = body.input.trim();
    }
    let history = [];

    if (typeof body.messages === "string") {
      try {
        history = JSON.parse(body.messages);
      } catch {
        history = [];
      }
    } else if (Array.isArray(body.messages)) {
      history = body.messages;
    }
    

    // Allow text-only, image-only, or both
    if (!message && !req.file) {
      return res.status(400).json({
        error: "Please type a message or upload an image to continue 🙂"
      });
    }

    const rawUserText = message || "Please look at the uploaded image and help me.";
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

    // --------- Build indexes ----------
    const {
  projectNames,
  projectsByName,
  lessonsByProject,
  canonicalPinsText,
  safetyText,
  kitOverview,
  componentsSummary,
  projectsSummary,
  componentsMap,
  supportConfig,
} = buildIndexes(kb);


    // --------- Intent detection (deterministic) ----------
  const rawIntent = detectIntent(rawUserText, projectNames, componentsMap);
const rawDetectedProject = detectProject(rawUserText, projectNames);
let detectedComponent = detectComponent(rawUserText, componentsMap);


    // --------- Handle KIT_OVERVIEW intent ----------
    if (rawIntent.type === "KIT_OVERVIEW") {
      return res.status(200).json({
        text: kitOverview,
        debug: {
          intent: rawIntent,
          kbMode: "deterministic_overview",
        },
      });
    }

    // --------- Handle COMPONENTS_LIST intent ----------
    if (rawIntent.type === "COMPONENTS_LIST") {
      const componentsText = formatComponentsList(componentsSummary);
      return res.status(200).json({
        text: componentsText,
        debug: {
          intent: rawIntent,
          kbMode: "deterministic_components",
        },
      });
    }

    // --------- Handle LIST_PROJECTS intent ----------
    if (rawIntent.type === "LIST_PROJECTS" && !rawDetectedProject) {
      return res.status(200).json({
        text:
          `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\n\nHere are the ${projectsSummary.totalCount} live projects:\n\n` +
          projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n") +
          "\n\nTell me which project you'd like to learn more about!",
        debug: {
          detectedProject: rawDetectedProject || null,
          intent: rawIntent,
          kbMode: "deterministic",
        },
      });
    }

    // --------- Handle PROJECT_VIDEOS intent ----------
    if (rawIntent.type === "PROJECT_VIDEOS") {
      if (!rawDetectedProject) {
        return res.status(200).json({
          text:
            "Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I'll share all the relevant lesson videos for that project.",
          debug: { detectedProject: null, intent: rawIntent },
        });
      }
      const videos = lessonsByProject[rawDetectedProject] || [];
      if (!videos.length) {
        return res.status(200).json({
          text:
            `I found the project "${rawDetectedProject}", but no lesson videos are mapped for it in the knowledge file.\n` +
            "If you share the lesson links for this project, I'll add them into the KB mapping.",
          debug: { detectedProject: rawDetectedProject, intent: rawIntent, videosFound: 0 },
        });
      }
      const out =
  `Lesson videos for ${rawDetectedProject}:\n\n` +
  videos
    .map((v, idx) => {

      const links = v.videoLinks || [];

const linksText = links.map((u) => `- ${u}`).join("\n");

return (
  `${idx + 1}. ${v.lessonName}\n` +
  `${v.explainLine ? `Why this helps: ${v.explainLine}\n` : ""}` +
  `Links:\n${linksText}`
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


    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res
        .status(500)
        .json({ error: "OPENAI_API_KEY is not set in env." });
    }

  
    let plannedUserText = rawUserText;
    try {
      // Only use the planner if there is NO file.
      if (!req.file) {
        plannedUserText = await planChatPrompt(rawUserText, apiKey);
      }
    } catch (plannerErr) {
      plannedUserText = rawUserText;
    }


const { lastProject, lastComponent } = resolveContextFromHistory(
  history,
  projectNames,
  componentsMap
);

const detectedProject =
  rawDetectedProject ||
  detectProject(plannedUserText, projectNames) ||
  lastProject;

detectedComponent =
  detectComponent(plannedUserText, componentsMap) || detectedComponent || lastComponent;

const intent = rawIntent;
const supportReason = detectSupportFailure({
  userText: rawUserText,
  intent,
  detectedProject,
  projectContext: detectedProject ? projectsByName[detectedProject] : null,
  detectedComponent,
  componentsMap
});


if (
  supportReason &&
  supportConfig?.enabled &&
  supportConfig.show_when?.includes(supportReason)
) {
  return res.status(200).json({
    text:
      `⚠️ **Need help?**\n\n${supportConfig.message}\n\n` +
      `📧 ${supportConfig.contact.email}\n` +
      `📞 ${supportConfig.contact.phone}\n` +
      `⏰ ${supportConfig.contact.hours}`,
    debug: {
      supportTriggered: true,
      supportReason,
      detectedProject,
      detectedComponent,
      intent
    }
  });
}


    // --------- Build grounded context for model ----------
    const projectContext = detectedProject
      ? projectsByName[detectedProject] || null
      : null;
    const groundedContext = buildGroundedContext({
      detectedProject,
      projectContext,
      lessonsByProject,
      canonicalPinsText,
      safetyText,
      kitOverview,
      componentsSummary,
      projectsSummary,
    });

    // --------- System Prompt ----------
    const systemPrompt = buildSystemPrompt(groundedContext);

    // --------- Prepare messages for OpenAI ----------
    const systemMsg = { role: "system", content: systemPrompt };
    const conversationMsgs = buildConversationHistory(history);

    // --- START OF IMAGE HANDLING FIX ---
    let userMsgContent = [{ type: "text", text: plannedUserText }];

    if (req.file) {
      const base64Image = req.file.buffer.toString('base64');
      userMsgContent.push({
        type: "image_url",
        image_url: { url: `data:${req.file.mimetype};base64,${base64Image}` }
      });
    }

    const userMsg = { role: "user", content: userMsgContent };
    // --- END OF IMAGE HANDLING FIX ---

    const messages = [systemMsg, ...conversationMsgs, userMsg];

    // --------- Call OpenAI ----------
    const r = await fetch(OPENAI_CHAT_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.7,
        max_tokens: 800,
        messages,
      }),
    });

    if (!r.ok) {
      const t = await r.text().catch(() => "");
      return res.status(500).json({
        error: "OpenAI API error",
        details: t.slice(0, 800),
      });
    }

    const data = await r.json();
   let assistantReply = data?.choices?.[0]?.message?.content?.trim() || "";

assistantReply = assistantReply.replace(/\*\*(.*?)\*\*/g, "$1");

assistantReply = assistantReply.replace(/^\s*#{1,6}\s*(.+)$/gm, "• $1");

assistantReply = assistantReply.replace(/\n{3,}/g, "\n\n");


    return res.status(200).json({
  text: assistantReply,
  debug: {
    detectedProject: detectedProject || null,
    detectedComponent: detectedComponent || null,
    intent,
    kbMode: "llm",
  },
});


  } catch (err) {
    console.error("Playground handler error:", err);
    return res.status(500).json({
      error: "Internal server error",
      message: String(err?.message || err).slice(0, 500),
    });
  }
}

function buildIndexes(kb) {
  const projectNames = extractProjectNames(kb);
  const projectsByName = {};
  const lessonsByProject = {};

  for (const pName of projectNames) {
    const projectBlock = extractProjectBlock(kb, pName);
    const lessons = extractLessons(kb, pName);
    projectsByName[pName] = projectBlock;
    lessonsByProject[pName] = lessons.sort(
      (a, b) => lessonRank(a.lessonName) - lessonRank(b.lessonName)
    );
  }

  const canonicalPinsText = extractCanonicalPins(kb);
  const safetyText = extractSafety(kb);
  const kitOverview = extractKitOverview(kb);
  const componentsSummary = extractComponentsSummary(kb);
  const projectsSummary = extractProjectsSummary(kb);
  const componentsMap = extractComponentsMap(kb);
const supportConfig = extractSupportConfig(kb);

return {
  projectNames,
  projectsByName,
  lessonsByProject,
  canonicalPinsText,
  safetyText,
  kitOverview,
  componentsSummary,
  projectsSummary,
  componentsMap,
  supportConfig,
};

}

/* -------------------- Intent Detection (IMPROVED) -------------------- */
function detectIntent(text, projectNames, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();

  // Kit overview
  if (
    /what.*(is|in|about|contains?).*kit/i.test(lower) ||
    /tell me about.*kit/i.test(lower) ||
    /kit.*overview/i.test(lower) ||
    (/what.*robocoders/i.test(lower) && !/brain/i.test(lower))
  &&
  !projectNames.some(proj => lower.includes(proj.toLowerCase())))    //changed
  {
    return { type: "KIT_OVERVIEW" };
  }

  // Components list
  if (
    /what.*(components?|parts?|pieces?).*kit/i.test(lower) ||
    /list.*components?/i.test(lower) ||
    /show.*components?/i.test(lower) ||
    /components?.*list/i.test(lower)
  ) {
    return { type: "COMPONENTS_LIST" };
  }

  // IMPROVED: Component info detection
  // Check if asking ABOUT a component (not a project that USES it)
  const componentKeywords = Object.keys(componentsMap).map(id => 
    componentsMap[id].name?.toLowerCase()
  ).filter(Boolean);
  
  const hasComponentMention = componentKeywords.some(comp => lower.includes(comp));
  const isAskingAboutComponent = hasComponentMention && (
    /what.*(is|does)/i.test(lower) ||
    /tell me about/i.test(lower) ||
    /how.*(works?|use)/i.test(lower) ||
    /explain/i.test(lower)
  );
  
  // CRITICAL FIX: Don't trigger component intent if asking about a PROJECT
  const isAskingAboutProject = projectNames.some(proj => 
    lower.includes(proj.toLowerCase())
  );
  
  if (isAskingAboutComponent && !isAskingAboutProject) {
    return { type: "COMPONENT_INFO" };
  }

  // List projects
  if (
    /(?:list|show|what are|tell me).*(?:projects?|modules?)/i.test(lower) ||
    /how many projects?/i.test(lower) ||
    /all projects?/i.test(lower)
  ) {
    return { type: "LIST_PROJECTS" };
  }

  // Project videos
  if (
    /(?:video|lesson|tutorial|how to (?:build|make|create)).*(?:project|module)/i.test(lower) ||
    /show.*videos?/i.test(lower) ||
    /(?:project|module).*(?:video|lesson)/i.test(lower)
  ) {
    return { type: "PROJECT_VIDEOS" };
  }

  return { type: "GENERAL" };
}

/* -------------------- Project Detection (IMPROVED) -------------------- */
function detectProject(text, projectNames) {
  const lower = String(text || "").toLowerCase().trim();
  
  // Create scoring system for best match
  let bestMatch = null;
  let bestScore = 0;
  
  for (const pName of projectNames) {
    const pLower = pName.toLowerCase();
    let score = 0;
    
    // Exact match gets highest score
    if (lower === pLower) {
      return pName;
    }
    
    // Full name included in query
    if (lower.includes(pLower)) {
      score = pLower.length * 2; // Longer matches score higher
    }
    
    // Check for word-by-word match (handles spaces and special chars)
    const pWords = pLower.split(/\s+/);
    const tWords = lower.split(/\s+/);
    const matchedWords = pWords.filter(w => tWords.includes(w));
    
    if (matchedWords.length > 0) {
      score += matchedWords.length * 3;
      
      // Bonus if all words match
      if (matchedWords.length === pWords.length) {
        score += 10;
      }
    }
    
    // Try simplified match (remove special chars)
    const pSimple = pLower.replace(/[^a-z0-9]+/g, "");
    const tSimple = lower.replace(/[^a-z0-9]+/g, "");
    if (tSimple.includes(pSimple)) {
      score += pSimple.length;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = pName;
    }
  }
  
  // Only return match if score is significant enough
  return bestScore >= 3 ? bestMatch : null;
}

/* -------------------- Component Detection (IMPROVED) -------------------- */
function detectComponent(text, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();
  
  let bestMatch = null;
  let bestScore = 0;
  
  // Check each component name
  for (const [componentId, componentData] of Object.entries(componentsMap)) {
    const componentName = (componentData.name || "").toLowerCase();
    if (!componentName) continue;
    
    let score = 0;
    
    // Exact match
    if (lower === componentName) {
      return componentId;
    }
    
    // Full component name in query
    if (lower.includes(componentName)) {
      score = componentName.length * 2;
    }
    
    // Check variations
    const variations = getComponentVariations(componentName);
    for (const variation of variations) {
      if (lower.includes(variation)) {
        score += variation.length;
      }
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = componentId;
    }
  }

return bestScore >= 2 ? bestMatch : null;

}
function resolveContextFromHistory(history, projectNames, componentsMap) {
  let lastProject = null;
  let lastComponent = null;

  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    if (!msg?.content) continue;

    const text = String(msg.content);

    if (!lastProject) {
      const p = detectProject(text, projectNames);
      if (p) lastProject = p;
    }

    if (!lastComponent) {
      const c = detectComponent(text, componentsMap);
      if (c) lastComponent = c;
    }

    if (lastProject && lastComponent) break;
  }

  return { lastProject, lastComponent };
}

function getComponentVariations(name) {
  const variations = [name];
  
  // Common variations
  if (name.includes("robocoders brain")) {
    variations.push("brain", "main board", "robocoders");
  }
  if (name.includes("ir sensor")) {
    variations.push("infrared", "ir", "proximity sensor");
  }
  if (name.includes("ldr")) {
    variations.push("light sensor", "light dependent resistor");
  }
  if (name.includes("potentiometer")) {
    variations.push("knob", "pot", "dial");
  }
  if (name.includes("servo motor")) {
    variations.push("servo");
  }
  if (name.includes("dc motor")) {
    variations.push("motor");
  }
  if (name.includes("rgb led")) {
    variations.push("rgb", "color led");
  }
  if (name.includes("keys pcb")) {
    variations.push("keys", "buttons", "button panel");
  }
  
  return variations;
}

/* -------------------- Context Building -------------------- */
function buildGroundedContext(opts) {
  const {
    detectedProject,
    projectContext,
    lessonsByProject,
    canonicalPinsText,
    safetyText,
    kitOverview,
    componentsSummary,
    projectsSummary,
  } = opts;

  const sections = [];

  // Kit overview
  sections.push("=== KIT OVERVIEW ===\n" + kitOverview);

  // Safety (ENHANCED)
  if (safetyText) {
    sections.push("=== SAFETY RULES ===\n" + safetyText + "\n\nIMPORTANT SAFETY NOTES:\n- It is SAFE to plug and unplug sensors and components while the Robocoders Brain is powered on.\n- The system uses low voltage (5V from USB), so there is no risk of electric shock.\n- However, always handle components gently to avoid physical damage.\n- Do not force connections - they should fit smoothly.");
  }

  // Canonical pins
  if (canonicalPinsText) {
    sections.push("=== PORT MAPPINGS ===\n" + canonicalPinsText);
  }

  // Components summary
  if (componentsSummary) {
    sections.push(
      "=== COMPONENTS SUMMARY ===\n" +
      `Total Components: ${componentsSummary.totalCount}\n` +
      `Categories available: ${Object.keys(componentsSummary.categories || {}).join(", ")}`
    );
  }

  // Projects summary
  if (projectsSummary) {
    sections.push(
      "=== PROJECTS SUMMARY ===\n" +
      `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\n` +
     `Available Projects: ${projectsSummary.projectList.join(", ")}`

    );
  }

  // Detected project context
  if (detectedProject && projectContext) {
    sections.push(
      `=== PROJECT: ${detectedProject} ===\n${projectContext}`
    );

    const lessons = lessonsByProject[detectedProject] || [];
    if (lessons.length) {
      sections.push(
        `=== LESSONS FOR ${detectedProject} ===\n` +
        lessons
          .map(
            (l) =>
              `- ${l.lessonName}\n  ${l.explainLine || ""}\n  Videos: ${l.videoLinks.join(", ")}`
          )
          .join("\n\n")
      );
    }
  }

  return sections.join("\n\n");
}

function buildSystemPrompt(groundedContext) {
  return `You are Be Cre8v AI, a helpful and encouraging assistant for the Robocoders Kit.

Your role:
- Help children aged 8-14 learn electronics, coding, and creative project building
- Provide clear, simple explanations suitable for kids
- Be encouraging, friendly, and enthusiastic
- Use the knowledge base information provided below to answer questions accurately
- If you don't know something, admit it honestly and suggest where to find the information

Important guidelines:
- Keep explanations simple and fun
- Use examples and analogies kids can relate to
- Encourage creativity and experimentation
- Always prioritize safety
- Be patient and supportive
- Use positive, empowering language
- When asked about SAFETY: It is SAFE to plug/unplug sensors while the Robocoders Brain is on (low voltage 5V USB)
- When asked about PROJECTS: Focus on the specific project mentioned, not on components used in that project
- When asked about COMPONENTS: Provide component-specific information only
- When asked about PROJECT COUNT: Always say "There are 50 projects, out of which 21 are live. One project per week shall be launched."

KNOWLEDGE BASE:
${groundedContext}

Remember: Base your answers on the knowledge base provided. If information is not in the knowledge base, say so honestly.`;
}

function buildConversationHistory(history) {
  const msgs = [];
  for (const h of history || []) {
    if (h?.role === "user" && h?.content) {
      msgs.push({ role: "user", content: String(h.content) });
    } else if (h?.role === "assistant" && h?.content) {
      msgs.push({ role: "assistant", content: String(h.content) });
    }
  }
  return msgs;
}

/* -------------------- Extraction Functions -------------------- */
function extractProjectNames(kb) {
  // ✅ ALWAYS use canonical full list (21 projects)
  return [
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
    "Candle Lamp",
  ];
}



function extractKitOverview(kb) {
  if (kb?.overview) {
    const o = kb.overview;
    const parts = [];
    if (o.kitName) parts.push(`Kit: ${o.kitName}`);
    if (o.description) parts.push(o.description);
    if (o.whatIsInside) parts.push(`\nWhat's Inside:\n${o.whatIsInside}`);
    if (o.keyFeatures) {
      parts.push(`\nKey Features:\n${o.keyFeatures.map(f => `- ${f}`).join("\n")}`);
    }
    if (o.totalProjects) parts.push(`\nTotal Projects: ${o.totalProjects}`);
    if (o.totalComponents) parts.push(`Total Components: ${o.totalComponents}`);
    if (o.ageRange) parts.push(`Age Range: ${o.ageRange}`);
    return parts.join("\n");
  }
  
  return "The Robocoders Kit is an educational electronics and coding kit for children aged 8-14. It includes 21 exciting projects and 92 components to learn physical computing, Visual Block Coding, and creative project building.";
}

function extractComponentsSummary(kb) {
  if (kb?.componentsSummary) {
    return kb.componentsSummary;
  }
  if (kb?.glossary?.components) {
    return {
      totalCount: Object.keys(kb.glossary.components).length,
      components: kb.glossary.components,
    };
  }
  return {
    totalCount: 92,
    description: "The kit includes various sensors, actuators, LEDs, motors, structural components, and craft materials.",
    categories: {},
  };
}

function extractProjectsSummary(kb) {
  const projectList = extractProjectNames(kb);
  return {
    totalCount: projectList.length,
    projectList,
  };
}
function extractSupportConfig(kb) {
  if (kb?.support?.enabled) {
    return kb.support;
  }
  return null;
}

// IMPROVED: Extract components map for component detection
function extractComponentsMap(kb) {
  const componentsMap = {};
  
  // Try glossary.components first
  if (kb?.glossary?.components) {
    return kb.glossary.components;
  }
  
  // Try pages with type "component"
  if (Array.isArray(kb?.pages)) {
    for (const page of kb.pages) {
      if (page?.type === "component" && page?.componentId && page?.componentName) {
        componentsMap[page.componentId] = {
          name: page.componentName,
          description: extractDescriptionFromText(page.text),
          id: page.componentId,
        };
      }
    }
  }
  
  return componentsMap;
}

function extractDescriptionFromText(text) {
  const match = String(text || "").match(/Description:\s*([^\n]+(?:\n(?!Component|Type|Usage)[^\n]+)*)/i);
  return match ? match[1].trim() : "";
}

// Format component info for display
function formatComponentInfo(componentData) {
  const lines = [];
  lines.push(`**${componentData.name}**\n`);
  if (componentData.description) {
    lines.push(componentData.description);
  }
  if (componentData.usage) {
    lines.push(`\n**How to use it:**\n${componentData.usage}`);
  }
  if (componentData.ports) {
    lines.push(`\n**Connects to:**\n${componentData.ports}`);
  }
  lines.push("\n\nWould you like to know which projects use this component?");
  return lines.join("\n");
}

function formatComponentsList(componentsSummary) {
  const lines = [];
  lines.push(`The Robocoders Kit contains ${componentsSummary.totalCount || 92} components including:\n`);
  
  if (componentsSummary.categories) {
    const cats = componentsSummary.categories;
    if (cats.controller) {
      lines.push(`**Controller (${cats.controller.count}):**`);
      lines.push(`- ${cats.controller.components.join(", ")}`);
      lines.push("");
    }
    if (cats.sensors) {
      lines.push(`**Sensors (${cats.sensors.count}):**`);
      lines.push(`- ${cats.sensors.components.join(", ")}`);
      lines.push("");
    }
    if (cats.actuators) {
      lines.push(`**Actuators (${cats.actuators.count}):**`);
      lines.push(`- ${cats.actuators.components.join(", ")}`);
      lines.push("");
    }
    if (cats.lights) {
      lines.push(`**Lights (${cats.lights.count}):**`);
      lines.push(`- ${cats.lights.components.join(", ")}`);
      lines.push("");
    }
    if (cats.power) {
      lines.push(`**Power (${cats.power.count}):**`);
      lines.push(`- ${cats.power.components.join(", ")}`);
      lines.push("");
    }
    if (cats.structural) {
      lines.push(`**Structural Components (${cats.structural.count}):**`);
      lines.push(`- ${cats.structural.description}`);
      lines.push("");
    }
    if (cats.craft) {
      lines.push(`**Craft Materials (${cats.craft.count}):**`);
      lines.push(`- ${cats.craft.components.join(", ")}`);
      lines.push("");
    }
    if (cats.mechanical) {
      lines.push(`**Mechanical Parts (${cats.mechanical.count}):**`);
      lines.push(`- ${cats.mechanical.components.join(", ")}`);
      lines.push("");
    }
    if (cats.wiring) {
      lines.push(`**Wiring (${cats.wiring.count}):**`);
      lines.push(`- ${cats.wiring.description}`);
      lines.push("");
    }
  }
  
  lines.push("\nWould you like to know more about any specific component?");
  return lines.join("\n");
}

function extractProjectBlock(kb, projectName) {
  // Try pages first
  if (Array.isArray(kb?.pages)) {
    const norm = (s) => s.toLowerCase();
    const pNorm = norm(projectName);
    
    for (const page of kb.pages) {
      if (page?.type === "project" && norm(page.projectName || "") === pNorm) {
        return sanitizeChunk(page.text || "");
      }
    }
    
    // Try text-based search
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
  
  // Try structured projects
  const p =
    (kb?.projects && kb.projects[projectName]) ||
    (Array.isArray(kb?.projects)
      ? kb.projects.find((x) => x?.name === projectName)
      : null);
  if (p) {
    const parts = [];
    if (p.description) parts.push(p.description);
    if (p.componentsUsed)
      parts.push("Components Used:\n" + p.componentsUsed.join("\n"));
    if (p.connections)
      parts.push("Connections:\n" + p.connections.join("\n"));
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

  const links = Array.isArray(l.video_url)
  ? uniq(l.video_url.map(u => String(u).trim()))
  : [String(l.video_url || "").trim()];

  links.forEach((link, i) => {
    lessons.push({
      lessonName:
        links.length > 1
          ? `${l.lesson_name} - Part ${i + 1}`
          : l.lesson_name,
      videoLinks: [link],
      explainLine: "",
    });
  });
}


    // ⛔ IMPORTANT: agar structured lessons mil gaye
    // text-based parsing NAHI chalega
    return dedupeLessons(lessons);
  }

  // 🔽 FALLBACK: text-based extraction (ONLY if JSON lessons missing)
  const lessons = [];

  if (Array.isArray(kb?.pages)) {
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : ""));
    const p = projectName.toLowerCase();
    let inProject = false;

    for (let i = 0; i < pages.length; i++) {
      const txt = pages[i];
      const low = txt.toLowerCase();

      if (
        low.includes("project:") ||
        low.includes("project :") ||
        low.includes("project name")
      ) {
        inProject = low.includes(p);
      }

      if (!inProject) continue;

      const blocks = txt.split(
        /(Lesson ID\s*[:\-]|Build\s*\d+|Coding\s*Part\s*\d+)/i
      );

      for (let b = 1; b < blocks.length; b++) {
        const block = blocks[b];

        const lessonName =
          matchLine(block, /Lesson Name\s*(?:\(Canonical\))?\s*:\s*([^\n]+)/i) ||
          matchLine(block, /(Build\s*\d+)/i) ||
          matchLine(block, /(Coding\s*Part\s*\d+)/i) ||
          "";

        const links =
          block.match(/https?:\/\/[^\s\]\)\}\n]+/gi) || [];

        const explainLine =
          matchLine(
            block,
            /What this lesson helps with.*?:\s*([\s\S]*?)(?:When the AI|$)/i
          ) || "";

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

/* -------------------- Utility Functions -------------------- */
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
    const key =
      (l.lessonName || "").toLowerCase() +
      "::" +
      (l.videoLinks || []).join(",");

    if (!key || seen.has(key)) continue;

    seen.add(key);
    out.push(l);
  }

  return out;
}
function detectSupportFailure({
  userText,
  detectedProject,
  projectContext,
  detectedComponent
}) {
  const lower = String(userText || "").toLowerCase();


  if (/contact|customer support|support team|call|email|phone|helpline/i.test(lower)) {
    return "USER_REQUESTED_SUPPORT";
  }


  if (
    /missing|not in kit|not included|lost|component missing|nahi mila|gayab/i.test(lower)
  ) {
    return "PART_MISSING";
  }

  if (
    /broken|damaged|burnt|burned|melted|smoke|dead|faulty|cracked|not powering/i.test(lower)
  ) {
    return "HARDWARE_DAMAGED";
  }

  if (
    /sensor|motor|board|wire|led|wheel|fan|blade|battery|switch|sheet/i.test(lower) &&
    !detectedComponent &&
    !detectedProject
  ) {
    return "UNKNOWN_COMPONENT";
  }


  if (detectedProject && !projectContext) {
    return "PROJECT_NOT_IN_KB";
  }

  return null;
}
