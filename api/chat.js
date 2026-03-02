// =============================================================================
// Be Cre8v AI Backend — Full RAG Implementation with Pinecone
// =============================================================================
// Architecture:
//   1. Deterministic fast-paths (KIT_OVERVIEW, LIST_PROJECTS, PROJECT_VIDEOS,
//      COMPONENTS_LIST) — answered instantly without any LLM call.
//   2. Support-trigger detection — returns contact card when needed.
//   3. RAG pipeline for everything else:
//        a. Embed user query via OpenAI text-embedding-3-small
//        b. Query Pinecone for top-K relevant chunks
//        c. Inject retrieved chunks into system prompt
//        d. Call GPT-4o-mini with grounded context
//   4. Vision support — image attachments still work as before.
//   5. Conversation history — still forwarded to OpenAI.
//   6. Prompt planner — still runs before the main LLM call.
// =============================================================================
// Required env vars (set in Vercel dashboard):
//   OPENAI_API_KEY
//   PINECONE_API_KEY
//   PINECONE_INDEX_HOST   ← full host URL e.g. https://your-index-xxxx.svc.us-east1-gcp.pinecone.io
//   KNOWLEDGE_URL         ← still used for deterministic fast-paths (your existing JSON)
//   ALLOWED_ORIGIN        ← comma-separated allowed origins
// =============================================================================

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings";
const EMBED_MODEL = "text-embedding-3-small";

// ---------------------------------------------------------------------------
// CORS helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Prompt Planner
// ---------------------------------------------------------------------------
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
  return data?.choices?.[0]?.message?.content?.trim() || String(userText || "").trim();
}

// ---------------------------------------------------------------------------
// RAG — Pinecone query
// ---------------------------------------------------------------------------

/**
 * Embed a text string using OpenAI text-embedding-3-small.
 * Returns a float array (1536-dim).
 */
async function embedText(text, apiKey) {
  const r = await fetch(OPENAI_EMBED_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: String(text || "").trim(),
    }),
  });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error("Embedding error: " + t.slice(0, 800));
  }
  const data = await r.json();
  return data.data[0].embedding;
}

/**
 * Query Pinecone for the top-K most semantically similar chunks.
 * Returns an array of { text, metadata } objects.
 *
 * Pinecone REST API — query endpoint:
 *   POST https://<INDEX_HOST>/query
 *   Headers: Api-Key: <PINECONE_API_KEY>
 *   Body: { vector, topK, includeMetadata: true }
 */
async function queryPinecone(queryEmbedding, topK = 6) {
  const host = process.env.PINECONE_INDEX_HOST;
  const apiKey = process.env.PINECONE_API_KEY;

  if (!host || !apiKey) {
    throw new Error("PINECONE_INDEX_HOST or PINECONE_API_KEY not set in env.");
  }

  const r = await fetch(`${host}/query`, {
    method: "POST",
    headers: {
      "Api-Key": apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
    }),
  });

  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error("Pinecone query error: " + t.slice(0, 800));
  }

  const data = await r.json();

  // Each match has: id, score, metadata: { text, type, projectName, ... }
  return (data.matches || []).map((m) => ({
    text: m.metadata?.text || "",
    type: m.metadata?.type || "general",
    projectName: m.metadata?.projectName || null,
    score: m.score,
  }));
}

// ---------------------------------------------------------------------------
// Main Handler
// ---------------------------------------------------------------------------
export default async function handler(req, res) {
  const origin = req.headers.origin || "";

  // 1) Preflight
  if (req.method === "OPTIONS") {
    allow(res, origin);
    return res.status(204).end();
  }

  // 2) Origin allowlist
  if (!allow(res, origin)) {
    return res.status(403).json({ error: "Forbidden origin", origin, allowed: origins() });
  }

  // 3) Only POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  try {
    // ---- Parse body ----
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

    // Allow empty message if image attached
    if ((typeof message !== "string" || !message.trim()) && !attachment) {
      return res.status(400).json({ error: "Missing message or image input." });
    }

    const rawUserText =
      String(message || "").trim() ||
      (attachment ? "Analyze the uploaded image and describe what you see in detail." : "");

    // ---- Load KB (still needed for deterministic fast-paths) ----
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

    // ---- Build indexes (for deterministic paths) ----
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

    // ---- Intent + entity detection ----
    const rawIntent = detectIntent(rawUserText, projectNames, componentsMap);
    const rawDetectedProject = detectProject(rawUserText, projectNames);
    let detectedComponent = detectComponent(rawUserText, componentsMap);

    // ================================================================
    // DETERMINISTIC FAST-PATHS (no RAG, no LLM)
    // ================================================================

    // KIT_OVERVIEW
    if (rawIntent.type === "KIT_OVERVIEW") {
      return res.status(200).json({
        text: kitOverview,
        debug: { intent: rawIntent, kbMode: "deterministic_overview" },
      });
    }

    // COMPONENTS_LIST
    if (rawIntent.type === "COMPONENTS_LIST") {
      return res.status(200).json({
        text: formatComponentsList(componentsSummary),
        debug: { intent: rawIntent, kbMode: "deterministic_components" },
      });
    }

    // LIST_PROJECTS
    if (rawIntent.type === "LIST_PROJECTS" && !rawDetectedProject) {
      return res.status(200).json({
        text:
          `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\n\nHere are the ${projectsSummary.totalCount} live projects:\n\n` +
          projectNames.map((p, i) => `${i + 1}. ${p}`).join("\n") +
          "\n\nTell me which project you'd like to learn more about!",
        debug: { detectedProject: rawDetectedProject || null, intent: rawIntent, kbMode: "deterministic" },
      });
    }

    // PROJECT_VIDEOS
    if (rawIntent.type === "PROJECT_VIDEOS") {
      if (!rawDetectedProject) {
        return res.status(200).json({
          text: "Tell me the project/module name (example: Mood Lamp, Coin Counter, Game Controller), and I'll share all the relevant lesson videos for that project.",
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
        debug: { detectedProject: rawDetectedProject, intent: rawIntent, lessonsReturned: videos.length, kbMode: "deterministic" },
      });
    }

    // ================================================================
    // API KEY CHECK (needed for RAG + LLM path)
    // ================================================================
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "OPENAI_API_KEY is not set in env." });
    }

    // ================================================================
    // PROMPT PLANNER
    // ================================================================
    let plannedUserText = rawUserText;
    try {
      plannedUserText = await planChatPrompt(rawUserText, apiKey);
    } catch (_) {
      plannedUserText = rawUserText;
    }

    // ---- Resolve context from conversation history ----
    const { lastProject, lastComponent } = resolveContextFromHistory(history, projectNames, componentsMap);

    const detectedProject =
      rawDetectedProject ||
      detectProject(plannedUserText, projectNames) ||
      lastProject;

    detectedComponent =
      detectComponent(plannedUserText, componentsMap) || detectedComponent || lastComponent;

    const intent = rawIntent;

    // ================================================================
    // SUPPORT TRIGGER — check before RAG
    // ================================================================
    const supportReason = detectSupportFailure({
      userText: rawUserText,
      intent,
      detectedProject,
      projectContext: detectedProject ? projectsByName[detectedProject] : null,
      detectedComponent,
      componentsMap,
    });

    if (supportReason && supportConfig?.enabled && supportConfig.show_when?.includes(supportReason)) {
      return res.status(200).json({
        text:
          `⚠️ Need help?\n\n${supportConfig.message}\n\n` +
          `📧 ${supportConfig.contact.email}\n` +
          `📞 ${supportConfig.contact.phone}\n` +
          `⏰ ${supportConfig.contact.hours}`,
        debug: { supportTriggered: true, supportReason, detectedProject, detectedComponent, intent },
      });
    }

    // ================================================================
    // RAG — Embed query → Query Pinecone → Retrieve chunks
    // ================================================================
    let ragContext = "";
    let ragChunks = [];

    try {
      // Use the planned (improved) query for better semantic matching
      const queryText = plannedUserText || rawUserText;
      const queryEmbedding = await embedText(queryText, apiKey);
      ragChunks = await queryPinecone(queryEmbedding, 6);

      if (ragChunks.length > 0) {
        ragContext =
          "=== RETRIEVED KNOWLEDGE (from vector search) ===\n" +
          ragChunks
            .map((c, i) => `[Chunk ${i + 1} | type: ${c.type}${c.projectName ? ` | project: ${c.projectName}` : ""}]\n${c.text}`)
            .join("\n\n---\n\n");
      }
    } catch (ragErr) {
      // RAG failure is non-fatal — fall back to deterministic context
      console.error("RAG error (falling back to deterministic context):", ragErr.message);
    }

    // ================================================================
    // GROUNDED CONTEXT — combine RAG chunks + deterministic context
    // (Deterministic context ensures critical data like safety rules,
    //  port mappings, and the specific project block are always present)
    // ================================================================
    const projectContext = detectedProject
      ? projectsByName[detectedProject] || null
      : null;

    const deterministicContext = buildGroundedContext({
      detectedProject,
      projectContext,
      lessonsByProject,
      canonicalPinsText,
      safetyText,
      kitOverview,
      componentsSummary,
      projectsSummary,
    });

    // RAG context goes first (most relevant), deterministic context as fallback
    const fullContext = ragContext
      ? ragContext + "\n\n" + deterministicContext
      : deterministicContext;

    // ================================================================
    // SYSTEM PROMPT
    // ================================================================
    const systemPrompt = buildSystemPrompt(fullContext);

    // ================================================================
    // BUILD MESSAGES FOR OPENAI
    // ================================================================
    const systemMsg = { role: "system", content: systemPrompt };
    const conversationMsgs = buildConversationHistory(history);

    // User message (with optional Vision attachment)
    let userContent = [];
    if (plannedUserText && plannedUserText.trim()) {
      userContent.push({ type: "text", text: plannedUserText });
    }
    if (attachment) {
      let imageUrl = attachment;
      if (!attachment.startsWith("data:")) {
        imageUrl = `data:image/png;base64,${attachment}`;
      }
      userContent.push({ type: "image_url", image_url: { url: imageUrl } });
    }

    const userMsg = { role: "user", content: userContent };
    const messages = [systemMsg, ...conversationMsgs, userMsg];

    console.log("RAG chunks retrieved:", ragChunks.length);
    console.log("Detected project:", detectedProject);
    console.log("Intent:", intent.type);

    // ================================================================
    // CALL OPENAI
    // ================================================================
    const r = await fetch(OPENAI_CHAT_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.7,
        max_tokens: 1000,
        messages,
      }),
    });

    if (!r.ok) {
      const t = await r.text().catch(() => "");
      return res.status(500).json({ error: "OpenAI API error", details: t.slice(0, 800) });
    }

    const data = await r.json();
    let assistantReply = data?.choices?.[0]?.message?.content?.trim() || "";

    // Clean up markdown artifacts
    assistantReply = assistantReply.replace(/\*\*(.*?)\*\*/g, "$1");
    assistantReply = assistantReply.replace(/^\s*#{1,6}\s*(.+)$/gm, "• $1");
    assistantReply = assistantReply.replace(/\n{3,}/g, "\n\n");

    return res.status(200).json({
      text: assistantReply,
      debug: {
        detectedProject: detectedProject || null,
        detectedComponent: detectedComponent || null,
        intent,
        kbMode: "rag+llm",
        ragChunksRetrieved: ragChunks.length,
        ragChunkTypes: ragChunks.map((c) => c.type),
      },
    });
  } catch (err) {
    console.error("Handler error:", err);
    return res.status(500).json({
      error: "Internal server error",
      message: String(err?.message || err).slice(0, 500),
    });
  }
}

// =============================================================================
// buildIndexes — same as original, used for deterministic fast-paths
// =============================================================================
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

// =============================================================================
// Intent Detection — unchanged from original
// =============================================================================
function detectIntent(text, projectNames, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();

  // Kit overview
  if (
    (
      /what.*(is|in|about|contains?).*kit/i.test(lower) ||
      /tell me about.*kit/i.test(lower) ||
      /kit.*overview/i.test(lower) ||
      (/what.*robocoders/i.test(lower) && !/brain/i.test(lower))
    ) &&
    !projectNames.some((proj) => lower.includes(proj.toLowerCase()))
  ) {
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

  // Component info
  const componentKeywords = Object.keys(componentsMap)
    .map((id) => componentsMap[id].name?.toLowerCase())
    .filter(Boolean);
  const hasComponentMention = componentKeywords.some((comp) => lower.includes(comp));
  const isAskingAboutComponent =
    hasComponentMention &&
    (/what.*(is|does)/i.test(lower) ||
      /tell me about/i.test(lower) ||
      /how.*(works?|use)/i.test(lower) ||
      /explain/i.test(lower));
  const isAskingAboutProject = projectNames.some((proj) =>
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

// =============================================================================
// Project Detection — unchanged from original
// =============================================================================
function detectProject(text, projectNames) {
  const lower = String(text || "").toLowerCase().trim();
  let bestMatch = null;
  let bestScore = 0;

  for (const pName of projectNames) {
    const pLower = pName.toLowerCase();
    let score = 0;

    if (lower === pLower) return pName;

    if (lower.includes(pLower)) {
      score = pLower.length * 2;
    }

    const pWords = pLower.split(/\s+/);
    const tWords = lower.split(/\s+/);
    const matchedWords = pWords.filter((w) => tWords.includes(w));
    if (matchedWords.length > 0) {
      score += matchedWords.length * 3;
      if (matchedWords.length === pWords.length) score += 10;
    }

    const pSimple = pLower.replace(/[^a-z0-9]+/g, "");
    const tSimple = lower.replace(/[^a-z0-9]+/g, "");
    if (tSimple.includes(pSimple)) score += pSimple.length;

    if (score > bestScore) {
      bestScore = score;
      bestMatch = pName;
    }
  }

  return bestScore >= 3 ? bestMatch : null;
}

// =============================================================================
// Component Detection — unchanged from original
// =============================================================================
function detectComponent(text, componentsMap) {
  const lower = String(text || "").toLowerCase().trim();
  let bestMatch = null;
  let bestScore = 0;

  for (const [componentId, componentData] of Object.entries(componentsMap)) {
    const componentName = (componentData.name || "").toLowerCase();
    if (!componentName) continue;
    let score = 0;

    if (lower === componentName) return componentId;

    if (lower.includes(componentName)) score = componentName.length * 2;

    const variations = getComponentVariations(componentName);
    for (const variation of variations) {
      if (lower.includes(variation)) score += variation.length;
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

// =============================================================================
// Context Building — unchanged from original
// =============================================================================
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

  sections.push("=== KIT OVERVIEW ===\n" + kitOverview);

  if (safetyText) {
    sections.push(
      "=== SAFETY RULES ===\n" +
        safetyText +
        "\n\nIMPORTANT SAFETY NOTES:\n" +
        "- It is SAFE to plug and unplug sensors and components while the Robocoders Brain is powered on.\n" +
        "- The system uses low voltage (5V from USB), so there is no risk of electric shock.\n" +
        "- However, always handle components gently to avoid physical damage.\n" +
        "- Do not force connections - they should fit smoothly."
    );
  }

  if (canonicalPinsText) {
    sections.push("=== PORT MAPPINGS ===\n" + canonicalPinsText);
  }

  if (componentsSummary) {
    sections.push(
      "=== COMPONENTS SUMMARY ===\n" +
        `Total Components: ${componentsSummary.totalCount}\n` +
        `Categories available: ${Object.keys(componentsSummary.categories || {}).join(", ")}`
    );
  }

  if (projectsSummary) {
    sections.push(
      "=== PROJECTS SUMMARY ===\n" +
        `There are 50 projects, out of which ${projectsSummary.totalCount} are live. One project per week shall be launched.\n` +
        `Available Projects: ${projectsSummary.projectList.join(", ")}`
    );
  }

  if (detectedProject && projectContext) {
    sections.push(`=== PROJECT: ${detectedProject} ===\n${projectContext}`);
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

// =============================================================================
// Extraction Functions — all unchanged from original
// =============================================================================
function extractProjectNames(kb) {
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
  return { totalCount: projectList.length, projectList };
}

function extractSupportConfig(kb) {
  if (kb?.support?.enabled) return kb.support;
  return null;
}

function extractComponentsMap(kb) {
  const componentsMap = {};
  if (kb?.glossary?.components) return kb.glossary.components;
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
  const match = String(text || "").match(
    /Description:\s*([^\n]+(?:\n(?!Component|Type|Usage)[^\n]+)*)/i
  );
  return match ? match[1].trim() : "";
}

function extractProjectBlock(kb, projectName) {
  if (Array.isArray(kb?.pages)) {
    const norm = (s) => s.toLowerCase();
    const pNorm = norm(projectName);
    for (const page of kb.pages) {
      if (page?.type === "project" && norm(page.projectName || "") === pNorm) {
        return sanitizeChunk(page.text || "");
      }
    }
    const pages = kb.pages.map((pg) => (pg?.text ? String(pg.text) : "")).filter(Boolean);
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
      const links = Array.isArray(l.video_url)
        ? uniq(l.video_url.map((u) => String(u).trim()))
        : [String(l.video_url || "").trim()];
      links.forEach((link, i) => {
        lessons.push({
          lessonName: links.length > 1 ? `${l.lesson_name} - Part ${i + 1}` : l.lesson_name,
          videoLinks: [link],
          explainLine: "",
        });
      });
    }
    return dedupeLessons(lessons);
  }

  // Fallback: text-based extraction
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
      const blocks = txt.split(/(Lesson ID\s*[:\-]|Build\s*\d+|Coding\s*Part\s*\d+)/i);
      for (let b = 1; b < blocks.length; b++) {
        const block = blocks[b];
        const lessonName =
          matchLine(block, /Lesson Name\s*(?:\(Canonical\))?\s*:\s*([^\n]+)/i) ||
          matchLine(block, /(Build\s*\d+)/i) ||
          matchLine(block, /(Coding\s*Part\s*\d+)/i) ||
          "";
        const links = block.match(/https?:\/\/[^\s\]\)\}\n]+/gi) || [];
        const explainLine =
          matchLine(block, /What this lesson helps with.*?:\s*([\s\S]*?)(?:When the AI|$)/i) || "";
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

// =============================================================================
// Formatting Helpers — unchanged from original
// =============================================================================
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

// =============================================================================
// Support Detection — unchanged from original
// =============================================================================
function detectSupportFailure({ userText, detectedProject, projectContext, detectedComponent }) {
  const lower = String(userText || "").toLowerCase();

  if (/contact|customer support|support team|call|email|phone|helpline/i.test(lower)) {
    return "USER_REQUESTED_SUPPORT";
  }
  if (/missing|not in kit|not included|lost|component missing|nahi mila|gayab/i.test(lower)) {
    return "PART_MISSING";
  }
  if (/broken|damaged|burnt|burned|melted|smoke|dead|faulty|cracked|not powering/i.test(lower)) {
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

// =============================================================================
// Utility Functions — unchanged from original
// =============================================================================
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
      (l.lessonName || "").toLowerCase() + "::" + (l.videoLinks || []).join(",");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(l);
  }
  return out;
}
