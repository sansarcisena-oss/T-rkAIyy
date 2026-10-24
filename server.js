// TürkAI Next — server.js
// Konuşma API'si ve çoklu model desteği

import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";
import "dotenv/config";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT || 3000);
const PUBLIC_DIR = path.join(__dirname, "public");
const DATA_DIR = path.join(__dirname, "data");
const LOG_DIR = path.join(DATA_DIR, "logs");
const CHAT_DIR = path.join(DATA_DIR, "chats");

app.set("trust proxy", 1);
app.disable("x-powered-by");

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https:"]
    }
  }
}));

app.use(cors({
  origin: process.env.CORS_ORIGIN || true,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: false, limit: "100kb" }));

app.use((req, res, next) => {
  req.requestId = crypto.randomUUID();
  res.setHeader("X-Request-ID", req.requestId);
  res.setHeader("Cache-Control", "no-store");
  next();
});

const generalLimiter = rateLimit({
  windowMs: 60_000,
  limit: 120,
  standardHeaders: "draft-8",
  legacyHeaders: false
});

const chatLimiter = rateLimit({
  windowMs: 60_000,
  limit: 12,
  standardHeaders: "draft-8",
  legacyHeaders: false,
  message: {
    error: "Çok fazla istek gönderildi.",
    retry: "Biraz bekleyip tekrar dene."
  }
});

app.use(generalLimiter);

const MODEL_CONFIG = {
  deepseek: {
    id: "deepseek-ai/deepseek-v4.1-flash",
    name: "DeepSeek V4.1 Flash",
    description: "Genel sohbet ve kodlama",
    enabled: true
  },
  diffusiongemma: {
    id: "google/diffusiongemma-26b-a4b-it",
    name: "Google DiffusionGemma",
    description: "Deneysel model",
    enabled: true
  }
};

const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "deepseek";

let client = null;
let server = null;
let shuttingDown = false;

const stats = {
  startedAt: Date.now(),
  requests: 0,
  successfulChats: 0,
  failedChats: 0,
  totalTokens: 0,
  activeRequests: 0
};

const activeControllers = new Set();

function ensureDirectories() {
  for (const dir of [DATA_DIR, LOG_DIR, CHAT_DIR]) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function logEvent(type, details = {}) {
  const entry = {
    time: new Date().toISOString(),
    type,
    ...details
  };

  const file = path.join(
    LOG_DIR,
    `${new Date().toISOString().slice(0, 10)}.jsonl`
  );

  try {
    fs.appendFileSync(file, JSON.stringify(entry) + "\n");
  } catch {
    console.error("Log yazılamadı:", type);
  }
}

function getClient() {
  const apiKey = process.env.NVIDIA_API_KEY;

  if (!apiKey) {
    throw new Error("NVIDIA_API_KEY yapılandırılmamış.");
  }

  if (!client) {
    client = new OpenAI({
      apiKey,
      baseURL: "https://integrate.api.nvidia.com/v1",
      timeout: 90_000,
      maxRetries: 1
    });
  }

  return client;
}

function getModel(key) {
  const model = MODEL_CONFIG[key];

  if (!model || !model.enabled) {
    return null;
  }

  return model;
}

function cleanText(value, maxLength = 12000) {
  if (typeof value !== "string") return "";
  return value.trim().slice(0, maxLength);
}

function validateHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .slice(-20)
    .filter(item =>
      item &&
      ["user", "assistant"].includes(item.role) &&
      typeof item.content === "string"
    )
    .map(item => ({
      role: item.role,
      content: cleanText(item.content, 6000)
    }))
    .filter(item => item.content.length > 0);
}

function safeErrorMessage(error) {
  const status = error?.status;

  if (status === 401 || status === 403) {
    return "API kimlik doğrulaması başarısız. Sunucu anahtarını kontrol et.";
  }

  if (status === 429) {
    return "Model kullanım sınırına ulaşıldı. Daha sonra tekrar dene.";
  }

  if (status === 404) {
    return "Model bulunamadı veya bu endpoint üzerinden kullanılamıyor.";
  }

  if (error?.name === "AbortError") {
    return "İstek zaman aşımına uğradı.";
  }

  return "Model yanıt veremedi. Sunucu günlüklerini kontrol et.";
}

function makeSystemPrompt() {
  return [
    "Sen TürkAI Next adlı yapay zekâ asistanısın.",
    "Kullanıcıyla öncelikle Türkçe konuş.",
    "Sorulara açık, doğru ve faydalı cevaplar ver.",
    "Bilmediğin bilgileri kesinmiş gibi sunma.",
    "Güncel internet araştırması yapamadığında bunu açıkça belirt.",
    "Kod verirken kodun nasıl çalıştırılacağını gerektiğinde açıkla.",
    "Kullanıcının API anahtarlarını veya gizli bilgilerini isteme.",
    "Kullanıcının önceki konuşmaları yalnızca gönderilen geçmişte bulunuyorsa onları kullan."
  ].join(" ");
}

// -----------------------------
// TEMEL SAYFA VE DURUM UÇLARI
// -----------------------------

app.get("/", (req, res) => {
  const indexPath = path.join(PUBLIC_DIR, "index.html");

  if (fs.existsSync(indexPath)) {
    return res.sendFile(indexPath);
  }

  res.status(200).json({
    name: "TürkAI Next",
    status: "online",
    message: "public/index.html dosyası bulunamadı.",
    endpoints: ["/health", "/api/status", "/api/models", "/api/chat"]
  });
});

app.use(express.static(PUBLIC_DIR, {
  index: "index.html",
  maxAge: "1h"
}));

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "TürkAI Next",
    timestamp: new Date().toISOString(),
    uptimeSeconds: Math.floor(process.uptime())
  });
});

app.get("/api", (req, res) => {
  res.json({
    name: "TürkAI Next API",
    version: "1.0.0",
    endpoints: {
      health: "GET /health",
      status: "GET /api/status",
      models: "GET /api/models",
      chat: "POST /api/chat",
      modelTest: "POST /api/models/:key/test"
    }
  });
});

app.get("/api/status", (req, res) => {
  res.json({
    name: "TürkAI Next",
    status: shuttingDown ? "shutting_down" : "online",
    uptimeSeconds: Math.floor(process.uptime()),
    requests: stats.requests,
    successfulChats: stats.successfulChats,
    failedChats: stats.failedChats,
    totalTokens: stats.totalTokens,
    activeRequests: stats.activeRequests,
    apiConfigured: Boolean(process.env.NVIDIA_API_KEY),
    timestamp: new Date().toISOString()
  });
});

// -----------------------------
// MODEL LİSTESİ VE BİLGİLERİ
// -----------------------------

app.get("/api/models", (req, res) => {
  res.json({
    defaultModel: DEFAULT_MODEL,
    models: Object.entries(MODEL_CONFIG).map(([key, model]) => ({
      key,
      id: model.id,
      name: model.name,
      description: model.description,
      enabled: model.enabled
    }))
  });
});

app.get("/api/models/:key", (req, res) => {
  const model = getModel(req.params.key);

  if (!model) {
    return res.status(404).json({
      error: "Model bulunamadı."
    });
  }

  res.json({
    key: req.params.key,
    ...model
  });
});

// -----------------------------
// ANA KONUŞMA SİSTEMİ
// -----------------------------

app.post("/api/chat", chatLimiter, async (req, res) => {
  stats.requests++;
  stats.activeRequests++;

  const controller = new AbortController();
  activeControllers.add(controller);

  let finished = false;

  const finishRequest = () => {
    if (finished) return;
    finished = true;
    stats.activeRequests = Math.max(0, stats.activeRequests - 1);
    activeControllers.delete(controller);
  };

  req.on("aborted", () => controller.abort());

  try {
    if (shuttingDown) {
      return res.status(503).json({
        error: "Sunucu yeniden başlatılıyor."
      });
    }

    const message = cleanText(req.body?.message, 12000);
    const modelKey = cleanText(req.body?.model || DEFAULT_MODEL, 60);
    const history = validateHistory(req.body?.history);

    if (!message) {
      return res.status(400).json({
        error: "Mesaj boş olamaz."
      });
    }

    const model = getModel(modelKey);

    if (!model) {
      return res.status(400).json({
        error: "Seçilen model desteklenmiyor.",
        availableModels: Object.keys(MODEL_CONFIG)
      });
    }

    if (!process.env.NVIDIA_API_KEY) {
      return res.status(503).json({
        error: "NVIDIA API anahtarı sunucuda ayarlanmamış."
      });
    }

    const messages = [
      { role: "system", content: makeSystemPrompt() },
      ...history,
      { role: "user", content: message }
    ];

    logEvent("chat_started", {
      requestId: req.requestId,
      model: model.id
    });

    const completion = await getClient().chat.completions.create(
      {
        model: model.id,
        messages,
        temperature: 0.6,
        max_tokens: 2048,
        stream: false
      },
      {
        signal: controller.signal
      }
    );

    const reply = completion.choices?.[0]?.message?.content;

    if (typeof reply !== "string" || !reply.trim()) {
      throw new Error("Model boş yanıt döndürdü.");
    }

    stats.successfulChats++;

    const usage = completion.usage || {};
    stats.totalTokens += Number(usage.total_tokens || 0);

    logEvent("chat_completed", {
      requestId: req.requestId,
      model: model.id,
      tokens: Number(usage.total_tokens || 0)
    });

    return res.json({
      success: true,
      reply: reply.trim(),
      model: {
        key: modelKey,
        id: model.id,
        name: model.name
      },
      usage: {
        promptTokens: usage.prompt_tokens ?? null,
        completionTokens: usage.completion_tokens ?? null,
        totalTokens: usage.total_tokens ?? null
      },
      requestId: req.requestId
    });
  } catch (error) {
    stats.failedChats++;

    logEvent("chat_error", {
      requestId: req.requestId,
      status: error?.status || null,
      errorName: error?.name || "Error"
    });

    if (!res.headersSent) {
      const status =
        error?.status === 429 ? 429 :
        error?.status === 401 || error?.status === 403 ? 502 :
        error?.name === "AbortError" ? 408 : 502;

      res.status(status).json({
        error: safeErrorMessage(error),
        requestId: req.requestId
      });
    }
  } finally {
    finishRequest();
  }
});

// -----------------------------
// MODEL BAĞLANTI TESTİ
// -----------------------------

app.post("/api/models/:key/test", async (req, res) => {
  const modelKey = cleanText(req.params.key, 60);
  const model = getModel(modelKey);

  if (!model) {
    return res.status(404).json({
      success: false,
      error: "Model bulunamadı."
    });
  }

  if (!process.env.NVIDIA_API_KEY) {
    return res.status(503).json({
      success: false,
      error: "NVIDIA_API_KEY yapılandırılmamış."
    });
  }

  try {
    const result = await getClient().chat.completions.create({
      model: model.id,
      messages: [
        {
          role: "user",
          content: "Türkçe olarak yalnızca 'Bağlantı başarılı' yaz."
        }
      ],
      max_tokens: 40,
      temperature: 0
    });

    res.json({
      success: true,
      model: model.id,
      reply: result.choices?.[0]?.message?.content || "",
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(502).json({
      success: false,
      model: model.id,
      error: safeErrorMessage(error)
    });
  }
});

// -----------------------------
// HATA YÖNETİMİ
// -----------------------------

app.use("/api", (req, res) => {
  res.status(404).json({
    error: "API uç noktası bulunamadı.",
    requestId: req.requestId
  });
});

app.use((error, req, res, next) => {
  logEvent("server_error", {
    requestId: req.requestId,
    errorName: error?.name || "Error"
  });

  if (res.headersSent) {
    return next(error);
  }

  const status = Number(error?.status) || 500;

  res.status(status >= 400 && status < 600 ? status : 500).json({
    error: status === 413
      ? "Gönderilen veri çok büyük."
      : "Sunucu tarafında bir hata oluştu.",
    requestId: req.requestId
  });
});

// -----------------------------
// SUNUCUYU BAŞLAT
// -----------------------------

function startServer() {
  ensureDirectories();

  server = app.listen(PORT, "0.0.0.0", () => {
    logEvent("server_started", {
      port: PORT,
      node: process.version
    });

    console.log(`TürkAI Next çalışıyor: ${PORT}`);
    console.log(`Node.js: ${process.version}`);
    console.log(`Model sayısı: ${Object.keys(MODEL_CONFIG).length}`);
    console.log(`API anahtarı ayarlı: ${Boolean(process.env.NVIDIA_API_KEY)}`);
  });

  server.requestTimeout = 120_000;
  server.headersTimeout = 125_000;
  server.keepAliveTimeout = 5_000;
}

function shutdown(signal) {
  if (shuttingDown) return;

  shuttingDown = true;
  logEvent("server_shutdown", { signal });

  for (const controller of activeControllers) {
    controller.abort();
  }

  if (server) {
    server.close(() => {
      console.log("TürkAI Next güvenli şekilde kapatıldı.");
      process.exit(0);
    });

    setTimeout(() => process.exit(1), 10_000).unref();
  } else {
    process.exit(0);
  }
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

process.on("unhandledRejection", () => {
  logEvent("unhandled_rejection");
});

process.on("uncaughtException", () => {
  logEvent("uncaught_exception");
  shutdown("uncaughtException");
});

startServer();
