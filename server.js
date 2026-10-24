
import express from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import OpenAI from "openai";
import "dotenv/config";

const app = express();
const PORT = process.env.PORT || 3000;

app.disable("x-powered-by");
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "10mb" }));

app.use("/api", rateLimit({
  windowMs: 60 * 1000,
  limit: 30,
  standardHeaders: "draft-7",
  legacyHeaders: false
}));

const NVIDIA_BASE_URL =
  process.env.NVIDIA_BASE_URL ||
  "https://integrate.api.nvidia.com/v1";

const client = process.env.NVIDIA_API_KEY
  ? new OpenAI({
      baseURL: NVIDIA_BASE_URL,
      apiKey: process.env.NVIDIA_API_KEY
    })
  : null;

const MODELS = {
  deepseek: "deepseek-ai/deepseek-v4.1-flash",
  diffusiongemma: "google/diffusiongemma-26b-a4b-it"
};

app.get("/", (_req, res) => {
  res.json({
    name: "TürkAI Next",
    status: "online",
    version: "1.0.0",
    endpoints: ["/health", "/api/models", "/api/chat"]
  });
});

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    apiConfigured: Boolean(client),
    uptimeSeconds: Math.round(process.uptime())
  });
});

app.get("/api/models", (_req, res) => {
  res.json({
    models: Object.entries(MODELS).map(([key, id]) => ({
      key,
      id
    }))
  });
});

app.post("/api/chat", async (req, res) => {
  try {
    if (!client) {
      return res.status(503).json({
        error: "NVIDIA_API_KEY Render Environment bölümünde ayarlanmamış."
      });
    }

    const {
      message,
      model = "deepseek",
      history = []
    } = req.body ?? {};

    if (typeof message !== "string" || !message.trim()) {
      return res.status(400).json({
        error: "message alanına bir mesaj yaz."
      });
    }

    if (message.length > 20000) {
      return res.status(413).json({
        error: "Mesaj en fazla 20000 karakter olabilir."
      });
    }

    if (!Array.isArray(history) || history.length > 20) {
      return res.status(400).json({
        error: "history en fazla 20 mesaj içerebilir."
      });
    }

    const modelId = MODELS[model];

    if (!modelId) {
      return res.status(400).json({
        error: "Bilinmeyen model.",
        availableModels: Object.keys(MODELS)
      });
    }

    const safeHistory = history.every(item =>
      item &&
      ["user", "assistant"].includes(item.role) &&
      typeof item.content === "string" &&
      item.content.length <= 10000
    );

    if (!safeHistory) {
      return res.status(400).json({
        error: "Geçersiz sohbet geçmişi."
      });
    }

    const messages = [
      {
        role: "system",
        content:
          "Sen TürkAI Next'sin. Türkçe konuş, doğru ve anlaşılır cevaplar ver. Bilmediğin şeyleri uydurma."
      },
      ...history,
      { role: "user", content: message.trim() }
    ];

    const result = await client.chat.completions.create({
      model: modelId,
      messages,
      temperature: 0.6,
      max_tokens: 2048
    });

    res.json({
      model: modelId,
      reply: result.choices?.[0]?.message?.content ?? ""
    });
  } catch (error) {
    console.error("NVIDIA API hatası:", error.status || "unknown");

    res.status(502).json({
      error: "Model şu anda cevap veremedi. API erişimini ve model desteğini kontrol et."
    });
  }
});

app.use((err, _req, res, _next) => {
  console.error("İstek hatası:", err.message);

  res.status(400).json({
    error: "İstek işlenemedi."
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`TürkAI Next çalışıyor: port ${PORT}`);
});
