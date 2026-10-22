/**
 * pi-image-gen — Provider-agnostic image generation extension for Pi.
 *
 * Mirrors Pi's text model architecture:
 * - Built-in models from image-models.generated.ts (build-time discovery)
 * - Custom models via config files (runtime override)
 * - API keys and base URLs resolved via Pi's ModelRegistry
 * - /image-model command for interactive model selection (Pi TUI + Minapi)
 * - Session-persisted selection via pi.appendEntry()
 *
 * API types:
 *   openai-images           — POST /v1/images/generations (OpenAI direct)
 *   openai-chat-image       — POST /v1/chat/completions with image output (OpenRouter)
 *   google-generative-ai-image — POST generateContent with responseModalities (Google)
 */

import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { extname, isAbsolute, join } from "node:path";
import { getAgentDir } from "@mariozechner/pi-coding-agent";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import type { AgentToolResult, AgentToolUpdateCallback } from "@mariozechner/pi-agent-core";
import { Container, fuzzyFilter, Input, Spacer, Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";
import { IMAGE_MODELS, type ImageModel, type ImageApi } from "../lib/image-models.generated.js";

// ── Image Model Registry ───────────────────────────────────────────
// Mirrors Pi's ModelRegistry: built-in + custom, with provider-aware lookups.

/** All built-in image models flattened */
function getBuiltInModels(): ImageModel[] {
  const models: ImageModel[] = [];
  for (const provider of Object.values(IMAGE_MODELS)) {
    for (const model of Object.values(provider)) {
      models.push(model);
    }
  }
  return models;
}

/** Find a built-in model by provider + id */
function findModel(provider: string, modelId: string): ImageModel | undefined {
  return IMAGE_MODELS[provider]?.[modelId];
}

/** Get all unique provider names that have image models */
function getImageProviders(): string[] {
  return Object.keys(IMAGE_MODELS).sort();
}

/** Get models for a specific provider */
function getModelsForProvider(provider: string): ImageModel[] {
  const providerModels = IMAGE_MODELS[provider];
  if (!providerModels) return [];
  return Object.values(providerModels);
}

// ── Config ─────────────────────────────────────────────────────────

const SAVE_MODES = ["none", "project", "global", "custom"] as const;
type SaveMode = (typeof SAVE_MODES)[number];

const DEFAULT_TIMEOUT_MS = 600_000; // 10 minutes

interface ExtensionConfig {
  save?: SaveMode;
  saveDir?: string;
  defaultProvider?: string;
  defaultModel?: string;
  /** Default model for image editing (accepts an input image), via /image-edit-model. */
  defaultEditProvider?: string;
  defaultEditModel?: string;
  /** Image generation timeout in seconds. Default: 600 (10 min). */
  timeout?: number;
  /** Image provider definitions. Each key is a provider name; value defines
   *  baseUrl, apiKey, api type, and models. Highest-priority model source. */
  providers?: Record<string, ModelsJsonImageProviderConfig>;
}

function readJsonFile(path: string): Record<string, unknown> {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function globalConfigPath(): string {
  return join(getAgentDir(), "settings", "pi-image-gen.json");
}

function loadConfig(cwd: string): ExtensionConfig {
  const globalConfig = readJsonFile(globalConfigPath());
  const projectConfig = readJsonFile(join(getAgentDir(), "settings", "pi-image-gen.json"));
  return { ...globalConfig, ...projectConfig } as ExtensionConfig;
}

/** Persist model selection to global config so it survives across sessions. */
async function saveDefaultModel(provider: string, modelId: string): Promise<void> {
  const path = globalConfigPath();
  const existing = readJsonFile(path);
  existing.defaultProvider = provider;
  existing.defaultModel = modelId;
  await mkdir(join(getAgentDir(), "settings"), { recursive: true });
  await writeFile(path, JSON.stringify(existing, null, 2));
}

/** Persist edit-model selection to global config so it survives across sessions. */
async function saveDefaultEditModel(provider: string, modelId: string): Promise<void> {
  const path = globalConfigPath();
  const existing = readJsonFile(path);
  existing.defaultEditProvider = provider;
  existing.defaultEditModel = modelId;
  await mkdir(join(getAgentDir(), "settings"), { recursive: true });
  await writeFile(path, JSON.stringify(existing, null, 2));
}

// ── models.json Integration ────────────────────────────────────────
// Read Pi's models.json and extract models with output: ["image"].
// Pi's ajv validation allows additional properties, so users can add
// `output: ["image"]` to any model definition and Pi ignores it while
// pi-image-gen picks it up.
//
// models.json format:
// {
//   "providers": {
//     "openrouter": {
//       "baseUrl": "https://openrouter.ai/api/v1",
//       "models": [
//         { "id": "openai/gpt-5-image", "name": "GPT-5 Image", "output": ["image"] }
//       ]
//     }
//   }
// }

interface ModelsJsonModelDef {
  id: string;
  name?: string;
  api?: string;
  output?: string[];
  /** Legacy explicit image API override in providers[].models[] */
  imageApi?: ImageApi;
  /** Legacy cost per image in USD in providers[].models[] */
  imageCost?: number;
  [key: string]: unknown;
}

interface ModelsJsonImageModelDef {
  id: string;
  name?: string;
  api?: ImageApi;
  cost?: number;
}

interface ModelsJsonProviderConfig {
  baseUrl?: string;
  apiKey?: string;
  models?: ModelsJsonModelDef[];
  modelOverrides?: Record<string, { output?: string[]; [key: string]: unknown }>;
}

interface ModelsJsonImageProviderConfig {
  baseUrl?: string;
  apiKey?: string;
  api?: ImageApi;
  models?: ModelsJsonImageModelDef[];
}

interface ModelsJsonConfig {
  providers?: Record<string, ModelsJsonProviderConfig>;
}

/** Infer legacy image API type from provider name and model api field */
function inferLegacyImageApi(provider: string, apiField?: string): ImageApi {
  if (provider === "google") return "google-generative-ai-image";
  if (provider === "openrouter") return "openai-chat-image";
  if (provider === "openai") return "openai-images";
  if (apiField?.includes("google")) return "google-generative-ai-image";
  return "openai-chat-image";
}

function readModelsJson(): Partial<ModelsJsonConfig> {
  const modelsJsonPath = join(getAgentDir(), "models.json");
  return readJsonFile(modelsJsonPath) as Partial<ModelsJsonConfig>;
}

/** Fallback API key lookup for providers defined in models.json providers section */
function getModelsJsonApiKeyForProvider(provider: string): string | undefined {
  const raw = readModelsJson();
  return raw.providers?.[provider]?.apiKey;
}

/** Read Pi's models.json and extract image-capable models from the legacy
 *  providers.<provider>.models[] with output:["image"] format. */
function getModelsJsonImageModels(): ImageModel[] {
  const raw = readModelsJson();
  const models = new Map<string, ImageModel>();

  if (raw.providers) {
    for (const [provider, config] of Object.entries(raw.providers)) {
      const baseUrl = config.baseUrl || "";

      if (config.models) {
        for (const modelDef of config.models) {
          if (!modelDef.output?.includes("image")) continue;
          const model: ImageModel = {
            id: modelDef.id,
            name: modelDef.name || modelDef.id,
            provider,
            baseUrl,
            api: modelDef.imageApi || inferLegacyImageApi(provider, modelDef.api),
            cost: modelDef.imageCost || 0,
          };
          models.set(`${provider}/${modelDef.id}`, model);
        }
      }

      if (config.modelOverrides) {
        for (const [modelId, override] of Object.entries(config.modelOverrides)) {
          if (!override.output?.includes("image")) continue;
          if (config.models?.some((m) => m.id === modelId)) continue;
          const model: ImageModel = {
            id: modelId,
            name: modelId,
            provider,
            baseUrl,
            api: inferLegacyImageApi(provider, undefined),
            cost: 0,
          };
          models.set(`${provider}/${modelId}`, model);
        }
      }
    }
  }

  return [...models.values()];
}

// ── Runtime Discovery ──────────────────────────────────────────────
// On startup, fetch image-capable models from OpenRouter. Results are
// cached to disk with a 24h TTL so we don't hit the API every launch.
// Discovery runs non-blocking — if the cache is warm, it's instant.
// If the fetch fails, we silently fall back to built-in + models.json.

function discoveryCacheDir(): string { return join(getAgentDir(), "cache"); }
function discoveryCacheFile(): string { return join(discoveryCacheDir(), "pi-image-gen-discovered.json"); }
const DISCOVERY_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

interface DiscoveryCache {
  fetchedAt: number;
  models: ImageModel[];
  /** OpenRouter image-model edit capability: id → accepts input references. */
  inputRefs?: Record<string, boolean>;
}

/** In-memory edit-capability map (OpenRouter image model id → accepts input images). */
const inputRefSupport = new Map<string, boolean>();

/** Read cached discovered models if cache is fresh */
function readDiscoveryCache(): ImageModel[] {
  const cacheFile = discoveryCacheFile();
  if (!existsSync(cacheFile)) return [];
  try {
    const stat = statSync(cacheFile);
    const ageMs = Date.now() - stat.mtimeMs;
    if (ageMs > DISCOVERY_TTL_MS) return []; // Stale

    const cache = JSON.parse(readFileSync(cacheFile, "utf-8")) as DiscoveryCache;
    if (cache.inputRefs) {
      for (const [id, supports] of Object.entries(cache.inputRefs)) inputRefSupport.set(id, supports);
    }
    return cache.models;
  } catch {
    return [];
  }
}

/** In-memory store for discovered models (populated on session_start) */
let discoveredModels: ImageModel[] = readDiscoveryCache();

/** Fetch image models from OpenRouter at runtime and update cache.
 *
 * Uses /v1/models?output_modalities=image which includes image-only models
 * (muse-image, FLUX, Riverflow, Seedream) that the plain /v1/models endpoint
 * omits. (The old frontend API /api/frontend/models now returns 404.)
 */
async function refreshDiscoveryCache(): Promise<void> {
  // Skip if cache is still fresh AND carries the edit-capability map
  // (caches written by older versions lack inputRefs — refetch once to fill it)
  const cacheFile = discoveryCacheFile();
  if (existsSync(cacheFile)) {
    try {
      const stat = statSync(cacheFile);
      const cached = JSON.parse(readFileSync(cacheFile, "utf-8")) as DiscoveryCache;
      if (Date.now() - stat.mtimeMs < DISCOVERY_TTL_MS && cached.inputRefs) return;
    } catch {
      // Continue to fetch
    }
  }

  try {
    let models: ImageModel[] = [];

    // Image-output filter includes image-only models (muse-image, FLUX,
    // Seedream, …) that the plain /v1/models endpoint omits
    const response = await fetch("https://openrouter.ai/api/v1/models?output_modalities=image");
    if (!response.ok) return;

    const data = (await response.json()) as {
      data: Array<{
        id: string;
        name: string;
        architecture?: { output_modalities?: string[] };
      }>;
    };

    for (const model of data.data) {
      const outputMods = model.architecture?.output_modalities ?? [];
      if (!outputMods.includes("image")) continue;
      if (model.id.startsWith("openrouter/auto")) continue;

      models.push({
        id: model.id,
        name: model.name,
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
        api: "openai-chat-image",
        cost: 0,
      });
    }

    // Capability map: which OpenRouter image models accept input references
    // (needed for image editing). Best-effort — edits just won't be offered
    // for unknown models if this fails.
    let inputRefs: Record<string, boolean> | undefined;
    try {
      const capsResponse = await fetch("https://openrouter.ai/api/v1/images/models");
      if (capsResponse.ok) {
        const caps = (await capsResponse.json()) as {
          data?: Array<{ id: string; supported_parameters?: Record<string, unknown> }>;
        };
        inputRefs = {};
        for (const m of caps.data ?? []) {
          inputRefs[m.id] = "input_references" in (m.supported_parameters ?? {});
        }
      }
    } catch {
      // Capability discovery is best-effort
    }

    // Write cache
    await mkdir(discoveryCacheDir(), { recursive: true });
    const cache: DiscoveryCache = { fetchedAt: Date.now(), models, ...(inputRefs ? { inputRefs } : {}) };
    await writeFile(discoveryCacheFile(), JSON.stringify(cache, null, 2));

    // Update in-memory
    discoveredModels = models;
    if (inputRefs) {
      for (const [id, supports] of Object.entries(inputRefs)) inputRefSupport.set(id, supports);
    }
  } catch {
    // Silent failure — we still have built-in + models.json
  }
}

/** Read image models from the settings file's providers section. */
function getSettingsImageModels(config: ExtensionConfig): ImageModel[] {
  if (!config.providers) return [];
  const models = new Map<string, ImageModel>();

  for (const [provider, providerConfig] of Object.entries(config.providers)) {
    const baseUrl = providerConfig.baseUrl || "";
    const providerApi = providerConfig.api || inferLegacyImageApi(provider);

    if (!providerConfig.models) continue;
    for (const modelDef of providerConfig.models) {
      models.set(`${provider}/${modelDef.id}`, {
        id: modelDef.id,
        name: modelDef.name || modelDef.id,
        provider,
        baseUrl,
        api: modelDef.api || providerApi,
        cost: modelDef.cost || 0,
      });
    }
  }

  return [...models.values()];
}

/** Merge all model sources: built-in → discovered → models.json → settings providers */
function getAllModels(config: ExtensionConfig): ImageModel[] {
  const models = new Map<string, ImageModel>();

  // 1. Built-in generated models (lowest priority)
  for (const m of getBuiltInModels()) {
    models.set(`${m.provider}/${m.id}`, m);
  }

  // 2. Runtime-discovered models
  for (const m of discoveredModels) {
    models.set(`${m.provider}/${m.id}`, m);
  }

  // 3. Pi's models.json — legacy output:["image"] entries
  for (const m of getModelsJsonImageModels()) {
    models.set(`${m.provider}/${m.id}`, m);
  }

  // 4. Settings providers (highest priority)
  for (const m of getSettingsImageModels(config)) {
    models.set(`${m.provider}/${m.id}`, m);
  }

  return [...models.values()];
}

// ── Edit (input image) Support ─────────────────────────────────────

/** Parse a data URL into mime type + raw bytes. */
function parseDataUrl(dataUrl: string): { mimeType: string; buffer: Buffer } {
  const match = dataUrl.match(/^data:([^;,]+)(;base64)?,(.*)$/s);
  if (!match) throw new Error("Invalid data URL for input image.");
  return { mimeType: match[1] || "image/png", buffer: Buffer.from(match[3] || "", match[2] ? "base64" : "utf-8") };
}

/** Whether a model accepts an input/reference image (image editing). */
function modelSupportsInputReferences(model: ImageModel): boolean {
  if (model.provider === "openrouter") {
    // Capabilities discovered from GET /api/v1/images/models
    return inputRefSupport.get(model.id) === true;
  }
  if (model.provider === "openai") return model.id.startsWith("gpt-image");
  if (model.provider === "google") return model.id.includes("-image") && !model.id.includes("imagen");
  return false;
}

/** All models that accept an input image (for /image-edit-model + edit routing). */
function listInputImageModels(config: ExtensionConfig): ImageModel[] {
  return getAllModels(config).filter(modelSupportsInputReferences);
}

/** Read a local image file as a base64 data URL for input_references/inlineData. */
function loadInputImageDataUrl(inputPath: string, cwd: string): string {
  const path = isAbsolute(inputPath) ? inputPath : join(cwd, inputPath);
  if (!existsSync(path)) throw new Error(`Input image not found: ${path}`);
  const ext = extname(path).toLowerCase();
  const mimeType = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : ext === ".webp" ? "image/webp" : ext === ".gif" ? "image/gif" : "image/png";
  return `data:${mimeType};base64,${readFileSync(path).toString("base64")}`;
}

// ── Save Logic ─────────────────────────────────────────────────────

function resolveSaveConfig(save: string | undefined, saveDir: string | undefined, cwd: string, config: ExtensionConfig): { mode: SaveMode; outputDir?: string } {
  const mode = (save || config.save || "none") as SaveMode;

  if (mode === "project") return { mode, outputDir: join(cwd, ".pi", "generated-images") };
  if (mode === "global") return { mode, outputDir: join(getAgentDir(), "generated-images") };
  if (mode === "custom") {
    const dir = saveDir || config.saveDir;
    if (!dir?.trim()) throw new Error("save=custom requires saveDir parameter or saveDir in config.");
    return { mode, outputDir: dir };
  }
  return { mode: "none" };
}

function imageExtension(mimeType: string): string {
  const lower = mimeType.toLowerCase();
  if (lower.includes("jpeg") || lower.includes("jpg")) return "jpg";
  if (lower.includes("gif")) return "gif";
  if (lower.includes("webp")) return "webp";
  return "png";
}

async function saveImageToDisk(base64Data: string, mimeType: string, outputDir: string): Promise<string> {
  await mkdir(outputDir, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const ext = imageExtension(mimeType);
  const filename = `image-${timestamp}-${randomUUID().slice(0, 8)}.${ext}`;
  const filePath = join(outputDir, filename);
  await writeFile(filePath, Buffer.from(base64Data, "base64"));
  return filePath;
}

// ── Provider Config Resolution ─────────────────────────────────────

interface ProviderConfig {
  apiKey: string;
  baseUrl: string;
}

async function resolveProviderConfig(
  imageModel: ImageModel,
  ctx: ExtensionContext,
  config: ExtensionConfig,
): Promise<ProviderConfig> {
  // Resolution order: Pi's ModelRegistry → settings providers → models.json providers
  let apiKey = await ctx.modelRegistry.getApiKeyForProvider(imageModel.provider);
  if (!apiKey) {
    apiKey = config.providers?.[imageModel.provider]?.apiKey;
  }
  if (!apiKey) {
    apiKey = getModelsJsonApiKeyForProvider(imageModel.provider);
  }
  if (!apiKey) {
    throw new Error(
      `No API key for provider "${imageModel.provider}". ` +
      `Set it in environment, /login, or add apiKey to providers.${imageModel.provider} in ~/.pi/agent/settings/pi-image-gen.json.`,
    );
  }

  // Use model's baseUrl, or fall back to Pi's registry for the provider
  let baseUrl = imageModel.baseUrl;
  if (!baseUrl) {
    const allModels = ctx.modelRegistry.getAll();
    const textModel = allModels.find((m) => m.provider === imageModel.provider);
    baseUrl = textModel?.baseUrl || "";
  }

  if (!baseUrl) {
    throw new Error(`No base URL for provider "${imageModel.provider}".`);
  }

  return { apiKey, baseUrl };
}

// ── Image Generation by API Type ───────────────────────────────────

interface GeneratedImage {
  data: string;       // base64
  mimeType: string;
  /** Billed cost in USD when the provider reports it (OpenRouter `usage.cost`). */
  billedCost?: number;
  /** Token usage when the provider reports it. */
  reportedUsage?: { prompt: number; completion: number; total: number };
}

/** `usage` object on OpenRouter chat/images responses. */
interface OpenRouterUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cost?: number;
}

/** Pull token counts and billed cost out of an OpenRouter response `usage` object. */
function openRouterUsageFields(usage: OpenRouterUsage | undefined): Partial<GeneratedImage> {
  if (!usage) return {};
  const prompt = usage.prompt_tokens ?? 0;
  const completion = usage.completion_tokens ?? 0;
  const total = usage.total_tokens ?? prompt + completion;
  return {
    reportedUsage: { prompt, completion, total },
    ...(typeof usage.cost === "number" ? { billedCost: usage.cost } : {}),
  };
}

/** Session `usage` entry kind used for image generation spend. */
const IMAGE_USAGE_KIND = "image_gen";

// OpenAI size mapping from aspect ratios
const OPENAI_SIZES: Record<string, string> = {
  "1:1": "1024x1024",
  "16:9": "1536x1024",
  "9:16": "1024x1536",
  "4:3": "1536x1024",
  "3:4": "1024x1536",
  "3:2": "1536x1024",
  "2:3": "1024x1536",
};

/** OpenAI /v1/images/generations endpoint */
async function generateViaOpenAIImages(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  aspectRatio: string,
  quality: string,
  signal?: AbortSignal,
): Promise<GeneratedImage> {
  const size = OPENAI_SIZES[aspectRatio] || "1024x1024";
  const body: Record<string, unknown> = {
    model: model.id,
    prompt,
    n: 1,
    size,
    response_format: "b64_json",
  };
  if (model.id === "gpt-image-1") body.quality = quality;
  if (model.id === "dall-e-3") body.quality = quality === "high" ? "hd" : "standard";

  const response = await fetch(`${config.baseUrl}/images/generations`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI image generation failed (${response.status}): ${errorText}`);
  }

  const result = (await response.json()) as { data: Array<{ b64_json?: string }> };
  if (!result.data[0]?.b64_json) throw new Error("OpenAI returned no image data");

  return { data: result.data[0].b64_json, mimeType: "image/png", billedCost: model.cost > 0 ? model.cost : undefined };
}

/** OpenAI /v1/images/edits (multipart) — edits existing images from a prompt. */
async function generateViaOpenAIEdits(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  aspectRatio: string,
  quality: string,
  inputImages: string[],
  signal?: AbortSignal,
): Promise<GeneratedImage> {
  const size = OPENAI_SIZES[aspectRatio] || "1024x1024";
  const form = new FormData();
  form.append("model", model.id);
  form.append("prompt", prompt);
  form.append("n", "1");
  form.append("size", size);
  if (model.id === "gpt-image-1") form.append("quality", quality);
  for (const [i, dataUrl] of inputImages.entries()) {
    const { mimeType, buffer } = parseDataUrl(dataUrl);
    const ext = mimeType.includes("jpeg") ? "jpg" : mimeType.includes("webp") ? "webp" : "png";
    form.append("image[]", new Blob([new Uint8Array(buffer)], { type: mimeType }), `input-${i}.${ext}`);
  }

  const response = await fetch(`${config.baseUrl}/images/edits`, {
    method: "POST",
    headers: { Authorization: `Bearer ${config.apiKey}` },
    body: form,
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI image edit failed (${response.status}): ${errorText}`);
  }

  const result = (await response.json()) as { data: Array<{ b64_json?: string }> };
  if (!result.data[0]?.b64_json) throw new Error("OpenAI returned no image data for edit");

  return { data: result.data[0].b64_json, mimeType: "image/png", billedCost: model.cost > 0 ? model.cost : undefined };
}

/** OpenRouter chat completions with image output.
 *
 * Uses `modalities: ["image"]` so image-only endpoints can route successfully.
 * Response includes `message.images[]` with base64 data URLs.
 * See: https://openrouter.ai/docs/features/image-generation
 */
async function generateViaChatImage(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  aspectRatio: string,
  imageSize: string,
  signal?: AbortSignal,
): Promise<GeneratedImage> {
  const body: Record<string, unknown> = {
    model: model.id,
    messages: [{ role: "user", content: prompt }],
    modalities: ["image"],
  };

  // Add image_config for aspect ratio and size
  const imageConfig: Record<string, string> = {};
  if (aspectRatio && aspectRatio !== "1:1") {
    imageConfig.aspect_ratio = aspectRatio;
  }
  if (imageSize && imageSize !== "1K") {
    imageConfig.image_size = imageSize;
  }
  if (Object.keys(imageConfig).length > 0) {
    body.image_config = imageConfig;
  }

  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    // Some models (e.g. meta/muse-image) refuse chat/completions and require
    // OpenRouter's unified Image API instead.
    if (errorText.includes("/api/v1/images") || errorText.includes("images endpoint")) {
      return generateViaUnifiedImage(config, model, prompt, aspectRatio, imageSize, signal);
    }
    throw new Error(`OpenRouter image generation failed (${response.status}): ${errorText}`);
  }

  const result = (await response.json()) as {
    choices: Array<{
      message: {
        content?: string;
        images?: Array<{
          type: string;
          image_url: { url: string };
        }>;
      };
    }>;
    usage?: OpenRouterUsage;
  };

  const message = result.choices[0]?.message;
  if (!message) throw new Error("No message in OpenRouter response");

  // Images are in message.images[], each with a base64 data URL
  if (message.images && message.images.length > 0) {
    const url = message.images[0].image_url.url;
    if (url.startsWith("data:")) {
      const commaIdx = url.indexOf(",");
      const header = url.slice(0, commaIdx);
      const data = url.slice(commaIdx + 1);
      const mimeType = header.match(/data:(.*);/)?.[1] || "image/png";
      return { data, mimeType, ...openRouterUsageFields(result.usage) };
    }
    // URL-based — fetch and convert
    const imgResponse = await fetch(url, { signal });
    const buffer = await imgResponse.arrayBuffer();
    return {
      data: Buffer.from(buffer).toString("base64"),
      mimeType: imgResponse.headers.get("content-type") || "image/png",
      ...openRouterUsageFields(result.usage),
    };
  }

  throw new Error("No images in OpenRouter response. Ensure the model supports image generation.");
}

/** OpenRouter unified Image API (POST /api/v1/images).
 *
 * Required by image-only models like meta/muse-image that reject the
 * chat/completions endpoint. Returns `{ data: [{ b64_json }] }` like OpenAI.
 * See: https://openrouter.ai/docs/api/api-reference/images/generate-an-image
 */
async function generateViaUnifiedImage(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  aspectRatio: string,
  imageSize: string,
  signal?: AbortSignal,
  inputImages?: string[],
): Promise<GeneratedImage> {
  const body: Record<string, unknown> = { model: model.id, prompt };
  if (aspectRatio && aspectRatio !== "1:1") body.aspect_ratio = aspectRatio;
  if (imageSize && imageSize !== "1K") body.resolution = imageSize;
  if (inputImages && inputImages.length > 0) {
    body.input_references = inputImages.map((url) => ({ type: "image_url", image_url: { url } }));
  }

  const response = await fetch(`${config.baseUrl}/images`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter image API failed (${response.status}): ${errorText}`);
  }

  const result = (await response.json()) as {
    data?: Array<{ b64_json?: string; url?: string }>;
    usage?: OpenRouterUsage;
  };
  const first = result.data?.[0];
  if (first?.b64_json) return { data: first.b64_json, mimeType: "image/png", ...openRouterUsageFields(result.usage) };
  if (first?.url) {
    const imgResponse = await fetch(first.url, { signal });
    const buffer = await imgResponse.arrayBuffer();
    return {
      data: Buffer.from(buffer).toString("base64"),
      mimeType: imgResponse.headers.get("content-type") || "image/png",
      ...openRouterUsageFields(result.usage),
    };
  }
  throw new Error("No image data in OpenRouter images response.");
}

/** Google Gemini generateContent with responseModalities: ["IMAGE"] */
async function generateViaGoogleImage(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  signal?: AbortSignal,
  inputImages?: string[],
): Promise<GeneratedImage> {
  const requestParts: Array<Record<string, unknown>> = [{ text: prompt }];
  for (const dataUrl of inputImages ?? []) {
    const { mimeType, buffer } = parseDataUrl(dataUrl);
    requestParts.push({ inlineData: { mimeType, data: buffer.toString("base64") } });
  }
  const response = await fetch(
    `${config.baseUrl}/models/${model.id}:generateContent?key=${config.apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: requestParts }],
        generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
      }),
      signal,
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Google image generation failed (${response.status}): ${errorText}`);
  }

  const result = (await response.json()) as {
    candidates?: Array<{
      content?: { parts?: Array<{ inlineData?: { mimeType?: string; data?: string } }> };
    }>;
  };

  const parts = result.candidates?.[0]?.content?.parts;
  if (!parts) throw new Error("Google returned no content");
  for (const part of parts) {
    if (part.inlineData?.data) {
      return { data: part.inlineData.data, mimeType: part.inlineData.mimeType || "image/png", billedCost: model.cost > 0 ? model.cost : undefined };
    }
  }
  throw new Error("Google returned no image data");
}

/** Route to the appropriate generation method based on model's API type */
async function generateImage(
  config: ProviderConfig,
  model: ImageModel,
  prompt: string,
  aspectRatio: string,
  imageSize: string,
  quality: string,
  signal?: AbortSignal,
  inputImages?: string[],
): Promise<GeneratedImage> {
  // Editing path: route by provider to an endpoint that accepts input images
  if (inputImages && inputImages.length > 0) {
    switch (model.provider) {
      case "openrouter":
        // Unified Image API (input_references); works for models that reject
        // chat/completions too (e.g. meta/muse-image)
        return generateViaUnifiedImage(config, model, prompt, aspectRatio, imageSize, signal, inputImages);
      case "openai":
        return generateViaOpenAIEdits(config, model, prompt, aspectRatio, quality, inputImages, signal);
      case "google":
        return generateViaGoogleImage(config, model, prompt, signal, inputImages);
      default:
        throw new Error(
          `Image editing is not supported for provider "${model.provider}". ` +
          `Pick an edit model with /image-edit-model.`,
        );
    }
  }

  switch (model.api) {
    case "openai-images":
      return generateViaOpenAIImages(config, model, prompt, aspectRatio, quality, signal);
    case "openai-chat-image":
      return generateViaChatImage(config, model, prompt, aspectRatio, imageSize, signal);
    case "google-generative-ai-image":
      return generateViaGoogleImage(config, model, prompt, signal);
    default:
      throw new Error(`Unknown image API type: ${model.api}`);
  }
}

// ── Model Resolution ───────────────────────────────────────────────

interface ModelSelection {
  provider: string;
  model: string;
}

function resolveModel(
  sessionSelection: ModelSelection | null,
  config: ExtensionConfig,
): ImageModel {
  const allModels = getAllModels(config);

  // 1. Session selection (set via /image-model)
  if (sessionSelection) {
    const found = allModels.find(
      (m) => m.provider === sessionSelection.provider && m.id === sessionSelection.model,
    );
    if (found) return found;
  }

  // 2. Config defaults
  if (config.defaultModel) {
    const found = allModels.find((m) => m.id === config.defaultModel);
    if (found) return found;
  }
  if (config.defaultProvider) {
    const providerModels = allModels.filter((m) => m.provider === config.defaultProvider);
    if (providerModels.length > 0) return providerModels[0];
  }

  // 4. Default: first openrouter model, then openai, then anything
  const orModels = allModels.filter((m) => m.provider === "openrouter");
  if (orModels.length > 0) return orModels[0];
  const openaiModels = allModels.filter((m) => m.provider === "openai");
  if (openaiModels.length > 0) return openaiModels[0];
  if (allModels.length > 0) return allModels[0];

  throw new Error("No image models available. Run /image-model to select one.");
}

/** Resolve the model used when an input image is provided (edit mode).
 *  Falls back through: session edit selection → config edit defaults →
 *  the current generation model (if it accepts input images) →
 *  the first known edit-capable model. Throws with advice when none exist. */
function resolveEditModel(
  editSelection: ModelSelection | null,
  config: ExtensionConfig,
  generationModel: ImageModel,
): ImageModel {
  const allModels = getAllModels(config);

  if (editSelection) {
    const found = allModels.find(
      (m) => m.provider === editSelection.provider && m.id === editSelection.model,
    );
    if (found && modelSupportsInputReferences(found)) return found;
  }
  if (config.defaultEditModel) {
    const found = allModels.find((m) => m.id === config.defaultEditModel);
    if (found && modelSupportsInputReferences(found)) return found;
  }
  if (modelSupportsInputReferences(generationModel)) return generationModel;

  const candidates = listInputImageModels(config);
  if (candidates.length > 0) return candidates[0];

  throw new Error(
    "No image model that accepts an input image is available, so this image can't be edited. " +
    "Run /image-edit-model to see and select edit-capable models " +
    "(e.g. openai/gpt-image-1, openai/gpt-image-2, google/gemini-3.1-flash-image via OpenRouter).",
  );
}

// ── Session State ──────────────────────────────────────────────────

const SESSION_ENTRY_TYPE = "pi-image-gen:model-select";
const SESSION_ENTRY_EDIT_TYPE = "pi-image-gen:edit-model-select";

// ── Tool Parameters ────────────────────────────────────────────────

const TOOL_PARAMS = Type.Object({
  prompt: Type.String({ description: "Description of the image to generate (or the edit to apply when inputImage is set)." }),
  inputImage: Type.Optional(Type.String({
    description: "Path to an existing image file to edit. When set, the edit model (from /image-edit-model) is used; it must accept input images.",
  })),
  inputImages: Type.Optional(Type.Array(Type.String(), {
    description: "Paths to multiple existing image files to use together as references (OpenRouter accepts up to 16). Merged with inputImage.",
  })),
  aspectRatio: Type.Optional(Type.String({
    description: "Aspect ratio: 1:1 (default), 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.",
  })),
  imageSize: Type.Optional(Type.String({
    description: "Image resolution: 1K (1024x1024, default), 2K, 4K. OpenRouter models only.",
  })),
  quality: Type.Optional(Type.String({
    description: "Image quality: auto (default), low, medium, high. OpenAI direct only.",
  })),
  save: Type.Optional(Type.String({
    description: "Save mode: none, project, global, custom. Default: none.",
  })),
  saveDir: Type.Optional(Type.String({
    description: "Directory to save image when save=custom.",
  })),
  timeout: Type.Optional(Type.Number({
    description: "Generation timeout in seconds. Default: 600 (10 min).",
  })),
});

type ToolParams = Static<typeof TOOL_PARAMS>;

// ── Extension Entry Point ──────────────────────────────────────────

/** Interactive fuzzy-search model picker shared by /image-model and
 *  /image-edit-model (mirrors Pi's /model UI). */
async function pickModelWithUI(
  ctx: ExtensionContext,
  models: ImageModel[],
  currentKey: string | null,
  title: string,
): Promise<ImageModel | undefined> {
  // Sort: current model first, then alphabetically by provider
  const sortedModels = [...models].sort((a, b) => {
    const aIsCurrent = `${a.provider}/${a.id}` === currentKey;
    const bIsCurrent = `${b.provider}/${b.id}` === currentKey;
    if (aIsCurrent && !bIsCurrent) return -1;
    if (!aIsCurrent && bIsCurrent) return 1;
    return a.provider.localeCompare(b.provider) || a.id.localeCompare(b.id);
  });

  return ctx.ui.custom<ImageModel | undefined>((tui, theme, kb, done) => {
    const searchInput = new Input();
    searchInput.focused = true;
    const listContainer = new Container();
    const previewText = new Text("", 1, 0);
    let filteredModels = sortedModels;
    let selectedIndex = 0;

    // Thin border that spans viewport width, matching Pi's DynamicBorder
    const border = { render: (w: number) => [theme.fg("border", "─".repeat(Math.max(1, w)))], invalidate() {} };

    function updateList() {
      listContainer.clear();
      const maxVisible = 10;
      const startIndex = Math.max(
        0,
        Math.min(selectedIndex - Math.floor(maxVisible / 2), filteredModels.length - maxVisible),
      );
      const endIndex = Math.min(startIndex + maxVisible, filteredModels.length);

      for (let i = startIndex; i < endIndex; i++) {
        const item = filteredModels[i];
        if (!item) continue;
        const isHighlighted = i === selectedIndex;
        const isCurrent = `${item.provider}/${item.id}` === currentKey;
        const providerBadge = theme.fg("muted", ` [${item.provider}]`);
        const checkmark = isCurrent ? theme.fg("success", " ✓") : "";

        const line = isHighlighted
          ? `${theme.fg("accent", "→ ")}${theme.fg("accent", item.id)}${providerBadge}${checkmark}`
          : `  ${item.id}${providerBadge}${checkmark}`;
        listContainer.addChild(new Text(line, 0, 0));
      }

      if (filteredModels.length > maxVisible) {
        listContainer.addChild(
          new Text(theme.fg("muted", `  (${selectedIndex + 1}/${filteredModels.length})`), 0, 0),
        );
      }

      if (filteredModels.length === 0) {
        listContainer.addChild(new Text(theme.fg("muted", "  No matching models"), 0, 0));
      }
    }

    function updatePreview() {
      const model = filteredModels[selectedIndex];
      previewText.setText(model ? theme.fg("muted", `Model Name: ${model.name}`) : "");
    }

    function filterModels(query: string) {
      filteredModels = query
        ? fuzzyFilter(sortedModels, query, (m) => `${m.id} ${m.provider} ${m.name}`)
        : sortedModels;
      selectedIndex = Math.min(selectedIndex, Math.max(0, filteredModels.length - 1));
      updateList();
      updatePreview();
    }

    searchInput.onSubmit = () => {
      if (filteredModels[selectedIndex]) done(filteredModels[selectedIndex]);
    };

    const container = new Container();
    container.addChild(border);
    container.addChild(new Spacer(1));
    container.addChild(new Text(theme.fg("accent", title), 1, 0));
    container.addChild(new Spacer(1));
    container.addChild(searchInput);
    container.addChild(new Spacer(1));
    container.addChild(listContainer);
    container.addChild(new Spacer(1));
    container.addChild(previewText);
    container.addChild(new Spacer(1));
    container.addChild(border);

    updateList();
    updatePreview();

    return {
      render(width: number) { return container.render(width); },
      invalidate() { container.invalidate(); },
      handleInput(data: string) {
        if (kb.matches(data, "tui.select.up")) {
          if (filteredModels.length === 0) return;
          selectedIndex = selectedIndex === 0 ? filteredModels.length - 1 : selectedIndex - 1;
          updateList();
          updatePreview();
        } else if (kb.matches(data, "tui.select.down")) {
          if (filteredModels.length === 0) return;
          selectedIndex = selectedIndex === filteredModels.length - 1 ? 0 : selectedIndex + 1;
          updateList();
          updatePreview();
        } else if (kb.matches(data, "tui.select.confirm")) {
          if (filteredModels[selectedIndex]) done(filteredModels[selectedIndex]);
        } else if (kb.matches(data, "tui.select.cancel")) {
          done(undefined);
        } else {
          searchInput.handleInput(data);
          filterModels(searchInput.getValue());
        }
        tui.requestRender();
      },
    };
  });
}

export default function piImageGen(pi: ExtensionAPI) {
  let sessionSelection: ModelSelection | null = null;
  let editModelSelection: ModelSelection | null = null;

  // ── Image cost accounting ──
  // Cumulative image spend for this session, surfaced as a footer status chip.
  let imageCostTotal = 0;

  function updateImageCostStatus(ctx: ExtensionContext): void {
    if (imageCostTotal > 0) {
      ctx.ui.setStatus("image-cost", `images: $${imageCostTotal.toFixed(3)}`);
    } else {
      ctx.ui.setStatus("image-cost", undefined);
    }
  }

  /**
   * Record image generation spend as a session `usage` entry so it flows into
   * pi's footer cost total and the /session cost breakdown (grouped per model).
   * Also refreshes the footer's cumulative image-cost chip.
   * Returns the cost recorded, in USD.
   *
   * NOTE: ctx.sessionManager is typed read-only (ReadonlySessionManager) but is
   * the live SessionManager at runtime — the same object pi's cache-warmer
   * appends usage entries with. Relies on runtime identity; flagged for an
   * official API upstream.
   */
  function recordImageUsage(ctx: ExtensionContext, model: ImageModel, image: GeneratedImage): number {
    const cost = image.billedCost ?? (model.cost > 0 ? model.cost : 0);
    const prompt = image.reportedUsage?.prompt ?? 0;
    const completion = image.reportedUsage?.completion ?? 0;
    const total = image.reportedUsage?.total ?? prompt + completion;
    if (cost <= 0 && total <= 0) return 0;

    const usage = {
      input: prompt,
      output: completion,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: total,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: cost },
    };
    try {
      const sm = ctx.sessionManager as unknown as {
        appendUsage(kind: string, provider: string, model: string, usage: typeof usage, note?: string): unknown;
      };
      sm.appendUsage(IMAGE_USAGE_KIND, model.provider, model.id, usage, "image generation");
    } catch {
      // Best-effort: generation succeeded even if accounting failed.
    }
    imageCostTotal += cost;
    updateImageCostStatus(ctx);
    return cost;
  }

  // ── Restore state from session ──

  pi.on("session_start", async (_event, ctx) => {
    // Restore session model selection from branch entries, then fall back to saved config
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === SESSION_ENTRY_TYPE) {
        const data = entry.data as ModelSelection | undefined;
        if (data?.provider && data?.model) {
          sessionSelection = data;
        }
      }
    }
    if (!sessionSelection) {
      const config = loadConfig(ctx.cwd);
      if (config.defaultProvider && config.defaultModel) {
        sessionSelection = { provider: config.defaultProvider, model: config.defaultModel };
      }
    }
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === SESSION_ENTRY_EDIT_TYPE) {
        const data = entry.data as ModelSelection | undefined;
        if (data?.provider && data?.model) {
          editModelSelection = data;
        }
      }
    }
    if (!editModelSelection) {
      const config = loadConfig(ctx.cwd);
      if (config.defaultEditProvider && config.defaultEditModel) {
        editModelSelection = { provider: config.defaultEditProvider, model: config.defaultEditModel };
      }
    }

    // Seed the footer image-cost chip from usage entries persisted in this session
    imageCostTotal = 0;
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type === "usage" && entry.kind === IMAGE_USAGE_KIND) {
        imageCostTotal += entry.usage.cost.total;
      }
    }
    updateImageCostStatus(ctx);

    // Runtime discovery — non-blocking, updates cache for this + future sessions
    refreshDiscoveryCache().catch(() => {});
  });

  // ── /image-model command ──

  pi.registerCommand("image-model", {
    description: "Select the default image generation model",

    getArgumentCompletions(prefix: string) {
      const config = loadConfig(process.cwd());
      const models = getAllModels(config);
      return models
        .map((m) => `${m.provider}/${m.id}`)
        .filter((label) => label.startsWith(prefix))
        .map((label) => {
          const model = models.find((m) => `${m.provider}/${m.id}` === label)!;
          return { value: label, label: `${model.name} (${model.provider})` };
        });
    },

    async handler(args: string, ctx) {
      const config = loadConfig(ctx.cwd);
      const allModels = getAllModels(config);

      // Direct selection: /image-model openai/gpt-image-1
      if (args.trim()) {
        const input = args.trim();
        // Try "provider/model" format
        const slashIdx = input.indexOf("/");
        let found: ImageModel | undefined;
        if (slashIdx > 0) {
          const provider = input.slice(0, slashIdx);
          const modelId = input.slice(slashIdx + 1);
          found = allModels.find((m) => m.provider === provider && m.id === modelId);
          // For openrouter models, the id itself contains slashes (e.g., "openai/gpt-5-image")
          if (!found) {
            found = allModels.find((m) => m.id === input);
          }
        } else {
          // Try just model id
          found = allModels.find((m) => m.id === input);
        }

        if (!found) {
          ctx.ui.notify(`Unknown image model "${input}". Run /image-model to see options.`, "error");
          return;
        }

        sessionSelection = { provider: found.provider, model: found.id };
        pi.appendEntry(SESSION_ENTRY_TYPE, sessionSelection);
        await saveDefaultModel(found.provider, found.id);
        ctx.ui.setStatus("image-model", undefined);
        ctx.ui.notify(`Image model: ${found.name} (${found.provider}/${found.id})`, "info");
        return;
      }

      // Interactive selection with search — mirrors Pi's /model UI
      const currentKey = sessionSelection ? `${sessionSelection.provider}/${sessionSelection.model}` : null;
      const selected = await pickModelWithUI(ctx, allModels, currentKey, "Select image model");
      if (!selected) return;

      sessionSelection = { provider: selected.provider, model: selected.id };
      pi.appendEntry(SESSION_ENTRY_TYPE, sessionSelection);
      await saveDefaultModel(selected.provider, selected.id);
      ctx.ui.setStatus("image-model", undefined);
      ctx.ui.notify(`Image model: ${selected.name} (${selected.provider}/${selected.id})`, "info");
    },
  });

  // ── /image-edit-model command ──

  pi.registerCommand("image-edit-model", {
    description: "Select the default image edit model (accepts an input image)",

    getArgumentCompletions(prefix: string) {
      const config = loadConfig(process.cwd());
      const models = listInputImageModels(config);
      return models
        .map((m) => `${m.provider}/${m.id}`)
        .filter((label) => label.startsWith(prefix))
        .map((label) => {
          const model = models.find((m) => `${m.provider}/${m.id}` === label)!;
          return { value: label, label: `${model.name} (${model.provider})` };
        });
    },

    async handler(args: string, ctx) {
      const config = loadConfig(ctx.cwd);
      const allModels = listInputImageModels(config);

      if (allModels.length === 0) {
        ctx.ui.notify(
          "No image models that accept an input image were found. Check your connection (capabilities are discovered from OpenRouter) or add edit-capable models to settings/providers.",
          "error",
        );
        return;
      }

      // Direct selection: /image-edit-model openai/gpt-image-1
      if (args.trim()) {
        const input = args.trim();
        const slashIdx = input.indexOf("/");
        let found: ImageModel | undefined;
        if (slashIdx > 0) {
          const provider = input.slice(0, slashIdx);
          const modelId = input.slice(slashIdx + 1);
          found = allModels.find((m) => m.provider === provider && m.id === modelId);
          if (!found) {
            found = allModels.find((m) => m.id === input);
          }
        } else {
          found = allModels.find((m) => m.id === input);
        }

        if (!found) {
          ctx.ui.notify(
            `Unknown edit-capable image model "${input}". Run /image-edit-model to see options.`,
            "error",
          );
          return;
        }

        editModelSelection = { provider: found.provider, model: found.id };
        pi.appendEntry(SESSION_ENTRY_EDIT_TYPE, editModelSelection);
        await saveDefaultEditModel(found.provider, found.id);
        ctx.ui.notify(`Image edit model: ${found.name} (${found.provider}/${found.id})`, "info");
        return;
      }

      // Interactive selection — only models that accept input images
      const currentKey = editModelSelection ? `${editModelSelection.provider}/${editModelSelection.model}` : null;
      const selected = await pickModelWithUI(
        ctx,
        allModels,
        currentKey,
        "Select image edit model (accepts input images)",
      );
      if (!selected) return;

      editModelSelection = { provider: selected.provider, model: selected.id };
      pi.appendEntry(SESSION_ENTRY_EDIT_TYPE, editModelSelection);
      await saveDefaultEditModel(selected.provider, selected.id);
      ctx.ui.notify(`Image edit model: ${selected.name} (${selected.provider}/${selected.id})`, "info");
    },
  });

  // ── generate_image tool ──

  pi.registerTool({
    name: "generate_image",
    label: "Generate image",
    description:
      "Generate an image from a text prompt, or edit an existing image when inputImage is provided. " +
      "Uses the model selected via /image-model (or the edit-capable model from /image-edit-model when editing). " +
      "Do NOT pass provider or model parameters — just provide the prompt.",
    parameters: TOOL_PARAMS,

    async execute(
      _toolCallId: string,
      params: ToolParams,
      signal: AbortSignal | undefined,
      onUpdate: AgentToolUpdateCallback<unknown> | undefined,
      ctx: ExtensionContext,
    ): Promise<AgentToolResult<unknown>> {
      const config = loadConfig(ctx.cwd);
      const inputPaths = [
        ...(params.inputImage ? [params.inputImage] : []),
        ...(params.inputImages ?? []),
      ];
      const inputImages = inputPaths.length > 0
        ? inputPaths.map((p) => loadInputImageDataUrl(p, ctx.cwd))
        : undefined;
      const generationModel = resolveModel(sessionSelection, config);
      const imageModel = inputImages
        ? resolveEditModel(editModelSelection, config, generationModel)
        : generationModel;
      const aspectRatio = params.aspectRatio || "1:1";
      const imageSize = params.imageSize || "1K";
      const quality = params.quality || "auto";

      // Build a combined signal: framework abort + configured timeout
      const timeoutMs = (params.timeout ?? config.timeout ?? DEFAULT_TIMEOUT_MS / 1000) * 1000;
      const timeoutSignal = AbortSignal.timeout(timeoutMs);
      const combinedSignal = signal
        ? AbortSignal.any([signal, timeoutSignal])
        : timeoutSignal;

      onUpdate?.({
        content: [{ type: "text", text: `${inputImages ? "Editing image with" : "Generating image with"} ${imageModel.provider}/${imageModel.id}...` }],
        details: { provider: imageModel.provider, model: imageModel.id, api: imageModel.api, editing: !!inputImages },
      });

      const providerConfig = await resolveProviderConfig(imageModel, ctx, config);
      const image = await generateImage(providerConfig, imageModel, params.prompt, aspectRatio, imageSize, quality, combinedSignal, inputImages);

      // Record spend: footer cost total + /session breakdown + footer image-cost chip
      const billedCost = recordImageUsage(ctx, imageModel, image);

      // Save if configured
      const saveConfig = resolveSaveConfig(params.save, params.saveDir, ctx.cwd, config);
      let savedPath: string | undefined;
      let saveError: string | undefined;
      if (saveConfig.mode !== "none" && saveConfig.outputDir) {
        try {
          savedPath = await saveImageToDisk(image.data, image.mimeType, saveConfig.outputDir);
        } catch (error) {
          saveError = error instanceof Error ? error.message : String(error);
        }
      }

      const summary: string[] = [
        inputPaths.length > 0 ? `Edited image using ${inputPaths.join(", ")}.` : `Generated image.`,
        `Aspect ratio: ${aspectRatio}.`,
      ];
      if (savedPath) summary.push(`Saved to: ${savedPath}`);
      if (saveError) summary.push(`Save failed: ${saveError}`);
      if (billedCost > 0) summary.push(`Cost: $${billedCost.toFixed(4)}.`);

      return {
        content: [
          { type: "text", text: summary.join(" ") },
          { type: "image", data: image.data, mimeType: image.mimeType },
        ],
        details: {
          provider: imageModel.provider,
          model: imageModel.id,
          api: imageModel.api,
          inputImage: params.inputImage,
          inputPaths: inputPaths.length > 0 ? inputPaths : undefined,
          aspectRatio,
          quality,
          savedPath,
          saveMode: saveConfig.mode,
          cost: billedCost,
        },
      };
    },
  });
}
