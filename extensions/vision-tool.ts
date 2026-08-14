/**
 * Vision Tool — delegates image analysis to a vision-capable model.
 *
 * Non-multimodal models (like DeepSeek Pro, GPT-5 Codex without image support, etc.)
 * can call this tool whenever they need to understand an image.
 *
 * The tool sends the image to a configurable vision model,
 * collects the full text response, and returns it to the calling model as a tool result.
 *
 * ## Configuration
 *
 * The vision model is resolved from Pi's model registry (models.json).
 *
 * **Recommended: /vision command (persistent)**
 *   Use `/vision config provider my-provider` and `/vision config model my-vision-model`
 *   to set the vision model. Settings are saved to `~/.pi/agent/vision-tool.json`
 *   and persist across all sessions. Run `/vision` with no arguments to see
 *   current configuration. Changes take effect immediately — no /reload needed.
 *
 * **Legacy: environment variables**
 *   PI_VISION_PROVIDER=my-provider  PI_VISION_MODEL=my-vision-model
 *   Env vars are read at session start as a fallback when no config file exists.
 *
 * **Priority:** /vision config settings > env vars > built-in defaults
 *
 * Make sure the provider and model are defined in ~/.pi/agent/models.json
 * with `input: ["text", "image"]`.
 *
 * ## Reasoning / Thinking
 *
 * For vision models that support extended thinking (marked with `reasoning: true`
 * in models.json), the tool can request reasoning effort levels:
 *   - "off"    — disabled, fast responses (default)
 *   - "minimal", "low", "medium", "high", "xhigh" — increasing reasoning depth
 *
 * The calling model decides per-call which level to use via the `reasoning`
 * parameter. It should use "off" for simple queries (what color is this?)
 * and higher levels for complex analysis (architecture diagrams, bug hunting).
 *
 * Default level is configurable via:
 *   /vision config reasoning-effort <level>
 *   PI_VISION_REASONING_EFFORT=medium
 *
 * ## Compression
 *
 * Images are automatically preprocessed to reduce payload size and token count:
 * - Downscaled to 1568px max dimension (configurable via PI_VISION_MAX_DIM)
 * - Alpha channel stripped (RGBA → RGB)
 * - Lossless PNG converted to JPEG (quality 85, configurable via PI_VISION_JPEG_QUALITY)
 *
 * Set PI_VISION_COMPRESS=false to disable all preprocessing (send raw bytes).
 * Requires `sharp` for image processing. Falls back to raw bytes if not installed.
 *
 * ## Usage
 *
 * The `prompt` parameter is a free-text instruction, so the calling model can ask
 * for exactly what it needs:
 *
 * - Description: "Describe everything visible in this image"
 * - Coordinates: "Give pixel coordinates [x,y,w,h] of the red button"
 * - Text: "Extract all visible text, preserving structure"
 * - Analysis: "Is there a compiler error shown? What does it say?"
 * - UI: "List all interactive elements and their states"
 * - Colors: "What hex color is the header bar?"
 * - Comparison: "Compare these two screenshots — what changed?"
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { Model } from "@earendil-works/pi-ai";
import type { Api } from "@earendil-works/pi-ai";
import { Text } from "@earendil-works/pi-tui";

// ---------------------------------------------------------------------------
// Config file path
// ---------------------------------------------------------------------------
const CONFIG_PATH = join(getAgentDir(), "vision-tool.json");

// ---------------------------------------------------------------------------
// Reasoning effort levels
// ---------------------------------------------------------------------------
const REASONING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;
type ReasoningLevel = (typeof REASONING_LEVELS)[number];

// ---------------------------------------------------------------------------
// Runtime config (mutable, populated on session_start)
// ---------------------------------------------------------------------------
interface VisionConfig {
  provider?: string;
  model?: string;
  maxDimension: number;
  jpegQuality: number;
  defaultReasoningEffort: ReasoningLevel;
  enabled: boolean;
  /** locate 模式粗定位输入图最大宽度(越小越快,默认1024) */
  locateMaxWidth: number;
  /** 禁用模型思考(发送 {"thinking":{"type":"disabled"}}),豆包等默认带思考的模型必须开启才快 */
  thinkingDisabled: boolean;
  /** 附加请求参数,任意 OpenAI 兼容 API 的特殊参数兜底(如 {"enable_thinking":false}) */
  extraBody: Record<string, unknown>;
  /** 精定位裁剪尺寸与放大倍数 */
  fineCropWidth: number;
  fineCropHeight: number;
  fineZoom: number;
  /** 阶梯式备用视觉模型: 主模型调用失败时按顺序尝试 [{provider, model}, ...] */
  fallbackModels: { provider: string; model: string }[];
}

let config: VisionConfig = {
  maxDimension: parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10),
  jpegQuality: parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10),
  defaultReasoningEffort: "off",
  enabled: true,
  locateMaxWidth: 1024,
  thinkingDisabled: false,
  extraBody: {},
  fineCropWidth: 320,
  fineCropHeight: 240,
  fineZoom: 3,
  fallbackModels: [],
};

const VISION_SYSTEM_PROMPT = [
  "You are an expert vision analysis assistant.",
  "Examine the provided image and respond to the user's request precisely.",
  "",
  "Guidelines:",
  "- If asked for a description, describe everything you see thoroughly.",
  "- If asked for pixel coordinates of elements, provide them in [x, y, width, height] format.",
  "- If asked to read text, extract all visible text verbatim.",
  "- If asked about UI elements, describe their appearance, position, and state.",
  "- Be precise and factual. Do not invent details that are not in the image.",
  "- Structure your response clearly with markdown formatting when appropriate.",
].join("\n");

// ---------------------------------------------------------------------------
// Config persistence helpers
// ---------------------------------------------------------------------------

/** Load config from the JSON file. Returns null if file doesn't exist. */
function loadConfigFile(): VisionConfig | null {
  try {
    if (!existsSync(CONFIG_PATH)) return null;
    const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    return {
      provider: raw.provider || undefined,
      model: raw.model || undefined,
      maxDimension: raw.maxDimension ?? config.maxDimension,
      jpegQuality: raw.jpegQuality ?? config.jpegQuality,
      defaultReasoningEffort: validateReasoningLevel(raw.defaultReasoningEffort) ?? config.defaultReasoningEffort,
      enabled: raw.enabled !== false,
      locateMaxWidth: raw.locateMaxWidth ?? config.locateMaxWidth,
      thinkingDisabled: raw.thinkingDisabled ?? config.thinkingDisabled,
      extraBody: raw.extraBody ?? config.extraBody,
      fineCropWidth: raw.fineCropWidth ?? config.fineCropWidth,
      fineCropHeight: raw.fineCropHeight ?? config.fineCropHeight,
      fineZoom: raw.fineZoom ?? config.fineZoom,
      fallbackModels: Array.isArray(raw.fallbackModels) ? raw.fallbackModels : config.fallbackModels,
    };
  } catch {
    return null;
  }
}

/** Save current config to the JSON file. */
function saveConfigFile() {
  try {
    mkdirSync(dirname(CONFIG_PATH), { recursive: true });
  } catch {
    // directory already exists or no perms — ignore
  }
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
}

/**
 * Validate and normalize a reasoning level string.
 */
function validateReasoningLevel(value: string | undefined): ReasoningLevel | undefined {
  if (!value) return undefined;
  const normalized = value.toLowerCase();
  if ((REASONING_LEVELS as readonly string[]).includes(normalized)) {
    return normalized as ReasoningLevel;
  }
  return undefined;
}

/**
 * Resolve config with priority:
 *   1. Config file (~/.pi/agent/vision-tool.json)
 *   2. Environment variables (PI_VISION_PROVIDER, PI_VISION_MODEL, etc.)
 *   3. Built-in defaults
 *
 * The file wins over env vars so that /vision config changes are sticky.
 */
function resolveConfig(): VisionConfig {
  const fileCfg = loadConfigFile();
  const envReasoning = validateReasoningLevel(process.env.PI_VISION_REASONING_EFFORT);
  return {
    provider: fileCfg?.provider || process.env.PI_VISION_PROVIDER || undefined,
    model: fileCfg?.model || process.env.PI_VISION_MODEL || undefined,
    maxDimension:
      fileCfg?.maxDimension ??
      parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10),
    jpegQuality:
      fileCfg?.jpegQuality ??
      parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10),
    defaultReasoningEffort:
      fileCfg?.defaultReasoningEffort ?? envReasoning ?? "off",
    enabled: fileCfg?.enabled !== false,
    locateMaxWidth: fileCfg?.locateMaxWidth ?? config.locateMaxWidth,
    thinkingDisabled: fileCfg?.thinkingDisabled ?? config.thinkingDisabled,
    extraBody: fileCfg?.extraBody ?? config.extraBody,
    fineCropWidth: fileCfg?.fineCropWidth ?? config.fineCropWidth,
    fineCropHeight: fileCfg?.fineCropHeight ?? config.fineCropHeight,
    fineZoom: fileCfg?.fineZoom ?? config.fineZoom,
    fallbackModels: fileCfg?.fallbackModels ?? config.fallbackModels,
  };
}

/**
 * Build a human-readable config summary for the /vision command.
 */
function configSummary(): string {
  const src = loadConfigFile() ? "config file" : process.env.PI_VISION_PROVIDER ? "env vars" : "none";
  const provider = config.provider ?? "(not set)";
  const model = config.model ?? "(not set)";
  return [
    `Vision tool configuration (source: ${src})`,
    `  Provider:          ${provider}`,
    `  Model:             ${model}`,
    `  Max dim:           ${config.maxDimension}px`,
    `  JPEG quality:      ${config.jpegQuality}`,
    `  Reasoning effort:  ${config.defaultReasoningEffort}`,
    `  Enabled:           ${config.enabled ? "yes" : "no"}`,
    ``,
    `Config file: ${CONFIG_PATH}`,
    ``,
    "Use /vision config <setting> <value> to set:",
    "  provider, model, max-dim, quality, reasoning-effort",
    "Use /vision on|off to enable or disable the tool",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Image processing
// ---------------------------------------------------------------------------

async function imageToBase64(
  pathOrData: string,
  compress: boolean,
): Promise<{ mimeType: string; data: string }> {
  // If it looks like a base64 data URL, parse it
  if (pathOrData.startsWith("data:")) {
    const match = /^data:([^;]+);base64,(.+)$/.exec(pathOrData);
    if (match) {
      const buffer = Buffer.from(match[2], "base64");
      return compress
        ? await optimizeImage(buffer, match[1])
        : { mimeType: match[1], data: match[2] };
    }
  }

  // If it's raw base64 without a data URL prefix, try to detect
  if (/^[A-Za-z0-9+/=]+$/.test(pathOrData) && pathOrData.length > 100) {
    const buffer = Buffer.from(pathOrData, "base64");
    return compress
      ? await optimizeImage(buffer, "image/png")
      : { mimeType: "image/png", data: pathOrData };
  }

  // Otherwise treat as a file path
  const resolvedPath = resolve(pathOrData);
  const buffer = await readFile(resolvedPath);
  const ext = resolvedPath.split(".").pop()?.toLowerCase() ?? "png";
  const mimeMap: Record<string, string> = {
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    gif: "image/gif",
    webp: "image/webp",
    bmp: "image/bmp",
  };
  const mimeType = mimeMap[ext] ?? "image/png";
  return compress
    ? await optimizeImage(buffer, mimeType)
    : { mimeType, data: buffer.toString("base64") };
}

/**
 * ── Calibration locate mode ────────────────────────────────────────────────
 *
 * Vision models see images on an internal grid (often ~1000×1000) and are
 * unreliable at absolute pixel coordinates. `locate: true` fixes this:
 *
 *   1. Draw 4 red cross calibration marks at known positions (corners, 40px in)
 *   2. Ask the model to report the cross centers + any element coordinates
 *   3. Fit a diagonal affine transform (x_real = a·x_model + c, y_real = b·y_model + d)
 *      from the cross centers (median over point pairs, robust)
 *   4. Map element coordinates into real pixels — accurate to ~2px in practice
 *
 * Falls back to a 1000×1000 model-space assumption (x_real = W/1000·x, …) when
 * sharp is unavailable or fewer than 2 calibration crosses are detected.
 */

const CALIB_INSET = 40; // calibration cross distance from image edges

interface CalibMark {
  name: "TL" | "TR" | "BL" | "BR";
  x: number;
  y: number;
}

function calibMarksFor(width: number, height: number): CalibMark[] {
  return [
    { name: "TL", x: CALIB_INSET, y: CALIB_INSET },
    { name: "TR", x: width - CALIB_INSET, y: CALIB_INSET },
    { name: "BL", x: CALIB_INSET, y: height - CALIB_INSET },
    { name: "BR", x: width - CALIB_INSET, y: height - CALIB_INSET },
  ];
}

/**
 * Zero-dependency image dimension sniffing (PNG/JPEG/GIF/WebP headers).
 * Used as fallback when sharp is unavailable in locate mode.
 */
function getImageSize(buf: Buffer): { width: number; height: number } | null {
  // PNG: 8-byte signature + IHDR
  if (buf.length >= 24 && buf.readUInt32BE(0) === 0x89504e47) {
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  }
  // GIF
  if (buf.length >= 10 && (buf.toString("ascii", 0, 6) === "GIF87a" || buf.toString("ascii", 0, 6) === "GIF89a")) {
    return { width: buf.readUInt16LE(6), height: buf.readUInt16LE(8) };
  }
  // JPEG: scan for SOFn marker
  if (buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let off = 2;
    while (off + 9 < buf.length) {
      if (buf[off] !== 0xff) { off++; continue; }
      const marker = buf[off + 1];
      if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) { off += 2; continue; }
      if (marker >= 0xd0 && marker <= 0xd7) { off += 2; continue; }
      const len = buf.readUInt16BE(off + 2);
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: buf.readUInt16BE(off + 5), width: buf.readUInt16BE(off + 7) };
      }
      off += 2 + len;
    }
  }
  return null;
}

/**
 * Draw calibration crosses onto the image (via sharp SVG composite) and
 * return the marked image + original dimensions. JPEG-encodes the result
 * (pixel grid is unchanged, so coordinates stay valid). Returns null when
 * sharp is unavailable.
 *
 * Optionally downscales to `maxWidth` (locate 粗定位提速: 大图会让视觉模型思考极慢),
 * 并返回 scaleX/scaleY 供坐标映射回原图。
 */
async function addCalibrationMarks(
  buffer: Buffer,
  maxWidth?: number,
): Promise<{ buffer: Buffer; width: number; height: number; marks: CalibMark[]; scaleX: number; scaleY: number } | null> {
  try {
    const sharp = (await import("sharp")).default;
    const meta = await sharp(buffer).metadata();
    const origW = meta.width ?? 0;
    const origH = meta.height ?? 0;
    if (!origW || !origH) return null;

    let width = origW;
    let height = origH;
    let scaleX = 1;
    let scaleY = 1;
    if (maxWidth && maxWidth > 0 && origW > maxWidth) {
      width = maxWidth;
      height = Math.round(origH * (maxWidth / origW));
      scaleX = origW / width;
      scaleY = origH / height;
    }

    const half = 14; // cross half-length
    const marks = calibMarksFor(width, height);
    const crosses = marks
      .map(
        (m) =>
          `<line x1="${m.x - half}" y1="${m.y}" x2="${m.x + half}" y2="${m.y}" stroke="red" stroke-width="4"/>` +
          `<line x1="${m.x}" y1="${m.y - half}" x2="${m.x}" y2="${m.y + half}" stroke="red" stroke-width="4"/>`,
      )
      .join("");
    const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${crosses}</svg>`;

    const out = await sharp(buffer)
      .resize(width, height, { kernel: "lanczos3" })
      .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
      .jpeg({ quality: 90 })
      .toBuffer();
    return { buffer: out, width, height, marks, scaleX, scaleY };
  } catch {
    return null;
  }
}

/**
 * Append calibration instructions to the user prompt so the model reports
 * cross centers + element coordinates in a parseable format.
 */
function buildCalibrationPrompt(prompt: string, width: number, height: number): string {
  return (
    prompt +
    `\n\n[坐标校准] 图片四角有4个红色十字校准标记：左上TL、右上TR、左下BL、右下BR` +
    `（实际像素位置：TL(${CALIB_INSET},${CALIB_INSET}) TR(${width - CALIB_INSET},${CALIB_INSET}) ` +
    `BL(${CALIB_INSET},${height - CALIB_INSET}) BR(${width - CALIB_INSET},${height - CALIB_INSET})）。\n` +
    `请先用以下严格格式输出4个十字中心的像素坐标（每行一个）：\n` +
    `TL: x,y\nTR: x,y\nBL: x,y\nBR: x,y\n` +
    `然后回答我的问题。如果问题涉及图中某个元素的位置，请额外用` +
    `“元素名: x1,y1,x2,y2”（矩形框）或“元素名: x,y”（中心点）格式输出其像素坐标。`
  );
}

interface ExtractedPoint {
  name: string;
  x1: number;
  y1: number;
  x2?: number;
  y2?: number;
}

/** Parse "Name: x,y" or "Name: x1,y1,x2,y2" lines (also Chinese colon). */
function extractPointLines(text: string): ExtractedPoint[] {
  const re = /([A-Za-z0-9_\u4e00-\u9fa5]{1,20})\s*[:：]\s*(\d{1,5})\s*[,\s，]+\s*(\d{1,5})(?:\s*[,\s，]+\s*(\d{1,5})\s*[,\s，]+\s*(\d{1,5}))?/g;
  const out: ExtractedPoint[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    out.push({
      name: m[1],
      x1: Number(m[2]),
      y1: Number(m[3]),
      x2: m[4] !== undefined ? Number(m[4]) : undefined,
      y2: m[5] !== undefined ? Number(m[5]) : undefined,
    });
  }
  // 兼容自由坐标格式(如智谱 GLM-4V-Flash 的 "x=420, y=150" / "x:420 y:150")
  if (!out.some((p) => p.name === "目标" || p.name === "Target")) {
    const re2 = /x\s*[:=＝]\s*(\d{1,5})[^\d]{1,12}?y\s*[:=＝]\s*(\d{1,5})/gi;
    let m2: RegExpExecArray | null;
    while ((m2 = re2.exec(text)) !== null) {
      out.push({ name: "目标", x1: Number(m2[1]), y1: Number(m2[2]), x2: undefined, y2: undefined });
    }
  }
  return out;
}

interface Affine {
  a: number;
  b: number;
  c: number;
  d: number;
}

/**
 * Fit diagonal affine from calibration crosses: median of pair-wise slopes.
 * Returns null when fewer than 2 crosses are usable.
 */
function fitAffine(calib: Map<string, { x: number; y: number }>, marks: CalibMark[]): Affine | null {
  const known = marks.filter((mk) => calib.has(mk.name));
  if (known.length < 2) return null;

  const aVals: number[] = [];
  const cVals: number[] = [];
  const bVals: number[] = [];
  const dVals: number[] = [];
  for (let i = 0; i < known.length; i++) {
    for (let j = i + 1; j < known.length; j++) {
      const m1 = calib.get(known[i].name)!;
      const m2 = calib.get(known[j].name)!;
      const r1 = known[i];
      const r2 = known[j];
      const dxm = m2.x - m1.x;
      const dym = m2.y - m1.y;
      if (Math.abs(dxm) > 2) {
        const a = (r2.x - r1.x) / dxm;
        aVals.push(a);
        cVals.push(r1.x - a * m1.x);
      }
      if (Math.abs(dym) > 2) {
        const b = (r2.y - r1.y) / dym;
        bVals.push(b);
        dVals.push(r1.y - b * m1.y);
      }
    }
  }
  if (!aVals.length || !bVals.length) return null;

  const median = (arr: number[]) => [...arr].sort((p, q) => p - q)[Math.floor(arr.length / 2)];
  return { a: median(aVals), b: median(bVals), c: median(cVals), d: median(dVals) };
}

/** Map a model-space point into real pixels. */
function toReal(affine: Affine, x: number, y: number): { x: number; y: number } {
  return { x: affine.a * x + affine.c, y: affine.b * y + affine.d };
}

/**
 * Main locate pipeline: mark image → build prompt → (caller sends request) →
 * parse + calibrate the model's output and return a calibration report block.
 */
function buildLocateReport(
  modelText: string,
  width: number,
  height: number,
  marks: CalibMark[],
  scaleX = 1,
  scaleY = 1,
): string {
  const points = extractPointLines(modelText);
  const calib = new Map<string, { x: number; y: number }>();
  for (const p of points) {
    if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") {
      calib.set(p.name, { x: (p.x1 + (p.x2 ?? p.x1)) / 2, y: (p.y1 + (p.y2 ?? p.y1)) / 2 });
    }
  }

  let affine = fitAffine(calib, marks);
  let method = "cross-calibration";
  if (!affine) {
    // Fallback: assume model space is 1000×1000 (observed for doubao-seed-2-1-turbo).
    affine = { a: width / 1000, b: height / 1000, c: 0, d: 0 };
    method = "assumed-1000x1000";
  }

  const rows: string[] = [];
  for (const p of points) {
    if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") continue;
    const r1 = toReal(affine, p.x1, p.y1);
    const rx1 = { x: Math.round(r1.x * scaleX), y: Math.round(r1.y * scaleY) };
    const rx2 = p.x2 !== undefined
      ? (() => {
          const r2 = toReal(affine, p.x2!, p.y2!);
          return { x: Math.round(r2.x * scaleX), y: Math.round(r2.y * scaleY) };
        })()
      : undefined;
    rows.push(
      rx2
        ? `${p.name}: (${rx1.x},${rx1.y},${rx2.x},${rx2.y})`
        : `${p.name}: (${rx1.x},${rx1.y})`,
    );
  }

  if (!rows.length) {
    return "[locate] 未在模型输出中找到可解析的元素坐标行（格式：元素名: x1,y1,x2,y2）。请重试或换用普通描述模式。";
  }
  return `[locate 校准坐标（真实像素，${method === "cross-calibration" ? "十字校准" : "默认1000×1000空间假设"}，可直接用于点击）]\n${rows.join("\n")}`;
}

/**
 * ── Multi-stage fine locate (V2) ───────────────────────────────────────────
 *
 * Coarse locate gives ±50-100px. This refines each target by:
 *   1. Cropping a small region around the coarse center (e.g. 320×240)
 *   2. Upscaling it 4× so the model sees a large, detailed local view
 *   3. Drawing fresh calibration crosses on the LOCAL crop (known local coords)
 *   4. Asking the model for the exact center of the target inside the crop
 *   5. Mapping local → physical pixels via the local affine
 *
 * Result: ±3-8px accuracy — good enough for clicking.
 *
 * `extraPrompt` (the original user prompt) is used to describe the target to
 * the model so it knows WHAT to pinpoint.
 */
async function refineLocate(
  rawBuffer: Buffer,
  targets: { name: string; cx: number; cy: number; desc?: string }[],
  visionModel: any,
  apiKey: string,
  signal: AbortSignal | undefined,
  reasoningLevel: any,
  extraPrompt: string,
): Promise<string> {
  let sharp: any = null;
  try {
    sharp = (await import("sharp")).default;
  } catch {
    return "[locate-fine] sharp 不可用，跳过精定位（粗定位结果见上）";
  }

  const meta = await sharp(rawBuffer).metadata();
  const W = meta.width ?? 0;
  const H = meta.height ?? 0;
  if (!W || !H) return "[locate-fine] 无法读取原图尺寸";

  const lines: string[] = [];
  for (const t of targets) {
    // 1) 裁剪区域（原图物理坐标）; 尺寸可配置(locateMaxWidth 提速场景下适当调小)
    const cw = Math.min(config.fineCropWidth, W);
    const ch = Math.min(config.fineCropHeight, H);
    let x1 = Math.round(t.cx - cw / 2);
    let y1 = Math.round(t.cy - ch / 2);
    x1 = Math.max(0, Math.min(W - cw, x1));
    y1 = Math.max(0, Math.min(H - ch, y1));
    const x2 = x1 + cw;
    const y2 = y1 + ch;

    try {
      // 2) 裁剪并放大 (zoom 可配置, 默认 3×)
      const zoom = config.fineZoom;
      const cropped = await sharp(rawBuffer)
        .extract({ left: x1, top: y1, width: cw, height: ch })
        .resize(cw * zoom, ch * zoom, { kernel: "lanczos3" })
        .jpeg({ quality: 92 })
        .toBuffer();

      // 3) 在局部图上画 4 个十字校准点（局部坐标，位置已知）
      const zw = cw * zoom;
      const zh = ch * zoom;
      const inset = 30;
      const marks = [
        { name: "TL", x: inset, y: inset },
        { name: "TR", x: zw - inset, y: inset },
        { name: "BL", x: inset, y: zh - inset },
        { name: "BR", x: zw - inset, y: zh - inset },
      ];
      const crosses = marks
        .map(
          (m) =>
            `<line x1="${m.x - 16}" y1="${m.y}" x2="${m.x + 16}" y2="${m.y}" stroke="red" stroke-width="5"/>` +
            `<line x1="${m.x}" y1="${m.y - 16}" x2="${m.x}" y2="${m.y + 16}" stroke="red" stroke-width="5"/>`,
        )
        .join("");
      const svg = `<svg width="${zw}" height="${zh}" xmlns="http://www.w3.org/2000/svg">${crosses}</svg>`;
      const marked = await sharp(cropped).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toBuffer();

      // 4) 请求精定位
      const targetDesc = t.desc
        ? `目标描述：${t.desc}（用户问题中的元素）`
        : `上一步定位到的元素"${t.name}"`;
      const finePrompt =
        `[精确定位] 这张图是原截图局部区域的 4× 放大图，原始区域 x∈[${x1},${x2}] y∈[${y1},${y2}]。\n` +
        `图四角有4个红色十字校准点：TL/TR/BL/BR（局部坐标：TL(${inset},${inset}) TR(${zw - inset},${inset}) BL(${inset},${zh - inset}) BR(${zw - inset},${zh - inset})）。\n` +
        `请先输出：TL: x,y / TR: x,y / BL: x,y / BR: x,y\n` +
        `然后精确定位${targetDesc}的中心点，输出：${t.name}: x,y\n` +
        `坐标必须是本图（${zw}×${zh}）内的像素坐标。`;

      const fineResult = await callVisionModel(
        visionModel,
        apiKey,
        marked.toString("base64"),
        "image/jpeg",
        finePrompt,
        signal,
        reasoningLevel,
        true, // 精定位也是定位流程, 强制禁用思考
      );

      // 5) 解析局部坐标并映射回物理坐标
      const pts = extractPointLines(fineResult);
      const calib = new Map<string, { x: number; y: number }>();
      for (const p of pts) {
        if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") {
          calib.set(p.name, { x: (p.x1 + (p.x2 ?? p.x1)) / 2, y: (p.y1 + (p.y2 ?? p.y1)) / 2 });
        }
      }
      const aff = fitAffine(calib, marks as any);
      // 目标匹配放宽: 精确点名 → "目标" → 任意非十字点(兼容智谱等模型输出不同点名/自由格式)
      const target =
        pts.find((p) => p.name === t.name) ||
        pts.find((p) => p.name === "目标") ||
        pts.find((p) => !["TL", "TR", "BL", "BR"].includes(p.name));
      if (!aff || !target) {
        lines.push(`${t.name}: 精定位失败（模型未返回可解析坐标）`);
        continue;
      }
      const cxLocal = (target.x1 + (target.x2 ?? target.x1)) / 2;
      const cyLocal = (target.y1 + (target.y2 ?? target.y1)) / 2;
      const r = toReal(aff, cxLocal, cyLocal);
      const px = x1 + r.x / zoom;
      const py = y1 + r.y / zoom;
      lines.push(`${t.name}: (${Math.round(px)},${Math.round(py)})`);
    } catch (e) {
      lines.push(`${t.name}: 精定位异常 ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  return `[locate-fine 精确定位坐标（多阶段校准，可直接点击）]\n${lines.join("\n")}`;
}

/**
 * Optimize an image before sending to the vision model.
 * - Downscales if larger than config.maxDimension on either axis
 * - Strips alpha channel (RGBA → RGB)
 * - Converts lossless PNG to JPEG for smaller payload
 * Falls back to raw bytes if sharp is not available.
 */
async function optimizeImage(
  buffer: Buffer,
  originalMime: string,
): Promise<{ mimeType: string; data: string }> {
  if (buffer.length === 0) {
    return { mimeType: originalMime, data: "" };
  }

  try {
    // Dynamic import — users who don't have sharp installed get raw bytes
    const sharp = (await import("sharp")).default;
    let pipeline = sharp(buffer);
    const metadata = await pipeline.metadata();
    const width = metadata.width ?? 0;
    const height = metadata.height ?? 0;

    // Downscale if needed
    if (width > config.maxDimension || height > config.maxDimension) {
      pipeline = pipeline.resize(config.maxDimension, config.maxDimension, {
        fit: "inside",
        withoutEnlargement: true,
      });
    }

    // Strip alpha — vision models often don't need it and it wastes tokens
    if (metadata.hasAlpha || metadata.channels === 4) {
      pipeline = pipeline.removeAlpha();
    }

    // Convert to JPEG for smaller payload (except GIF)
    if (originalMime !== "image/gif") {
      const optimized = await pipeline.jpeg({ quality: config.jpegQuality }).toBuffer();
      return { mimeType: "image/jpeg", data: optimized.toString("base64") };
    }

    // Keep GIF as-is (sharp can't re-encode animated GIF well)
    const optimized = await pipeline.toBuffer();
    return { mimeType: originalMime, data: optimized.toString("base64") };
  } catch {
    // sharp not available or decode failed — send raw bytes
    return { mimeType: originalMime, data: buffer.toString("base64") };
  }
}

/**
 * Build the reasoning/thinking parameters for the API request.
 *
 * Only sends reasoning params when:
 * - The vision model has `reasoning: true`
 * - The effective reasoning level (after thinkingLevelMap) is not mapped away
 *
 * Respects the model's `thinkingLevelMap` if defined,
 * and the model's `compat.thinkingFormat` for provider-specific formats:
 * - `qwen` / `qwen-chat-template`: sends `enable_thinking: true/false`
 * - `deepseek`: sends `reasoning: { effort }`
 * - `openrouter`: sends `reasoning: { effort }`
 * - `together`: sends `reasoning: { enabled: boolean }`
 * - default (OpenAI): sends `reasoning_effort`
 */
function buildReasoningParams(
  visionModel: Model<Api>,
  level: ReasoningLevel,
): Record<string, unknown> | undefined {
  if (!visionModel.reasoning) return undefined;

  // Resolve the effective value from thinkingLevelMap if present.
  // thinkingLevelMap maps pi's level names ("off", "minimal", etc.) to
  // provider-specific values. `null` means the level is unsupported.
  const levelMap = (visionModel as any).thinkingLevelMap as
    | Record<string, string | null>
    | undefined;

  let effectiveLevel: string | null = level;
  let mappedAway = false;

  if (levelMap) {
    const mapped = levelMap[level];
    if (mapped === null) {
      // Level explicitly unsupported — skip reasoning params entirely
      return undefined;
    }
    if (mapped !== undefined) {
      effectiveLevel = mapped;
    }
  }

  // After mapping, if the effective level means "off" (e.g. "none" for Kimi),
  // we still need to send it — some providers require explicit "none" to
  // disable reasoning that would otherwise be on by default.
  // Only skip if thinkingLevelMap explicitly mapped to null above.

  // Determine the parameter format based on compat.thinkingFormat
  const compat = (visionModel as any).compat as
    | { thinkingFormat?: string }
    | undefined;
  const format = compat?.thinkingFormat;

  if (format === "qwen" || format === "qwen-chat-template") {
    // Qwen models use enable_thinking: true/false
    // When mapped to a non-standard value (e.g. thinkingLevelMap: {"off": "none"}),
    // treat any value that isn't literally "off" as enabling thinking.
    const enable = effectiveLevel !== "off" && effectiveLevel !== "none";
    if (format === "qwen-chat-template") {
      return { chat_template_kwargs: { enable_thinking: enable } };
    }
    return { enable_thinking: enable };
  }

  if (format === "deepseek" || format === "openrouter") {
    return { reasoning: { effort: effectiveLevel } };
  }

  if (format === "together") {
    // Together AI uses reasoning.enabled bool + reasoning_effort
    const enabled = effectiveLevel !== "off" && effectiveLevel !== "none";
    if (!enabled) {
      return { reasoning: { enabled: false } };
    }
    return { reasoning: { enabled: true }, reasoning_effort: effectiveLevel };
  }

  // Default: standard OpenAI reasoning_effort
  return { reasoning_effort: effectiveLevel };
}

async function callVisionModel(
  visionModel: Model<Api>,
  apiKey: string | undefined,
  imageBase64: string,
  mimeType: string,
  prompt: string,
  signal: AbortSignal | undefined,
  reasoningLevel: ReasoningLevel,
  forceNoThinking = false,
): Promise<string> {
  const baseUrl = visionModel.baseUrl.replace(/\/+$/, "");

  const messages = [
    {
      role: "system",
      content: VISION_SYSTEM_PROMPT,
    },
    {
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: {
            url: `data:${mimeType};base64,${imageBase64}`,
          },
        },
        {
          type: "text",
          text: prompt,
        },
      ],
    },
  ];

  // locate 模式强制禁用思考(保速度); 描述模式由 config.thinkingDisabled 决定
  const noThinking = forceNoThinking || config.thinkingDisabled;
  // 禁用思考时不再发送 reasoning 参数(避免与 thinking:disabled 冲突,豆包等模型会报错)
  const reasoningParams = noThinking
    ? undefined
    : buildReasoningParams(visionModel, reasoningLevel);

  const body: Record<string, unknown> = {
    model: visionModel.id,
    messages,
    max_tokens: 4096,
    temperature: 0,
  };

  if (reasoningParams) {
    Object.assign(body, reasoningParams);
  }
  // 兼容任意视觉 API: 需要时显式禁用思考(豆包等默认带思考极慢); extraBody 兜底自定义参数
  if (noThinking) {
    body.thinking = { type: "disabled" };
  }
  if (config.extraBody && typeof config.extraBody === "object") {
    Object.assign(body, config.extraBody);
  }

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Vision model returned ${response.status}: ${body.slice(0, 500)}`);
  }

  const json = (await response.json()) as {
    choices?: Array<{ message?: { content?: string; reasoning_content?: string } }>;
  };

  const msg = json.choices?.[0]?.message;
  return msg?.content || msg?.reasoning_content || "(no response from vision model)";
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function visionToolExtension(pi: ExtensionAPI) {
  // -----------------------------------------------------------------------
  // Session lifecycle: load & persist config
  // -----------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    // Resolve config from file/env, then check for session-persisted overrides
    config = resolveConfig();

    // Restore any mid-session config changes from session entries
    const entries = ctx.sessionManager.getEntries();
    for (const entry of entries) {
      if (entry.type === "custom" && entry.customType === "vision-config") {
        const data = entry.data as Partial<VisionConfig> | undefined;
        if (data?.provider !== undefined) config.provider = data.provider || undefined;
        if (data?.model !== undefined) config.model = data.model || undefined;
        if (data?.maxDimension !== undefined) config.maxDimension = data.maxDimension;
        if (data?.jpegQuality !== undefined) config.jpegQuality = data.jpegQuality;
        if (data?.defaultReasoningEffort !== undefined) config.defaultReasoningEffort = data.defaultReasoningEffort;
        if (data?.enabled !== undefined) config.enabled = data.enabled;
        if (data?.locateMaxWidth !== undefined) config.locateMaxWidth = data.locateMaxWidth;
        if (data?.thinkingDisabled !== undefined) config.thinkingDisabled = data.thinkingDisabled;
        if (data?.extraBody !== undefined) config.extraBody = data.extraBody;
        if (data?.fineCropWidth !== undefined) config.fineCropWidth = data.fineCropWidth;
        if (data?.fineCropHeight !== undefined) config.fineCropHeight = data.fineCropHeight;
        if (data?.fineZoom !== undefined) config.fineZoom = data.fineZoom;
        if (data?.fallbackModels !== undefined) config.fallbackModels = data.fallbackModels;
      }
    }

    updateStatus(ctx);
  });

  /**
   * Persist the current config into the session file.
   * Called whenever config changes via /vision config.
   */
  function persistConfig() {
    pi.appendEntry("vision-config", { ...config });
  }

  // -----------------------------------------------------------------------
  // /vision command
  // -----------------------------------------------------------------------

  pi.registerCommand("vision", {
    description: "Vision tool settings (config, show, clear, on, off)",
    handler: async (args, ctx) => {
      const trimmed = args?.trim() ?? "";

      // No args: show current config
      if (!trimmed) {
        ctx.ui.notify(configSummary(), "info");
        return;
      }

      // /vision on — enable the tool
      if (trimmed === "on") {
        config.enabled = true;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify("Vision tool enabled. The 👁 indicator is now visible in the footer.", "info");
        return;
      }

      // /vision off — disable the tool
      if (trimmed === "off") {
        config.enabled = false;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify("Vision tool disabled. The 👁 indicator will be hidden and describe_image calls will return an error.", "info");
        return;
      }

      // Parse subcommand
      const parts = trimmed.split(/\s+/);
      const subcommand = parts[0].toLowerCase();
      const rest = parts.slice(1).join(" ");

      // /vision show — show current config
      if (subcommand === "show" || subcommand === "status") {
        ctx.ui.notify(configSummary(), "info");
        return;
      }

      // /vision clear — reset to defaults
      if (subcommand === "clear" || subcommand === "reset") {
        config.provider = undefined;
        config.model = undefined;
        config.maxDimension = parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10);
        config.jpegQuality = parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10);
        config.defaultReasoningEffort = validateReasoningLevel(process.env.PI_VISION_REASONING_EFFORT) ?? "off";
        config.enabled = true;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify("Vision config reset to defaults", "info");
        return;
      }

      // /vision config <setting> [value]
      if (subcommand === "config" || subcommand === "cfg") {
        const settingParts = rest.split(/\s+/);
        const setting = settingParts[0]?.toLowerCase();
        const value = settingParts.slice(1).join(" ");

        if (!setting) {
          ctx.ui.notify(configSummary(), "info");
          return;
        }

        if (setting === "provider") {
          if (!value) {
            ctx.ui.notify(`Current provider: ${config.provider ?? "(not set)"}`, "info");
            return;
          }
          config.provider = value || undefined;
          saveConfigFile();
          persistConfig();
          updateStatus(ctx);
          ctx.ui.notify(`Vision provider set to "${config.provider}" (saved to ${CONFIG_PATH})`, "info");
          return;
        }

        if (setting === "model") {
          if (!value) {
            ctx.ui.notify(`Current model: ${config.model ?? "(not set)"}`, "info");
            return;
          }
          config.model = value || undefined;
          saveConfigFile();
          persistConfig();
          updateStatus(ctx);
          ctx.ui.notify(`Vision model set to "${config.model}" (saved to ${CONFIG_PATH})`, "info");
          return;
        }

        if (setting === "max-dim" || setting === "maxdim") {
          if (!value) {
            ctx.ui.notify(`Current max dimension: ${config.maxDimension}px`, "info");
            return;
          }
          const dim = parseInt(value, 10);
          if (isNaN(dim) || dim < 1) {
            ctx.ui.notify(`Invalid dimension: "${value}". Must be a positive number.`, "error");
            return;
          }
          config.maxDimension = dim;
          saveConfigFile();
          persistConfig();
          ctx.ui.notify(`Max image dimension set to ${config.maxDimension}px`, "info");
          return;
        }

        if (setting === "quality" || setting === "jpeg-quality") {
          if (!value) {
            ctx.ui.notify(`Current JPEG quality: ${config.jpegQuality}`, "info");
            return;
          }
          const q = parseInt(value, 10);
          if (isNaN(q) || q < 1 || q > 100) {
            ctx.ui.notify(`Invalid quality: "${value}". Must be 1-100.`, "error");
            return;
          }
          config.jpegQuality = q;
          saveConfigFile();
          persistConfig();
          ctx.ui.notify(`JPEG quality set to ${config.jpegQuality}`, "info");
          return;
        }

        if (setting === "reasoning-effort" || setting === "reasoning" || setting === "thinking") {
          if (!value) {
            ctx.ui.notify(`Current reasoning effort: ${config.defaultReasoningEffort}`, "info");
            return;
          }
          const level = validateReasoningLevel(value);
          if (!level) {
            ctx.ui.notify(
              `Invalid reasoning level: "${value}". Use: ${REASONING_LEVELS.join(", ")}`,
              "error",
            );
            return;
          }
          config.defaultReasoningEffort = level;
          saveConfigFile();
          persistConfig();
          ctx.ui.notify(`Reasoning effort set to "${config.defaultReasoningEffort}"`, "info");
          return;
        }

        ctx.ui.notify(
          `Unknown config setting: "${setting}". Use: provider, model, max-dim, quality, reasoning-effort`,
          "error",
        );
        return;
      }

      // Shorthand: /vision provider <name> or /vision model <name>
      if (subcommand === "provider") {
        if (!rest) {
          ctx.ui.notify(`Current provider: ${config.provider ?? "(not set)"}`, "info");
          return;
        }
        config.provider = rest || undefined;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify(`Vision provider set to "${config.provider}" (saved to ${CONFIG_PATH})`, "info");
        return;
      }

      if (subcommand === "model") {
        if (!rest) {
          ctx.ui.notify(`Current model: ${config.model ?? "(not set)"}`, "info");
          return;
        }
        config.model = rest || undefined;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify(`Vision model set to "${config.model}" (saved to ${CONFIG_PATH})`, "info");
        return;
      }

      ctx.ui.notify(
        `Unknown subcommand: "${subcommand}". Use: config, show, clear (or provider/model)`,
        "error",
      );
    },
  });

  /**
   * Update the footer status bar to show current vision config.
   */
  function updateStatus(ctx: { ui: { setStatus: (id: string, text: string | undefined) => void } }) {
    if (config.provider && config.model && config.enabled) {
      ctx.ui.setStatus("vision", `👁 ${config.provider}/${config.model}`);
    } else {
      ctx.ui.setStatus("vision", undefined);
    }
  }

  // -----------------------------------------------------------------------
  // describe_image tool
  // -----------------------------------------------------------------------

  pi.registerTool({
    name: "describe_image",
    label: "Describe Image",
    description: [
      "Analyze an image using a vision-capable model.",
      "Use this when you need to understand the content of an image:",
      "screenshots, diagrams, photos, UI mockups, error dialogs, charts, etc.",
      "",
      "The `image_path` can be:",
      "- A file path (e.g., /tmp/screenshot.png)",
      "- A data URL (e.g., data:image/png;base64,...)",
      "- A raw base64-encoded image string",
      "",
      "Set `prompt` to exactly what you need:",
      '- Description: "Describe everything visible in this image"',
      '- Coordinates: "Give pixel coordinates [x,y,w,h] of the red button"',
      '- Text: "Extract all visible text, preserving structure"',
      '- Analysis: "Is there a compiler error shown? What does it say?"',
      "",
      "Set `compress` to control image optimization:",
      "- `true`: Resize large images, strip alpha, convert to JPEG (~4x faster, fewer tokens).",
      "  Use for general descriptions, text extraction, UI analysis.",
      "- `false`: Send raw pixels unchanged.",
      "  Use when you need pixel-perfect analysis: exact coordinates, fine text, color accuracy.",
      "IMPORTANT: Always decide between true/false based on what the user needs.",
      "",
      "Set `locate` to true when you need REAL clickable pixel coordinates:",
      "- The tool draws 4 calibration crosses on the image (invisible to the user), asks the",
      "  model for coordinates, then calibrates them into real pixels (~2px accuracy).",
      "- Vision models are unreliable at absolute coordinates (they see a ~1000x1000 internal",
      "  grid), so `locate` is the fix: cross-calibration → real coordinates.",
      "- Use when you need to click/find UI elements on screenshots, game frames, or any",
      "  image type UI where the accessibility tree is unavailable.",
      "- Elements are reported as `Name: (x1,y1,x2,y2)` in real pixels.",
    ].join("\n"),
    promptSnippet: "Analyze the provided image and respond to the prompt",
    promptGuidelines: [
      "Use describe_image when you need to understand the visual content of any image (screenshot, diagram, photo, etc.). Provide a specific prompt describing exactly what information you need from the image.",
      "For most tasks (descriptions, text extraction, general analysis), use compress: true.",
      "Only set compress: false for pixel-perfect accuracy (exact coordinates or fine-detail inspection).",
      "For UI element location on screenshots (clickable coordinates), set locate: true instead of asking for raw coordinates — it calibrates the model's internal grid into real pixels (~2px accuracy).",
      "For simple/fast queries, set reasoning: 'off' to get quick responses. For complex analysis (architecture diagrams, debugging screenshots, multi-step visual reasoning), set reasoning to 'medium', 'high', or 'xhigh' to get deeper analysis. Omit the parameter to use the configured default.",
    ],
    parameters: Type.Object({
      image_path: Type.String({
        description:
          "Path to image file, data URL (data:image/...;base64,...), or raw base64-encoded image data. HTTP(S) URLs are NOT accepted.",
      }),
      prompt: Type.String({
        description:
          "What to analyze or extract from the image. Be specific: 'Describe all UI elements and their positions', 'Read all text in this screenshot', 'What error is shown?', 'Give coordinates of the submit button', etc.",
      }),
      compress: Type.Boolean({
        description:
          "Whether to compress the image before sending. Use true for most tasks (faster, fewer tokens). Use false when pixel-perfect accuracy is needed (exact coordinates, fine text, color precision).",
      }),
      reasoning: Type.Optional(Type.Union([
        Type.Literal("off"),
        Type.Literal("minimal"),
        Type.Literal("low"),
        Type.Literal("medium"),
        Type.Literal("high"),
        Type.Literal("xhigh"),
      ], {
        description:
          "Reasoning effort for the vision model. Use 'off' for fast/cheap queries (e.g., 'what color is this?'), and higher levels for complex analysis (e.g., architecture diagrams, debugging screenshots). Only applies to models with reasoning: true. Falls back to the configured default if omitted.",
      })),
      locate: Type.Optional(Type.Boolean({
        description:
          "Calibration locate mode: draw 4 calibration crosses, ask the model for element coordinates, then calibrate them into REAL pixel coordinates accurate to ~2px. Use when you need clickable coordinates of UI elements (screenshots, game frames, image UIs). The output is prefixed with '[locate 校准坐标] lines like `Name: (x1,y1,x2,y2)`.',",
      })),
    }),
    renderCall(args, theme, _context) {
      const modelLine = theme.fg("toolTitle", theme.bold(`describe_image via ${config.provider}/${config.model}`));
      const promptLine = theme.fg("dim", `prompt: ${args.prompt}`);
      const flags: string[] = [];
      flags.push(args.compress ? "compress: true" : "compress: false");
      const reasoningLevel = args.reasoning ?? config.defaultReasoningEffort;
      if (reasoningLevel !== "off") {
        flags.push(`reasoning: ${reasoningLevel}`);
      }
      const flagsLine = theme.fg("dim", flags.join(", "));
      return new Text(`${modelLine}\n  ${promptLine}\n  ${flagsLine}`, 0, 0);
    },
    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      // Check if the tool is disabled
      if (!config.enabled) {
        return {
          content: [
            {
              type: "text",
              text: [
                "Vision tool is currently disabled.",
                "",
                "Use /vision on to enable it.",
              ].join("\n"),
            },
          ],
          details: { error: "tool_disabled" },
          isError: true,
        };
      }

      // Validate configuration
      if (!config.provider || !config.model) {
        return {
          content: [
            {
              type: "text",
              text: [
                "Vision tool is not configured.",
                "",
                "Use /vision to set the vision provider and model:",
                "  /vision config provider my-provider",
                "  /vision config model my-vision-model",
                "",
                "Or set environment variables (legacy):",
                "  export PI_VISION_PROVIDER=my-provider",
                "  export PI_VISION_MODEL=my-vision-model",
              ].join("\n"),
            },
          ],
          details: { error: "not_configured" },
          isError: true,
        };
      }

      // 阶梯式模型候选: 主模型 + fallbackModels(按顺序; 跳过重复/无效项)
      const candidates: { provider: string; model: string; visionModel: any; apiKey?: string; err?: string }[] = [];
      const seen = new Set<string>();
      const allCands: { provider?: string; model?: string }[] = [
        { provider: config.provider, model: config.model },
        ...(config.fallbackModels || []),
      ];
      for (const c of allCands) {
        const key = `${c.provider}/${c.model}`;
        if (!c.provider || !c.model || seen.has(key)) continue;
        seen.add(key);
        const vm = ctx.modelRegistry.find(c.provider, c.model);
        if (!vm) {
          candidates.push({ provider: c.provider, model: c.model, visionModel: null as any, err: `model "${key}" not found in model registry` });
          continue;
        }
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(vm);
        if (!auth.ok) {
          candidates.push({ provider: c.provider, model: c.model, visionModel: vm, err: `unable to resolve API key: ${auth.error}` });
          continue;
        }
        candidates.push({ provider: c.provider, model: c.model, visionModel: vm, apiKey: auth.apiKey });
      }
      if (candidates.length === 0) {
        return {
          content: [{ type: "text", text: "Vision tool error: 未配置任何视觉模型。请先设置 provider/model 或配置 fallbackModels。" }],
          details: { error: "not_configured" },
          isError: true,
        };
      }
      // 第一个就绪的候选(主模型优先)
      const activeCandidate = candidates.find((c) => c.visionModel && c.apiKey) || candidates[0];
      const activeVisionModel = activeCandidate.visionModel;

      // Decode the image
      const compress = params.compress;
      const locate = (params as any).locate === true;
      let imageData: { mimeType: string; data: string };
      let calibInfo: { width: number; height: number; marks: CalibMark[]; scaleX: number; scaleY: number } | null = null;
      let rawBuffer: Buffer | null = null;
      let effectivePrompt = params.prompt;
      try {
        if (locate) {
          // locate mode: need the raw image (no resize) to draw calibration marks
          const raw = await imageToBase64(params.image_path, false);
          rawBuffer = Buffer.from(raw.data, "base64");
          const marked = await addCalibrationMarks(rawBuffer, config.locateMaxWidth);
          if (marked) {
            imageData = { mimeType: "image/jpeg", data: marked.buffer.toString("base64") };
            calibInfo = { width: marked.width, height: marked.height, marks: marked.marks, scaleX: marked.scaleX, scaleY: marked.scaleY };
            effectivePrompt = buildCalibrationPrompt(params.prompt, marked.width, marked.height);
          } else {
            // sharp unavailable — fall back to assumed 1000×1000 space; still needs W/H
            imageData = raw;
            const dims = getImageSize(rawBuffer);
            if (dims) {
              calibInfo = { width: dims.width, height: dims.height, marks: calibMarksFor(dims.width, dims.height), scaleX: 1, scaleY: 1 };
              effectivePrompt = buildCalibrationPrompt(params.prompt, dims.width, dims.height);
            }
          }
        } else {
          imageData = await imageToBase64(params.image_path, compress);
        }
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `Vision tool error: could not read image "${params.image_path}": ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: { error: "image_read_error" },
          isError: true,
        };
      }

      // Determine reasoning level: tool parameter > default config
      const rawReasoning: string | undefined = (params as any).reasoning;
      let reasoningLevel = config.defaultReasoningEffort;
      if (rawReasoning !== undefined && rawReasoning !== null) {
        const parsed = validateReasoningLevel(rawReasoning);
        if (parsed) {
          reasoningLevel = parsed;
        }
      }

      const compressLabel = compress
        ? `compressed (${(imageData.data.length / 1024).toFixed(0)}KB base64)`
        : `raw (${(imageData.data.length / 1024).toFixed(0)}KB base64)`;

      const reasoningLabel =
        activeVisionModel?.reasoning && reasoningLevel !== "off"
          ? `, reasoning: ${reasoningLevel}`
          : "";

      // Static initial update in the tool output
      onUpdate?.({
        content: [
          {
            type: "text",
            text: `Analyzing image with ${config.model} (${compressLabel}${reasoningLabel})...`,
          },
        ],
        details: {
          model: `${config.provider}/${config.model}`,
          image_path: params.image_path,
          compressed: compress,
        },
      });

      // Animated spinner in the footer status line
      const spinnerFrames = ["◐", "◓", "◑", "◒"];
      let spinnerIndex = 0;
      let spinnerTimer: ReturnType<typeof setInterval> | null = null;

      if (config.enabled) {
        const updateSpinner = () => {
          spinnerIndex = (spinnerIndex + 1) % spinnerFrames.length;
          ctx.ui.setStatus("vision", `${spinnerFrames[spinnerIndex]} ${config.provider}/${config.model}`);
        };
        updateSpinner();
        spinnerTimer = setInterval(updateSpinner, 200);
      }

      // Call the vision model (阶梯式: 主模型失败自动切换 fallbackModels)
      try {
        let result: string | null = null;
        let usedVisionModel: any = null;
        let usedApiKey = "";
        let lastErr: string | null = null;
        for (const cand of candidates) {
          if (!cand.visionModel || !cand.apiKey) { lastErr = cand.err || "未就绪"; continue; }
          try {
            result = await callVisionModel(
              cand.visionModel,
              cand.apiKey,
              imageData.data,
              imageData.mimeType,
              effectivePrompt,
              signal,
              reasoningLevel,
              locate, // locate 模式强制禁用思考(豆包等思考会慢 10 倍)
            );
            usedVisionModel = cand.visionModel;
            usedApiKey = cand.apiKey;
            if (usedVisionModel.id !== config.model) {
              // 通知模型发生了阶梯切换
              onUpdate?.({
                content: [{ type: "text", text: `主模型 ${config.model} 失败,已自动切换到 ${cand.provider}/${cand.model} ✓` }],
                details: { fallback_used: `${cand.provider}/${cand.model}` },
              });
            }
            break;
          } catch (e) {
            lastErr = `${cand.provider}/${cand.model}: ${e instanceof Error ? e.message : String(e)}`;
          }
        }
        if (result === null) {
          return {
            content: [{ type: "text", text: `Vision tool error: 所有视觉模型均失败。${lastErr}` }],
            details: { error: "vision_call_error" },
            isError: true,
          };
        }

        let finalText = result;
        if (locate && calibInfo) {
          const report = buildLocateReport(result, calibInfo.width, calibInfo.height, calibInfo.marks, calibInfo.scaleX, calibInfo.scaleY);
          // V2: 多阶段精定位——把粗定位目标中心映射到物理坐标，再局部裁剪放大精确定位
          try {
            if (rawBuffer) {
              const rawBuf = rawBuffer; // 捕获: await 后 let 的 narrowing 会失效
              const pts = extractPointLines(result);
              const calibMap = new Map<string, { x: number; y: number }>();
              for (const p of pts) {
                if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") {
                  calibMap.set(p.name, { x: (p.x1 + (p.x2 ?? p.x1)) / 2, y: (p.y1 + (p.y2 ?? p.y1)) / 2 });
                }
              }
              const aff = fitAffine(calibMap, calibInfo.marks);
              const targets = [];
              for (const p of pts) {
                if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") continue;
                if (!aff) continue;
                const r = toReal(aff, (p.x1 + (p.x2 ?? p.x1)) / 2, (p.y1 + (p.y2 ?? p.y1)) / 2);
                // 缩放后的坐标映射回原图物理像素
                targets.push({ name: p.name, cx: Math.round(r.x * calibInfo.scaleX), cy: Math.round(r.y * calibInfo.scaleY) });
              }
              if (targets.length > 0) {
                const fine = await refineLocate(
                  rawBuf,
                  targets,
                  usedVisionModel,
                  usedApiKey ?? "",
                  signal,
                  reasoningLevel,
                  params.prompt ?? "",
                );
                finalText = report + "\n\n" + fine + "\n\n" + result;
              } else {
                finalText = report + "\n\n" + result;
              }
            } else {
              finalText = report + "\n\n" + result;
            }
          } catch (e) {
            finalText = report + "\n\n" + result + "\n[locate-fine 精定位异常] " + (e instanceof Error ? e.message : String(e));
          }
        }

        return {
          content: [{ type: "text", text: finalText }],
          details: {
            model: `${config.provider}/${config.model}`,
            image_path: params.image_path,
            prompt: params.prompt,
            compressed: compress,
            reasoning: reasoningLevel,
            locate: locate || undefined,
          },
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `Vision tool error: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: { error: "vision_call_error" },
          isError: true,
        };
      } finally {
        if (spinnerTimer) clearInterval(spinnerTimer);
        // Restore the static indicator
        if (config.enabled) {
          ctx.ui.setStatus("vision", `👁 ${config.provider}/${config.model}`);
        }
      }
    },
  });
}
