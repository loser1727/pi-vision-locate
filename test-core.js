// pi-vision-locate 核心纯函数单测 (从 vision-tool.ts 提取逻辑)
// 运行: node test-core.js
"use strict";

// ---- 从 vision-tool.ts 复制的纯函数 ----
function extractPointLines(text) {
  const re = /([A-Za-z0-9_\u4e00-\u9fa5]{1,20})\s*[:：]\s*(\d{1,5})\s*[,\s，]+\s*(\d{1,5})(?:\s*[,\s，]+\s*(\d{1,5})\s*[,\s，]+\s*(\d{1,5}))?/g;
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push({ name: m[1], x1: Number(m[2]), y1: Number(m[3]), x2: m[4] !== undefined ? Number(m[4]) : undefined, y2: m[5] !== undefined ? Number(m[5]) : undefined });
  }
  // 兼容自由坐标格式(如智谱 GLM-4V-Flash 的 "x=420, y=150" / "x:420 y:150")
  // 不管十字是否已解析,都尝试解析目标(十字标准格式+目标自由格式的混合情况)
  if (!out.some((p) => p.name === "目标" || p.name === "Target")) {
    const re2 = /x[ \t]*[:=＝][ \t]*(\d{1,5})[^\d]{1,12}?y[ \t]*[:=＝][ \t]*(\d{1,5})/gi;
    let m2;
    while ((m2 = re2.exec(text)) !== null) {
      out.push({ name: "目标", x1: Number(m2[1]), y1: Number(m2[2]), x2: undefined, y2: undefined });
    }
  }
  return out;
}
function fitAffine(calib, marks) {
  const known = marks.filter((mk) => calib.has(mk.name));
  if (known.length < 2) return null;
  const aVals = [], cVals = [], bVals = [], dVals = [];
  for (let i = 0; i < known.length; i++) for (let j = i + 1; j < known.length; j++) {
    const m1 = calib.get(known[i].name), m2 = calib.get(known[j].name);
    const r1 = known[i], r2 = known[j];
    const dxm = m2.x - m1.x, dym = m2.y - m1.y;
    if (Math.abs(dxm) > 2) { const a = (r2.x - r1.x) / dxm; aVals.push(a); cVals.push(r1.x - a * m1.x); }
    if (Math.abs(dym) > 2) { const b = (r2.y - r1.y) / dym; bVals.push(b); dVals.push(r1.y - b * m1.y); }
  }
  if (!aVals.length || !bVals.length) return null;
  const median = (arr) => [...arr].sort((p, q) => p - q)[Math.floor(arr.length / 2)];
  return { a: median(aVals), b: median(bVals), c: median(cVals), d: median(dVals) };
}
function toReal(affine, x, y) { return { x: affine.a * x + affine.c, y: affine.b * y + affine.d }; }
function calibMarksFor(width, height, inset = 40) {
  return [
    { name: "TL", x: inset, y: inset },
    { name: "TR", x: width - inset, y: inset },
    { name: "BL", x: inset, y: height - inset },
    { name: "BR", x: width - inset, y: height - inset },
  ];
}
function buildLocateReport(modelText, width, height, marks, scaleX = 1, scaleY = 1) {
  const points = extractPointLines(modelText);
  const calib = new Map();
  for (const p of points) {
    if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") {
      calib.set(p.name, { x: (p.x1 + (p.x2 ?? p.x1)) / 2, y: (p.y1 + (p.y2 ?? p.y1)) / 2 });
    }
  }
  let affine = fitAffine(calib, marks);
  let method = "cross-calibration";
  if (!affine) { affine = { a: width / 1000, b: height / 1000, c: 0, d: 0 }; method = "assumed-1000x1000"; }
  const rows = [];
  for (const p of points) {
    if (p.name === "TL" || p.name === "TR" || p.name === "BL" || p.name === "BR") continue;
    const r1 = toReal(affine, p.x1, p.y1);
    const rx1 = { x: Math.round(r1.x * scaleX), y: Math.round(r1.y * scaleY) };
    const rx2 = p.x2 !== undefined ? (() => { const r2 = toReal(affine, p.x2, p.y2); return { x: Math.round(r2.x * scaleX), y: Math.round(r2.y * scaleY) }; })() : undefined;
    rows.push(rx2 ? `${p.name}: (${rx1.x},${rx1.y},${rx2.x},${rx2.y})` : `${p.name}: (${rx1.x},${rx1.y})`);
  }
  if (!rows.length) return "[locate] 未在模型输出中找到可解析的元素坐标行";
  return `[locate 校准坐标（真实像素，${method}）]\n${rows.join("\n")}`;
}

// ---- 测试 ----
let pass = 0, fail = 0;
function check(name, cond, detail = "") {
  if (cond) { pass++; console.log("  ✅", name); }
  else { fail++; console.log("  ❌", name, detail); }
}

// T1: 解析 "目标: x,y" 和多格式
{
  const pts = extractPointLines("TL: 40,40 TR: 1520,40\nBL: 40,860 BR: 1520,860 目标: 700,400");
  check("T1 解析4十字+目标", pts.length === 5 && pts[4].name === "目标" && pts[4].x1 === 700 && pts[4].y1 === 400, JSON.stringify(pts));
}
// T2: 仿射校准 (模型空间 1000x1000 假设 vs 真实 1024x460 图)
{
  const marks = calibMarksFor(1024, 460);
  const text = `TL: 40,40 TR: 984,40 BL: 40,420 BR: 984,420 目标: 500,230`;
  const report = buildLocateReport(text, 1024, 460, marks);
  check("T2 十字校准输出目标坐标", report.includes("目标: (500,230)"), report);
}
// T3: scaleX/scaleY 换算 (2400x1080 原图 → 1024 宽 = 461 高, scaleX=2.34375, scaleY≈2.343)
{
  const origW = 2400, origH = 1080, maxW = 1024;
  const width = maxW, height = Math.round(origH * (maxW / origW));
  const scaleX = origW / width, scaleY = origH / height;
  const marks = calibMarksFor(width, height);
  const text = `TL: 40,40 TR: 984,40 BL: 40,420 BR: 984,420 目标: 500,230`;
  const report = buildLocateReport(text, width, height, marks, scaleX, scaleY);
  const m = report.match(/目标: \((\d+),(\d+)\)/);
  const rx = +m[1], ry = +m[2];
  check("T3 scale 换算回原图像素", Math.abs(rx - 1172) <= 2 && Math.abs(ry - 539) <= 2, `${rx},${ry} 期望≈1172,539`);
}
// T4: fallback assumed-1000x1000 (无十字输出时)
{
  const text = `目标: 500,230`;
  const report = buildLocateReport(text, 1024, 460, calibMarksFor(1024, 460));
  check("T4 1000x1000 fallback", report.includes("assumed-1000x1000") && report.includes("目标: (512,106)"), report);
}
// T5: 中文目标名 + 边界值
{
  const text = `下载按钮: 120,60`;
  const report = buildLocateReport(text, 1024, 460, calibMarksFor(1024, 460), 2.34375, 2.343);
  check("T5 中文名+scale", report.includes("下载按钮: (288,65)"), report);
}
// T6: 精定位坐标映射 (refineLocate 的局部→物理换算逻辑)
{
  const cw = 320, ch = 240, zoom = 3, inset = 30;
  const zw = cw * zoom, zh = ch * zoom;
  const marks = [
    { name: "TL", x: inset, y: inset }, { name: "TR", x: zw - inset, y: inset },
    { name: "BL", x: inset, y: zh - inset }, { name: "BR", x: zw - inset, y: zh - inset },
  ];
  const text = `TL: 30,30 TR: 930,30 BL: 30,690 BR: 930,690 目标: 480,360`;
  const pts = extractPointLines(text);
  const calib = new Map();
  for (const p of pts) if (["TL","TR","BL","BR"].includes(p.name)) calib.set(p.name, { x: (p.x1+(p.x2??p.x1))/2, y: (p.y1+(p.y2??p.y1))/2 });
  const aff = fitAffine(calib, marks);
  const t = pts.find(p => p.name === "目标");
  const r = toReal(aff, t.x1, t.y1);
  const x1 = 100, y1 = 200;
  const physX = Math.round(x1 + r.x / zoom), physY = Math.round(y1 + r.y / zoom);
  check("T6 精定位仿射+映射", aff !== null && physX === 260 && physY === 320, `${physX},${physY} 期望 260,320`);
}
// T7: 智谱自由坐标格式兼容(十字标准格式 + 目标自由格式混合)
{
  const pts = extractPointLines("TL: 40,40 TR: 984,40 BL: 40,421 BR: 984,421 底部白色输入框的中心位置为：x=580, y=240");
  check("T7 智谱自由格式解析(混合格式)", pts.length === 5 && pts[4].name === "目标" && pts[4].x1 === 580 && pts[4].y1 === 240, JSON.stringify(pts));
}
// T8: 纯自由格式(无十字)
{
  const pts = extractPointLines("The center of the red rectangle is at coordinates x = 420, y = 150");
  check("T8 纯自由格式", pts.length === 1 && pts[0].name === "目标" && pts[0].x1 === 420 && pts[0].y1 === 150, JSON.stringify(pts));
}

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail ? 1 : 0);
