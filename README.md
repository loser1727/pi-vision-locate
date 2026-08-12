# 🎯 pi-vision-locate

**Pi Coding Agent 扩展：识图 + 三阶段精确定位（可点击级坐标）**

让 Pi 通过任意视觉模型（OpenAI 兼容 API）识别图片内容，并用 **三阶段渐进放大 + 十字校准** 把目标元素定位到 **±1~5px** 的真实像素坐标。

> 基于 [pi-vision-tool](https://github.com/xezpeleta/pi-vision-tool) 增强：新增 `locate: true` 多阶段精定位（V2），解决视觉模型"绝对坐标不可靠（内部网格失真，误差 100px+）"的通病。

---

## ✨ 它能做什么

| 能力 | 说明 |
|---|---|
| 🖼️ 识图 | 描述图片、提取文字、分析截图、找 Bug、UI 审查…… |
| 🎯 精确定位 | 在截图/图片中找到任意目标元素，返回**真实像素坐标**（可点击级） |
| 📐 三阶段校准 | 粗定位 → 裁剪放大 4× → 再放大 6×，配合四角十字校准 + 仿射变换 |
| 🔌 任意视觉模型 | 支持任何 OpenAI 兼容 /chat/completions 视觉端点（豆包、Gemini、MiniMax、GLM…） |

## 🚀 安装

```bash
# 方式一：直接放在扩展目录
git clone https://github.com/loser1727/pi-vision-locate ~/.pi/agent/extensions/pi-vision-locate

# 方式二：作为包安装（若有 npm 发布版）
pi install npm:pi-vision-locate
```

重启 Pi 后，`describe_image` 工具即注册（同时保留原 pi-vision-tool 的 `/vision` 命令体系）。

## ⚙️ 配置

### 1. 视觉模型（models.json）

在 `~/.pi/agent/models.json` 添加一个支持图片的 provider：

```jsonc
{
  "providers": {
    "my-vision": {
      "baseUrl": "https://你的API网关/v1",   // 任意 OpenAI 兼容端点
      "api": "openai-completions",
      "apiKey": "sk-xxxx",                    // 或放 auth.json
      "compat": { "supportsDeveloperRole": false },
      "models": [
        { "id": "你的视觉模型", "input": ["text", "image"] }  // input 必须含 image
      ]
    }
  }
}
```

### 2. 选择模型

```bash
/vision config provider my-vision
/vision config model 你的视觉模型
```

（或写 `~/.pi/agent/vision-tool.json`：`{"provider": "...", "model": "...", "enabled": true}`）

## 💡 使用

### 普通识图

```
describe_image(image_path="C:/shot.png", prompt="描述这个界面", compress=true)
```

### 🎯 精确定位（核心卖点）

```
describe_image(
  image_path="C:/shot.png",
  prompt="找到'下载'按钮",
  locate=true          // ← 开启三阶段精定位
)
```

输出示例：

```
[locate 校准坐标（真实像素，十字校准，可直接用于点击）]
下载: (1180,340)

[locate-fine 精确定位坐标（多阶段校准，可直接点击）]
下载: (1180,341)     ← 误差 ±1~5px，直接拿去点击
```

**典型用法**：定位 + [CursorClick](https://github.com/loser1727/desktop-automation-mcp) 点击闭环。

## 🔬 工作原理（为什么这么准）

视觉模型的通病：**绝对像素坐标不可靠**。后端把任意图缩放进内部网格（如 1000×1000），模型没有"尺子"，直接报坐标会系统性偏移 100~200px；但**相对位置、局部细节**非常准。

三阶段方案：

```
① 粗定位（全图）
   截图 → 1600px 宽 → 四角画红色十字（坐标已知）
   → 模型报十字+目标位置 → 仿射变换（模型空间→物理空间）
   → 误差 ±50px

② 细定位（裁剪 400×300，放大 4×）
   以粗定位为中心裁剪 → 放大 → 局部十字校准 → 仿射反推
   → 误差 ±10px

③ 超细定位（裁剪 150×110，放大 6×）
   同理再来一轮 → 误差 ±1~5px ✅
```

关键点：
- **裁剪放大**：模型在局部大图上定位准（±1~2px @6×）
- **十字校准**：每次图都带已知坐标的标记点，用仿射变换把模型坐标系数学映射到物理像素
- **渐进缩小**：每轮搜索范围减半，精度指数提升

实测（800×500 合成 UI）：

| 目标 | 定位结果 | 真实中心 | 误差 |
|---|---|---|---|
| Cancel 按钮 | (341,306) | (340,305) | **1px** |
| Settings 按钮 | (661,181) | (660,180) | **1px** |

## 📁 文件结构

```
pi-vision-locate/
├── extensions/
│   ├── vision-tool.ts        ← 增强版扩展（含 locate V2 三阶段精定位）
│   └── vision-tool.ts.orig   ← 原始版（对比参考）
├── package.json              ← pi 包 manifest
└── README.md
```

## ⚠️ 注意事项

- 需要 `sharp`（图片裁剪/放大）：`npm install sharp`（pi 环境通常已带）
- `locate: true` 会发起 **3 次**视觉 API 调用（粗/细/超细），比普通识图慢，但值得
- 坐标基于**截图物理像素**；配合 DPI 100% 或按物理分辨率换算
- 若视觉 API 不稳定，可在模型配置里选更稳的模型（实测 Gemini 3.5 Flash 定位最稳）

## 📄 License

MIT
