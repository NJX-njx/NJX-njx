# Changelog

## Unreleased

### README / Profile
- 将 `README.md` 的主视觉统一为 CRT 风格横幅，当前顶部只展示 `assets/banners/crt-banner.gif`。
- 放弃早期信息过长、结构松散的 profile 呈现，收敛为更聚焦的 banner-first 方案。
- 将 README 中对旧路径 `assets/crt_banner.gif` 的引用更新为新的规范路径。

### CRT Banner
- 重写并持续打磨 `tools/generate_crt_banner.py`，将其确立为当前主路线横幅生成器。
- 横幅生成路线从“程序化像素人物”逐步切换为“原图直接融入 CRT 屏幕”，以保留眼睛、发丝、表情和手势等关键细节。
- 左侧人物区改为直接读取 `images/portraits/portrait-reference.jpg`，并叠加 CRT 风格的扫描线、泛光、闪烁、暗角和显示器质感。
- 调整左侧图像裁切范围，使原图内容显示更多，不再过度近景裁切。
- 减弱左侧整体绿调，恢复原图的眼睛、发丝和面部层次。
- 保留右侧终端信息面板和整体 CRT 外框风格，使横幅更像“原图进入老显示器”，而不是“人物被重新像素化”。

### Right Panel
- 移除右侧底部的色块装饰条。
- 更新 `Editor` 字段为两行展示：`Cursor / VS Code` 与 `Opencode / Trae`。
- 更新 `Languages` 字段为三行展示：`Python, CUDA,`、`C/C++, JavaScript,`、`TypeScript, Bash`。
- 更新 `Skills` 字段为英文方向描述：`LLM fine-tuning`、`Prompt/context eng`、`Distributed training`、`LLM evaluation`。
- 新增 `Recent` 字段，内容为 `Watching The Young` 与 `Brewmaster's Adventure`。
- 修正 `Recent` 的断行方式，使其在 `Young` 后换行。
- 统一右侧所有值列的起始对齐位置，修复不同字段横向缩进不一致的问题。

### Metrics / Widgets / Workflows
- 保留并恢复 `metrics` 路线能力，但不再作为 README 当前主视觉展示。
- 整理 `.github/workflows/metrics.yml`，保留 `header`、`activity`、`repositories`、`metadata`、`languages`、`lines`、`stargazers`、`habits` 等插件路线。
- 为 `metrics` 路线加入可选的 `WakaTime` 集成配置，支持通过 `WAKATIME_TOKEN` 条件启用。
- 保留 trophies 更新路线，并将资源迁移到 `assets/widgets/trophies.svg`。
- 更新 `.github/workflows/update-trophies.yml`，使其适配新的资源目录结构。

### Asset Structure
- 对 `assets` 和 `images` 做了规范化整理，确立规则：普通原始图片放在 `images`，展示资源、生成结果和中间产物放在 `assets`。
- 新建 `images/portraits/`，用于存放人物原图和参考图。
- 新建 `images/logos/`，用于存放 logo 和 icon 类图片。
- 新建 `assets/banners/`，用于存放 README 横幅和其他 banner 输出。
- 新建 `assets/widgets/`，用于存放 `trophies.svg` 这类非 banner 资源。
- 新建 `assets/intermediate/`，用于存放 `ascii-base.png` 这类中间产物。
- 新建 `assets/animations/`，用于存放 `code.gif` 这类动画资源。
- 将多份人物图统一归档到 `images/portraits/`。
- 将 logo / icon 图片统一归档到 `images/logos/`。
- 将 banner 输出统一归档到 `assets/banners/`。
- 将 trophies 统一归档到 `assets/widgets/`。
- 将历史 ASCII 中间产物统一归档到 `assets/intermediate/`。
- 对资源文件进行了更规范的 kebab-case 重命名，例如 `crt_banner.gif` 调整为 `crt-banner.gif`，`header_profile.svg` 调整为 `header-profile.svg`。

### Tooling Cleanup
- 统一整理 `tools` 目录下多支历史 banner 生成脚本的说明文字与输出约定。
- 为 `generate_banner.py`、`generate_arch_banner.py`、`generate_interactive_banner.py`、`generate_pro_banner.py`、`generate_profile_banner.py`、`generate_final_banner.py`、`generate_crt_banner.py` 补充或统一模块说明。
- 统一这些脚本的输出路径风格，使其全部指向新的 `assets/banners/` 或对应规范目录。
- 统一这些脚本的入口写法，使其从项目根目录解析路径，而不是依赖脆弱的相对字符串。
- 为旧脚本补充输出目录自动创建逻辑，减少未来运行时因目录不存在导致的失败。
- 将 `generate_final_banner.py` 的中间输入路径更新为 `assets/intermediate/ascii-base.png`。
- 将 `generate_profile_banner.py` 与 `generate_final_banner.py` 的人物素材路径更新为 `images/portraits/`。

### Verification
- 多次重新生成当前主横幅 `assets/banners/crt-banner.gif`，确保视觉调整真实落地。
- 在新目录结构下实际运行 `tools/generate_crt_banner.py`，确认新路径可正常读取与输出。
- 使用 `ReadLints` 检查关键文件，未发现新增 lint 问题。
- 使用 `python -m py_compile` 对多支 banner 脚本进行语法检查，结果通过。
- 扫描旧路径引用，未发现仍指向旧目录结构的文本残留。
