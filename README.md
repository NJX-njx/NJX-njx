<div align="center">
  <img src="assets/header_interactive.svg" width="100%" alt="NJX Interactive Architecture">
</div>

<div align="center">
  <h3><code>NJX-70B-Instruct</code></h3>
  <p><em>Interactive Model Visualization. Click the nodes above to navigate the system architecture.</em></p>
</div>

---

<!-- Terminal / IDE Layout -->
<table>
<tr>
<td valign="top" width="25%">

### 📂 Explorer

```bash
.
├── 📁 core_identity
│   ├── bio.txt
│   └── config.json
├── 📁 research_frontiers
│   ├── vision.py
│   ├── llm.py
│   └── agents.py
├── 📁 deployments
│   ├── yolo_pt
│   └── gemini-cli
└── 📁 telemetry
    └── stats.log
```

<br/>

### 🧠 Model Arch

```yaml
# NJX-Internal-v1
architecture:
  input:
    - source: "Curiosity"
    - dtype: "Raw_Data"
  encoder:
    - layer: "CS_Fundamentals"
    - activation: "Deep_Learning"
  attention:
    - heads: ["Vision", "LLM"]
    - mechanism: "System_Design"
  decoder:
    - task: "Engineering"
    - output: "Innovation"
```

</td>
<td valign="top" width="75%">

<h3 id="user-content-core-identity"><code>cat core_identity/bio.txt</code></h3>

> **"Bridging the gap between SOTA Research and Production Engineering."**

I treat AI research not just as academic exploration, but as **system architecture**. My goal is to understand the emergent properties of large models and engineer the infrastructure that makes them accessible.

*   **Affiliation**: Beijing Institute of Technology (BIT) · CS
*   **Focus**: AI Inference, Model Architecture, Embodied Intelligence

---

<h3 id="user-content-research-frontiers"><code>python3 research_frontiers/main.py</code></h3>

```python
class ResearchInterests(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = "Vision Transformers (ViT), Object Detection (YOLO)"
        self.llm = "Architecture Design, PEFT, KV Cache Optimization"
        self.agents = "Multi-Agent Orchestration, Tool Use & Planning"
        self.infra = "High-performance Inference, Quantization"

    def forward(self, x):
        return self.agents(self.llm(self.vision(x)))
```

---

<h3 id="user-content-selected-deployments"><code>ls -la deployments/</code></h3>

| Permission | Owner | Name | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| `drwxr-xr-x` | `njx` | **[YOLOv8-pt](https://github.com/NJX-njx/YOLOv8-pt)** | Optimized vision pipeline (+18% speed) | ![Active](https://img.shields.io/badge/Active-success?style=flat-square) |
| `drwxr-xr-x` | `njx` | **[gemini-cli](https://github.com/NJX-njx/gemini-cli)** | Terminal-native multimodal assistant | ![Stable](https://img.shields.io/badge/Stable-blue?style=flat-square) |
| `drwxr-xr-x` | `njx` | **[explainai](https://github.com/NJX-njx/explainai)** | Visualizing attention maps & Grad-CAM | ![Beta](https://img.shields.io/badge/Beta-orange?style=flat-square) |
| `drwxr-xr-x` | `njx` | **[game-demo](https://github.com/NJX-njx/game-demo)** | WebGL/WebGPU physics playground | ![Archived](https://img.shields.io/badge/Archived-inactive?style=flat-square) |

---

<h3 id="contact-api"><code>curl -X POST https://api.njx.dev/contact</code></h3>

```bash
# Initialize connection handshake
curl -X POST https://api.njx.dev/v1/contact \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "visitor",
    "intent": "collaboration",
    "message": "Let's build the future."
  }'
```

```json
# Server Response
{
  "status": "200 OK",
  "message": "Connection established. Ready to collaborate.",
  "latency": "12ms"
}
```

</td>
</tr>
</table>

---

<div align="center">
  <a href="mailto:your-email@example.com"><code>[ POST REQUEST ]</code></a> · 
  <a href="https://github.com/NJX-njx"><code>[ GET REPO ]</code></a>
</div>
