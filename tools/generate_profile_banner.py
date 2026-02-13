"""
Generate a cyberpunk-style animated SVG banner with embedded girl.jpg avatar.
Uses CSS @keyframes animations for dynamic effects that render on GitHub.
"""

import base64
import os

def generate_profile_banner(output_path, image_path):
    """Generate an animated SVG banner with embedded avatar image."""
    
    # Read and encode the image
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    img_uri = f"data:image/jpeg;base64,{img_data}"
    
    width = 1200
    height = 400
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<defs>
  <style>
    /* ===== Global ===== */
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 0.6; }}
      50%      {{ opacity: 1; }}
    }}
    @keyframes blink {{
      0%, 100% {{ opacity: 1; }}
      50%      {{ opacity: 0; }}
    }}
    @keyframes glow-ring {{
      0%   {{ stroke-dashoffset: 565; opacity: 0.4; }}
      50%  {{ stroke-dashoffset: 0;   opacity: 1;   }}
      100% {{ stroke-dashoffset: -565; opacity: 0.4; }}
    }}
    @keyframes scan {{
      0%   {{ transform: translateY(-200px); opacity: 0; }}
      10%  {{ opacity: 0.15; }}
      90%  {{ opacity: 0.15; }}
      100% {{ transform: translateY(400px); opacity: 0; }}
    }}
    @keyframes data-flow {{
      0%   {{ stroke-dashoffset: 20; }}
      100% {{ stroke-dashoffset: 0; }}
    }}
    @keyframes typing {{
      from {{ width: 0; }}
      to   {{ width: 340px; }}
    }}
    @keyframes cursor-blink {{
      0%, 100% {{ opacity: 1; }}
      50%      {{ opacity: 0; }}
    }}
    @keyframes particle-float {{
      0%   {{ transform: translateY(0) translateX(0); opacity: 0; }}
      20%  {{ opacity: 0.8; }}
      80%  {{ opacity: 0.8; }}
      100% {{ transform: translateY(-80px) translateX(20px); opacity: 0; }}
    }}
    @keyframes hex-rotate {{
      0%   {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    @keyframes bar-grow {{
      from {{ width: 0; }}
      to   {{ width: var(--bar-w); }}
    }}
    @keyframes border-trace {{
      0%   {{ stroke-dashoffset: 3200; }}
      100% {{ stroke-dashoffset: 0; }}
    }}
    @keyframes wave {{
      0%, 100% {{ transform: translateY(0); }}
      50%      {{ transform: translateY(-3px); }}
    }}

    .fade-in       {{ animation: fadeIn 1s ease-out both; }}
    .fade-in-d1    {{ animation: fadeIn 1s ease-out 0.3s both; }}
    .fade-in-d2    {{ animation: fadeIn 1s ease-out 0.6s both; }}
    .fade-in-d3    {{ animation: fadeIn 1s ease-out 0.9s both; }}
    .fade-in-d4    {{ animation: fadeIn 1s ease-out 1.2s both; }}
    .fade-in-d5    {{ animation: fadeIn 1s ease-out 1.5s both; }}
    .mono          {{ font-family: 'Courier New', Consolas, monospace; }}
    .sans          {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }}
  </style>

  <!-- Clip for circular avatar -->
  <clipPath id="avatar-clip">
    <circle cx="160" cy="200" r="70"/>
  </clipPath>

  <!-- Glow filter -->
  <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="4" result="blur"/>
    <feComposite in="SourceGraphic" in2="blur" operator="over"/>
  </filter>

  <!-- Background grid pattern -->
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e3a5f" stroke-width="0.3" opacity="0.5"/>
  </pattern>

  <!-- Gradient for accent bar -->
  <linearGradient id="accent-grad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%"   stop-color="#58a6ff"/>
    <stop offset="50%"  stop-color="#bc8cff"/>
    <stop offset="100%" stop-color="#ff6b6b"/>
  </linearGradient>

  <linearGradient id="bar-grad-1" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%"  stop-color="#58a6ff"/>
    <stop offset="100%" stop-color="#4ECDC4"/>
  </linearGradient>
  <linearGradient id="bar-grad-2" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%"  stop-color="#bc8cff"/>
    <stop offset="100%" stop-color="#ff6b6b"/>
  </linearGradient>
  <linearGradient id="bar-grad-3" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%"  stop-color="#FFE66D"/>
    <stop offset="100%" stop-color="#4ECDC4"/>
  </linearGradient>
</defs>

<!-- ============ BACKGROUND ============ -->
<rect width="{width}" height="{height}" fill="#0d1117"/>
<rect width="{width}" height="{height}" fill="url(#grid)"/>

<!-- Scan line effect -->
<rect x="0" y="0" width="{width}" height="3" fill="url(#accent-grad)" opacity="0.15"
      style="animation: scan 6s linear infinite;"/>

<!-- Animated border trace -->
<rect x="2" y="2" width="{width-4}" height="{height-4}" rx="8"
      fill="none" stroke="url(#accent-grad)" stroke-width="1.5"
      stroke-dasharray="3200" stroke-dashoffset="3200"
      style="animation: border-trace 4s ease-out 0.5s forwards;" opacity="0.6"/>

<!-- Top accent line -->
<rect x="0" y="0" width="{width}" height="3" fill="url(#accent-grad)" class="fade-in"/>

<!-- ============ FLOATING PARTICLES ============ -->
<g opacity="0.6">
  <circle cx="100"  cy="350" r="2" fill="#58a6ff" style="animation: particle-float 4s ease-in-out infinite;"/>
  <circle cx="300"  cy="380" r="1.5" fill="#bc8cff" style="animation: particle-float 5s ease-in-out 1s infinite;"/>
  <circle cx="500"  cy="360" r="2" fill="#4ECDC4" style="animation: particle-float 4.5s ease-in-out 0.5s infinite;"/>
  <circle cx="700"  cy="370" r="1.5" fill="#FFE66D" style="animation: particle-float 5.5s ease-in-out 1.5s infinite;"/>
  <circle cx="900"  cy="350" r="2" fill="#ff6b6b" style="animation: particle-float 4s ease-in-out 2s infinite;"/>
  <circle cx="1050" cy="380" r="1.5" fill="#58a6ff" style="animation: particle-float 5s ease-in-out 0.8s infinite;"/>
  <circle cx="200"  cy="50"  r="1" fill="#bc8cff" style="animation: particle-float 6s ease-in-out 2.5s infinite;"/>
  <circle cx="800"  cy="60"  r="1.5" fill="#4ECDC4" style="animation: particle-float 5s ease-in-out 3s infinite;"/>
  <circle cx="1100" cy="100" r="1" fill="#FFE66D" style="animation: particle-float 4s ease-in-out 1.2s infinite;"/>
</g>

<!-- ============ AVATAR SECTION ============ -->
<g class="fade-in">
  <!-- Rotating hex ring behind avatar -->
  <circle cx="160" cy="200" r="82" fill="none" stroke="#58a6ff" stroke-width="1"
          stroke-dasharray="10,8" opacity="0.4"
          style="animation: hex-rotate 20s linear infinite; transform-origin: 160px 200px;"/>
  
  <!-- Glowing ring -->
  <circle cx="160" cy="200" r="76" fill="none" stroke="#bc8cff" stroke-width="2"
          stroke-dasharray="565" stroke-dashoffset="565"
          style="animation: glow-ring 4s ease-in-out infinite;" filter="url(#neon-glow)"/>

  <!-- Avatar image -->
  <image x="90" y="130" width="140" height="140"
         href="{img_uri}"
         clip-path="url(#avatar-clip)"
         preserveAspectRatio="xMidYMid slice"/>

  <!-- Inner border ring -->
  <circle cx="160" cy="200" r="70" fill="none" stroke="#c9d1d9" stroke-width="2.5"/>
</g>

<!-- Status indicator -->
<g class="fade-in-d1">
  <circle cx="210" cy="250" r="8" fill="#0d1117" stroke="#c9d1d9" stroke-width="2"/>
  <circle cx="210" cy="250" r="5" fill="#3fb950" style="animation: pulse 2s ease-in-out infinite;"/>
</g>

<!-- ============ TEXT & IDENTITY ============ -->
<g class="fade-in-d1">
  <!-- Name -->
  <text x="280" y="140" class="sans" font-size="32" font-weight="bold" fill="#e6edf3">NJX</text>
  <text x="348" y="140" class="mono" font-size="18" fill="#8b949e" dy="0">  // AI Engineer</text>

  <!-- Typing animation subtitle -->
  <g>
    <rect x="280" y="155" width="0" height="22" fill="transparent"
          style="animation: typing 2.5s steps(30) 1s forwards;"/>
    <text x="280" y="172" class="mono" font-size="13" fill="#58a6ff">
      <tspan>$ echo &quot;Bridging SOTA Research ↔ Production&quot;</tspan>
    </text>
  </g>

  <!-- Cursor -->
  <rect x="622" y="158" width="8" height="15" fill="#58a6ff"
        style="animation: cursor-blink 1s step-end infinite;"/>
</g>

<!-- ============ INFO TAGS ============ -->
<g class="fade-in-d2 mono" font-size="11">
  <g transform="translate(280, 200)">
    <rect width="100" height="22" rx="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>
    <text x="12" y="15" fill="#58a6ff">🎓 BIT · CS</text>
  </g>
  <g transform="translate(390, 200)">
    <rect width="130" height="22" rx="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>
    <text x="12" y="15" fill="#bc8cff">🧠 AI Inference</text>
  </g>
  <g transform="translate(530, 200)">
    <rect width="160" height="22" rx="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>
    <text x="12" y="15" fill="#4ECDC4">🏗️ Model Architecture</text>
  </g>
  <g transform="translate(700, 200)">
    <rect width="155" height="22" rx="4" fill="#161b22" stroke="#30363d" stroke-width="1"/>
    <text x="12" y="15" fill="#FFE66D">🤖 Embodied Intel</text>
  </g>
</g>

<!-- ============ TECH SKILL BARS ============ -->
<g class="fade-in-d3" transform="translate(280, 245)">
  <text x="0" y="12" class="mono" font-size="10" fill="#8b949e">CORE STACK</text>

  <!-- Bar: PyTorch -->
  <text x="0" y="32" class="mono" font-size="10" fill="#c9d1d9">PyTorch</text>
  <rect x="70" y="23" width="200" height="10" rx="3" fill="#161b22"/>
  <rect x="70" y="23" width="0" height="10" rx="3" fill="url(#bar-grad-1)"
        style="--bar-w: 180px; animation: bar-grow 1.5s ease-out 1.2s forwards;"/>

  <!-- Bar: Transformers -->
  <text x="0" y="50" class="mono" font-size="10" fill="#c9d1d9">HF/vLLM</text>
  <rect x="70" y="41" width="200" height="10" rx="3" fill="#161b22"/>
  <rect x="70" y="41" width="0" height="10" rx="3" fill="url(#bar-grad-2)"
        style="--bar-w: 160px; animation: bar-grow 1.5s ease-out 1.4s forwards;"/>

  <!-- Bar: Vision -->
  <text x="0" y="68" class="mono" font-size="10" fill="#c9d1d9">YOLO/ViT</text>
  <rect x="70" y="59" width="200" height="10" rx="3" fill="#161b22"/>
  <rect x="70" y="59" width="0" height="10" rx="3" fill="url(#bar-grad-3)"
        style="--bar-w: 150px; animation: bar-grow 1.5s ease-out 1.6s forwards;"/>
</g>

<!-- ============ DATA FLOW VISUALIZATION ============ -->
<g class="fade-in-d4" transform="translate(600, 245)">
  <text x="0" y="12" class="mono" font-size="10" fill="#8b949e">DATA PIPELINE</text>

  <!-- Pipeline boxes -->
  <g>
    <rect x="0" y="22" width="70" height="28" rx="4" fill="#161b22" stroke="#58a6ff" stroke-width="1"/>
    <text x="35" y="40" text-anchor="middle" class="mono" font-size="9" fill="#58a6ff">INPUT</text>
  </g>

  <!-- Arrow 1 -->
  <line x1="75" y1="36" x2="110" y2="36" stroke="#58a6ff" stroke-width="1.5"
        stroke-dasharray="4,3" style="animation: data-flow 1s linear infinite;"/>
  <polygon points="108,32 115,36 108,40" fill="#58a6ff"/>

  <g>
    <rect x="118" y="22" width="80" height="28" rx="4" fill="#161b22" stroke="#bc8cff" stroke-width="1"/>
    <text x="158" y="40" text-anchor="middle" class="mono" font-size="9" fill="#bc8cff">ENCODER</text>
  </g>

  <!-- Arrow 2 -->
  <line x1="203" y1="36" x2="238" y2="36" stroke="#bc8cff" stroke-width="1.5"
        stroke-dasharray="4,3" style="animation: data-flow 1s linear 0.3s infinite;"/>
  <polygon points="236,32 243,36 236,40" fill="#bc8cff"/>

  <g>
    <rect x="246" y="22" width="85" height="28" rx="4" fill="#161b22" stroke="#FFE66D" stroke-width="1"/>
    <text x="288" y="40" text-anchor="middle" class="mono" font-size="9" fill="#FFE66D">ATTENTION</text>
  </g>

  <!-- Arrow 3 -->
  <line x1="336" y1="36" x2="371" y2="36" stroke="#FFE66D" stroke-width="1.5"
        stroke-dasharray="4,3" style="animation: data-flow 1s linear 0.6s infinite;"/>
  <polygon points="369,32 376,36 369,40" fill="#FFE66D"/>

  <g>
    <rect x="379" y="22" width="75" height="28" rx="4" fill="#161b22" stroke="#3fb950" stroke-width="1"/>
    <text x="416" y="40" text-anchor="middle" class="mono" font-size="9" fill="#3fb950">OUTPUT</text>
  </g>

  <!-- Sub labels -->
  <text x="0" y="70" class="mono" font-size="9" fill="#484f58">
    <tspan style="animation: wave 2s ease-in-out infinite;">▸</tspan> Data → Feature Extraction → Multi-Head Attention → Intelligence
  </text>
</g>

<!-- ============ RIGHT SIDE METRICS PANEL ============ -->
<g class="fade-in-d4" transform="translate(1020, 80)">
  <rect x="0" y="0" width="150" height="130" rx="6" fill="#161b22" stroke="#30363d" stroke-width="1" opacity="0.9"/>
  <text x="12" y="22" class="mono" font-size="10" fill="#58a6ff">⬡ SYS STATUS</text>
  <line x1="10" y1="30" x2="140" y2="30" stroke="#30363d" stroke-width="0.5"/>

  <text x="12" y="48" class="mono" font-size="9" fill="#8b949e">UPTIME</text>
  <text x="135" y="48" text-anchor="end" class="mono" font-size="9" fill="#3fb950">● ONLINE</text>

  <text x="12" y="66" class="mono" font-size="9" fill="#8b949e">FOCUS</text>
  <text x="135" y="66" text-anchor="end" class="mono" font-size="9" fill="#FFE66D">LLM+VIS</text>

  <text x="12" y="84" class="mono" font-size="9" fill="#8b949e">LANG</text>
  <text x="135" y="84" text-anchor="end" class="mono" font-size="9" fill="#bc8cff">Python</text>

  <text x="12" y="102" class="mono" font-size="9" fill="#8b949e">DEPLOY</text>
  <text x="135" y="102" text-anchor="end" class="mono" font-size="9" fill="#4ECDC4">Docker+K8s</text>

  <text x="12" y="120" class="mono" font-size="9" fill="#8b949e">BUILD</text>
  <rect x="60" y="112" width="72" height="6" rx="2" fill="#21262d"/>
  <rect x="60" y="112" width="0" height="6" rx="2" fill="#3fb950"
        style="--bar-w: 65px; animation: bar-grow 2s ease-out 2s forwards;"/>
  <text x="135" y="120" text-anchor="end" class="mono" font-size="8" fill="#3fb950" style="animation: pulse 2s ease-in-out infinite;">OK</text>
</g>

<!-- ============ BOTTOM INFO BAR ============ -->
<g class="fade-in-d5">
  <rect x="0" y="{height-30}" width="{width}" height="30" fill="#161b22" opacity="0.8"/>
  <line x1="0" y1="{height-30}" x2="{width}" y2="{height-30}" stroke="#30363d" stroke-width="0.5"/>

  <text x="20" y="{height-10}" class="mono" font-size="10" fill="#484f58">
    <tspan fill="#3fb950">●</tspan> github.com/NJX-njx
    <tspan dx="30" fill="#58a6ff">📧</tspan> 3771829673@qq.com
    <tspan dx="30" fill="#bc8cff">🌐</tspan> njx-njx.github.io
    <tspan dx="30" fill="#484f58">│</tspan>
    <tspan dx="5" fill="#484f58">Last sync: auto</tspan>
  </text>

  <!-- Blinking signal dot -->
  <circle cx="{width-20}" cy="{height-15}" r="3" fill="#3fb950"
          style="animation: pulse 2s ease-in-out infinite;"/>
</g>

</svg>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    
    print(f"✅ Banner generated: {output_path}")
    print(f"   Image embedded: {image_path} ({os.path.getsize(image_path) // 1024}KB)")
    print(f"   SVG size: {os.path.getsize(output_path) // 1024}KB")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    generate_profile_banner(
        output_path=os.path.join(project_root, "assets", "header_profile.svg"),
        image_path=os.path.join(project_root, "images", "girl.jpg")
    )
