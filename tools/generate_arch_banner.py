import math

def generate_architecture_banner(filename):
    width = 1200
    height = 600
    bg_color = "#0d1117"
    
    # Colors matching the "LLM visualization" aesthetic (neon/cyberpunk)
    colors = {
        'input': '#FF6B6B',      # Red/Orange
        'hidden': '#4ECDC4',     # Teal/Cyan
        'attention': '#FFE66D',  # Yellow
        'output': '#1A535C',     # Dark Blue
        'connection': '#58a6ff', # GitHub Blue
        'text': '#c9d1d9'
    }

    svg_content = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color: {bg_color};">',
        '<defs>',
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feGaussianBlur stdDeviation="2" result="blur" />',
        '<feComposite in="SourceGraphic" in2="blur" operator="over" />',
        '</filter>',
        '</defs>'
    ]

    # Simple isometric projection
    # x_screen = x - z
    # y_screen = y + (x + z) * 0.5
    def project(x, y, z):
        scale = 40
        iso_x = (x - z) * math.cos(math.radians(30)) * scale + width / 2
        iso_y = (x + z) * math.sin(math.radians(30)) * scale - y * scale + height / 2 + 100
        return iso_x, iso_y

    def draw_cube(x, y, z, color, opacity=0.8, label=None):
        px, py = project(x, y, z)
        size = 20 # visual size
        
        # Cube faces (simplified for SVG)
        # Top
        p1 = (px, py - size)
        p2 = (px + size * 0.866, py - size * 0.5)
        p3 = (px, py)
        p4 = (px - size * 0.866, py - size * 0.5)
        
        # Side 1 (Right)
        p5 = (px + size * 0.866, py + size * 0.5)
        p6 = (px, py + size)
        
        # Side 2 (Left)
        p7 = (px - size * 0.866, py + size * 0.5)

        # Draw faces
        svg_content.append(f'<path d="M{p1[0]},{p1[1]} L{p2[0]},{p2[1]} L{p3[0]},{p3[1]} L{p4[0]},{p4[1]} Z" fill="{color}" fill-opacity="{opacity}" stroke="{bg_color}" stroke-width="1"/>')
        svg_content.append(f'<path d="M{p3[0]},{p3[1]} L{p2[0]},{p2[1]} L{p5[0]},{p5[1]} L{p6[0]},{p6[1]} Z" fill="{color}" fill-opacity="{opacity*0.8}" stroke="{bg_color}" stroke-width="1"/>')
        svg_content.append(f'<path d="M{p4[0]},{p4[1]} L{p3[0]},{p3[1]} L{p6[0]},{p6[1]} L{p7[0]},{p7[1]} Z" fill="{color}" fill-opacity="{opacity*0.6}" stroke="{bg_color}" stroke-width="1"/>')
        
        if label:
             svg_content.append(f'<text x="{px}" y="{py-size*1.5}" text-anchor="middle" font-family="monospace" font-size="10" fill="{colors["text"]}">{label}</text>')

    # Draw "Layers" of the model
    
    # Layer 1: Input Embeddings (Bottom)
    for i in range(-3, 4):
        draw_cube(i, 0, 0, colors['input'], label="Token" if i==0 else None)

    # Layer 2: Attention Heads (Middle)
    for i in range(-2, 3):
        for j in range(-2, 3):
            draw_cube(i, 3, j, colors['attention'], opacity=0.6)
            
    # Layer 3: Feed Forward (Top)
    for i in range(-3, 4):
        draw_cube(i, 6, 0, colors['hidden'])

    # Draw connecting lines (Attention mechanism visualization)
    # Connect Input center to Attention center
    p_in = project(0, 0, 0)
    p_att = project(0, 3, 0)
    svg_content.append(f'<line x1="{p_in[0]}" y1="{p_in[1]}" x2="{p_att[0]}" y2="{p_att[1]}" stroke="{colors["connection"]}" stroke-width="2" stroke-dasharray="5,5" opacity="0.5" />')

    # Connect Attention to Output
    p_out = project(0, 6, 0)
    svg_content.append(f'<line x1="{p_att[0]}" y1="{p_att[1]}" x2="{p_out[0]}" y2="{p_out[1]}" stroke="{colors["connection"]}" stroke-width="2" stroke-dasharray="5,5" opacity="0.5" />')

    # Add Text Labels
    svg_content.append(f'''
    <text x="100" y="100" font-family="monospace" font-size="24" fill="{colors['text']}" font-weight="bold">MODEL: NJX-TRANSFORMER</text>
    <text x="100" y="130" font-family="monospace" font-size="14" fill="{colors['connection']}">> ARCHITECTURE: DECODER-ONLY</text>
    <text x="100" y="150" font-family="monospace" font-size="14" fill="{colors['connection']}">> PARAMETERS: INFINITE</text>
    
    <!-- Annotations -->
    <text x="{width-200}" y="{height-100}" text-anchor="end" font-family="monospace" font-size="12" fill="{colors['input']}">INPUT: CURIOSITY</text>
    <text x="{width-200}" y="{height-80}" text-anchor="end" font-family="monospace" font-size="12" fill="{colors['hidden']}">OUTPUT: INNOVATION</text>
    ''')

    svg_content.append('</svg>')

    with open(filename, 'w') as f:
        f.write('\n'.join(svg_content))

if __name__ == "__main__":
    generate_architecture_banner("assets/header_arch.svg")
