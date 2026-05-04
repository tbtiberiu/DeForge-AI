import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture(output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    branch_rgb_color = '#e1f5fe'
    branch_forensic_color = '#fff3e0'
    fusion_color = '#f3e5f5'
    head_color = '#e8f5e9'

    def draw_box(x, y, text, width=15, height=6, color='white'):
        patch = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                      linewidth=1.5, edgecolor='black', facecolor=color)
        ax.add_patch(patch)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)
        return x, y, width, height

    def draw_arrow(start_pos, end_pos):
        ax.annotate('', xy=end_pos, xytext=start_pos,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black', shrinkA=0, shrinkB=0))

    # Input
    draw_box(42.5, 90, "Imagine Intrare\n(256x256)", width=15, height=6)
    
    # Arrows from input to branches
    draw_arrow((50, 90), (30, 80))  # To RGB
    draw_arrow((50, 90), (70, 80))  # To Forensic

    # --- RGB Branch ---
    ax.text(15, 82, "Ramura RGB (Semantică)", fontsize=11, fontweight='bold', color='#01579b')
    draw_box(15, 74, "DINOv2/v3 (ViT-L/16)\n(Pre-antrenat)", width=30, height=6, color=branch_rgb_color)
    draw_box(15, 64, "Adaptare LoRA\n(r=16, alpha=32)", width=30, height=6, color=branch_rgb_color)
    draw_box(15, 54, "Attention Pooling\n(Query învățat)", width=30, height=6, color=branch_rgb_color)
    
    draw_arrow((30, 74), (30, 70))
    draw_arrow((30, 64), (30, 60))

    # --- Forensic Branch ---
    ax.text(55, 82, "Ramura Forensic (ForensicBranch)", fontsize=11, fontweight='bold', color='#e65100')
    draw_box(55, 74, "Calcul Reziduu\n(I - AvgPool(I))", width=30, height=6, color=branch_forensic_color)
    draw_box(55, 64, "Filtre SRM\n(5 nuclee 5x5)", width=30, height=6, color=branch_forensic_color)
    draw_box(55, 54, "Encoder Conv\n(BatchNorm + GELU)", width=30, height=6, color=branch_forensic_color)
    draw_box(55, 44, "Global Avg Pooling", width=30, height=6, color=branch_forensic_color)

    draw_arrow((70, 74), (70, 70))
    draw_arrow((70, 64), (70, 60))
    draw_arrow((70, 54), (70, 50))

    # --- Fusion ---
    draw_box(35, 30, "Fuziune Trăsături\n(Concatenare)", width=30, height=6, color=fusion_color)
    ax.text(68, 35, "Forensic Gate (λ)", fontsize=9, style='italic')
    
    # Arrows to fusion
    draw_arrow((30, 54), (45, 36))
    draw_arrow((70, 44), (55, 36))

    # --- Head ---
    draw_box(35, 18, "MLP Head\n(512 -> 128 -> 1)", width=30, height=8, color=head_color)
    draw_arrow((50, 30), (50, 26))

    # --- Output ---
    draw_box(42.5, 5, "Scor Detecție\n(Sigmoid)", width=15, height=6)
    draw_arrow((50, 18), (50, 11))

    plt.title("Arhitectura DeForge-AI: Fluxul de Procesare", fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    import os
    os.makedirs('images', exist_ok=True)
    draw_architecture('images/arhitectura_deforge.png')
