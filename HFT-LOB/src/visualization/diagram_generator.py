"""
Research Architecture Diagram Generator for LOB-PROFIT Pipeline.
Generates research diagrams:
- Figure 1: 7-Phase End-to-End LOB-PROFIT Pipeline Architecture
- Figure 2: Multi-Asset LOB Data Integration Architecture
- Figure 3: 4-Tier Feature Engineering Taxonomy
- Figure 4: Deep Learning & Transformer Model Architectures
- Figure 15: Operational Prediction-to-Execution Deployment Architecture
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ResearchDiagramGenerator:
    """
    Generates publication research flowcharts and diagrams.
    """
    def __init__(self, output_dir: str = "results/diagrams"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_figure_1_pipeline_diagram(self, save_name: str = "figure_01_lob_profit_pipeline.png"):
        """Figure 1: Complete 7-Phase LOB-PROFIT Framework."""
        fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
        ax.axis("off")
        
        phases = [
            ("Phase 1\nData &\nPreprocessing", "#4a90e2"),
            ("Phase 2\nFeature\nEngineering", "#50e3c2"),
            ("Phase 3\nModel\nTraining", "#f5a623"),
            ("Phase 4\nSignal\nConversion", "#e94e77"),
            ("Phase 5\nExecution\nSimulation\n(RECIM)", "#9013fe"),
            ("Phase 6\nProfit\nMetrics\n(MDMT)", "#7ed321"),
            ("Phase 7\nRegime\nRobustness\n(RAEP)", "#b8e986")
        ]
        
        box_width = 1.3
        box_height = 0.9
        spacing = 0.5
        y = 0.5
        
        for i, (title, color) in enumerate(phases):
            x = 0.5 + i * (box_width + spacing)
            # Draw box
            rect = patches.FancyBboxPatch(
                (x, y - box_height/2), box_width, box_height,
                boxstyle="round,pad=0.1",
                edgecolor="#333333", facecolor=color, alpha=0.85, lw=1.5
            )
            ax.add_patch(rect)
            ax.text(x + box_width/2, y, title, ha="center", va="center", fontsize=9, fontweight="bold", color="#111111")
            
            # Draw arrow
            if i < len(phases) - 1:
                ax.annotate(
                    "", xy=(x + box_width + spacing, y), xytext=(x + box_width, y),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333333")
                )
                
        # Feedback loop arrow from Phase 7 to Phase 3 (online adaptation)
        ax.annotate(
            "Online Adaptation / Dynamic Ensemble Feedback",
            xy=(0.5 + 2 * (box_width + spacing) + box_width/2, y - box_height/2),
            xytext=(0.5 + 6 * (box_width + spacing) + box_width/2, y - box_height/2),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#d9534f", linestyle="--", connectionstyle="arc3,rad=0.3"),
            ha="center", va="top", fontsize=9, color="#d9534f", fontweight="bold"
        )

        ax.set_xlim(0, 0.5 + len(phases) * (box_width + spacing))
        ax.set_ylim(-0.3, 1.2)
        plt.title("Figure 1: The LOB-PROFIT Framework — Seven-Phase End-to-End Evaluation Pipeline", fontsize=12, fontweight="bold", pad=15)
        
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path

    def generate_figure_4_architecture_diagram(self, save_name: str = "figure_04_model_architectures.png"):
        """Figure 4: Deep Learning & Transformer Model Architectures."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        for ax in axes:
            ax.axis("off")

        # DeepLOB diagram
        axes[0].set_title("A. DeepLOB (Spatial CNN + LSTM)", fontsize=11, fontweight="bold")
        components_deeplob = [
            "Input: 2D LOB Snapshot (K x 4)",
            "Spatial Convolutions (1x2, 4x1 Filters)",
            "Spatial Feature Maps & Inception Modules",
            "LSTM Temporal Dynamics (Hidden: 64)",
            "Fully Connected & Softmax Head (Ternary {-1,0,+1})"
        ]
        for idx, text in enumerate(components_deeplob):
            rect = patches.FancyBboxPatch((0.1, 0.8 - idx*0.16), 0.8, 0.12, boxstyle="round,pad=0.05",
                                          facecolor="#d4e6f1", edgecolor="#2980b9", lw=1.2)
            axes[0].add_patch(rect)
            axes[0].text(0.5, 0.86 - idx*0.16, text, ha="center", va="center", fontsize=9)
            if idx < len(components_deeplob) - 1:
                axes[0].annotate("", xy=(0.5, 0.8 - idx*0.16), xytext=(0.5, 0.8 - idx*0.16 - 0.04),
                                 arrowprops=dict(arrowstyle="<-", lw=1.5, color="#2c3e50"))

        # LiT Transformer diagram
        axes[1].set_title("B. LiT: Limit Order Book Transformer", fontsize=11, fontweight="bold")
        components_lit = [
            "Input: Feature Sequence (T x D_features)",
            "Linear Projection (d_model = 64) + Positional Encoding",
            "Multi-Head Self-Attention (4 Heads, 2 Layers)",
            "GELU Feed-Forward + Temporal Mean Pooling",
            "Classification MLP Head (Ternary {-1,0,+1})"
        ]
        for idx, text in enumerate(components_lit):
            rect = patches.FancyBboxPatch((0.1, 0.8 - idx*0.16), 0.8, 0.12, boxstyle="round,pad=0.05",
                                          facecolor="#d5f5e3", edgecolor="#27ae60", lw=1.2)
            axes[1].add_patch(rect)
            axes[1].text(0.5, 0.86 - idx*0.16, text, ha="center", va="center", fontsize=9)
            if idx < len(components_lit) - 1:
                axes[1].annotate("", xy=(0.5, 0.8 - idx*0.16), xytext=(0.5, 0.8 - idx*0.16 - 0.04),
                                 arrowprops=dict(arrowstyle="<-", lw=1.5, color="#2c3e50"))

        plt.suptitle("Figure 4: Deep Learning & Transformer Architectures for LOB Microstructure", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return out_path
