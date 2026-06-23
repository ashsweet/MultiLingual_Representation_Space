"""
Phase 4 Visualization Module

Generates publication-quality visualizations for orthogonal steering results:
1. Professional evaluation tables (LaTeX and Markdown)
2. PCA/t-SNE plots showing steering trajectories in representation space
3. Vector similarity heatmaps
4. Language preservation bar charts

Key visualization: Shows how Solution A pushes Bengali tokens into English cluster
while Solution B keeps them within Bengali manifold.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from loguru import logger


class Phase4Visualizer:
    """Generates visualizations for Phase 4 orthogonal steering results."""
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Professional color scheme
        self.colors = {
            "en_safe": "#2ecc71",      # Green - English safe
            "en_unsafe": "#e74c3c",    # Red - English unsafe
            "bn_safe": "#3498db",      # Blue - Bengali safe
            "bn_unsafe": "#9b59b6",    # Purple - Bengali unsafe
            "bn_steered_A": "#e67e22",  # Orange - Bengali steered with Solution A
            "bn_steered_B": "#1abc9c",  # Teal - Bengali steered with Solution B
            "baseline": "#7f8c8d",      # Gray - Baseline
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['figure.dpi'] = 150
    
    def generate_evaluation_table(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, Any],
        output_format: str = "both"
    ) -> Tuple[str, str]:
        """
        Generate professional evaluation table.
        
        Args:
            results: Results dict from test_steering()
            metrics: Metrics dict from compute_vectors()
            output_format: "markdown", "latex", or "both"
        
        Returns:
            Tuple of (markdown_table, latex_table)
        """
        summary = results["summary"]
        n = results["num_prompts"]
        
        # Build table data
        rows = [
            ("Baseline", summary["baseline_refusal_rate"], 
             summary["baseline_english_rate"], "-", "-"),
            ("Raw (A)", summary["raw_refusal_rate"],
             summary["raw_english_rate"], summary["raw_safety_improvement"],
             summary["raw_language_bleed_count"]),
            ("Scaled (A)", summary["scaled_refusal_rate"],
             summary["scaled_english_rate"], summary["scaled_safety_improvement"],
             summary["scaled_language_preserved_count"]),
            ("Mono (B)", summary["mono_refusal_rate"],
             summary["mono_english_rate"], summary["mono_safety_improvement"],
             summary["mono_language_bleed_count"]),
            ("Mono Scaled (B)", summary["mono_scaled_refusal_rate"],
             summary["mono_scaled_english_rate"], summary["mono_scaled_safety_improvement"],
             summary["mono_scaled_language_preserved_count"]),
        ]
        
        # Markdown table
        md_lines = [
            "| Method | Refusal Rate | English Rate | Safety Gain | Lang. Pres. |",
            "|--------|--------------|--------------|-------------|-------------|",
        ]
        for name, refusal, english, safety, lang in rows:
            if safety == "-":
                safety_str = "-"
            else:
                safety_str = f"+{safety*100:.1f}%"
            if lang == "-":
                lang_str = "-"
            elif "Scaled" in name or "Mono Scaled" in name:
                lang_str = f"{lang}/{n}"
            else:
                lang_str = f"{lang}/{n} bleed"
            md_lines.append(
                f"| {name} | {refusal*100:.1f}% | {english*100:.1f}% | "
                f"{safety_str} | {lang_str} |"
            )
        markdown_table = "\n".join(md_lines)
        
        # LaTeX table
        latex_lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Phase 4: Orthogonal Steering Evaluation Results}",
            r"\label{tab:steering_results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Refusal Rate & English Rate & Safety Gain & Lang. Preservation \\",
            r"\midrule",
        ]
        for name, refusal, english, safety, lang in rows:
            if safety == "-":
                safety_str = "-"
            else:
                safety_str = f"+{safety*100:.1f}\\%"
            if lang == "-":
                lang_str = "-"
            elif "Scaled" in name:
                lang_str = f"{lang}/{n}"
            else:
                lang_str = f"{lang}/{n}"
            latex_lines.append(
                f"{name} & {refusal*100:.1f}\\% & {english*100:.1f}\\% & "
                f"{safety_str} & {lang_str} \\\\"
            )
        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        latex_table = "\n".join(latex_lines)
        
        # Save tables
        if output_format in ("markdown", "both"):
            md_path = self.output_dir / "evaluation_table.md"
            with open(md_path, "w") as f:
                f.write(markdown_table)
            logger.info(f"Saved markdown table to {md_path}")
        
        if output_format in ("latex", "both"):
            tex_path = self.output_dir / "evaluation_table.tex"
            with open(tex_path, "w") as f:
                f.write(latex_table)
            logger.info(f"Saved LaTeX table to {tex_path}")
        
        return markdown_table, latex_table

    def generate_vector_metrics_table(
        self,
        metrics: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate table showing vector magnitudes and similarities."""
        
        md_lines = [
            "## Vector Analysis",
            "",
            "### Solution A (Cross-lingual Orthogonal Projection)",
            "| Vector | Magnitude | cos(v, v_lang) |",
            "|--------|-----------|----------------|",
            f"| v_steer_raw | {metrics.get('v_steer_raw_magnitude', 0):.2f} | "
            f"{metrics.get('raw_lang_similarity', 0):.4f} |",
            f"| v_steer_scaled | {metrics.get('v_steer_scaled_magnitude', 0):.2f} | "
            f"{metrics.get('scaled_lang_similarity', 0):.6f} |",
            f"| Scale Factor | {metrics.get('scale_factor_A', 0):.2f}x | - |",
            "",
            "### Solution B (Monolingual Safety Anchoring)",
            "| Vector | Magnitude | cos(v, v_lang) |",
            "|--------|-----------|----------------|",
            f"| v_safety_mono | {metrics.get('v_safety_mono_magnitude', 0):.2f} | "
            f"{metrics.get('mono_lang_similarity', 0):.4f} |",
            f"| v_safety_mono_scaled | {metrics.get('v_safety_mono_scaled_magnitude', 0):.2f} | "
            f"{metrics.get('mono_scaled_lang_similarity', 0):.6f} |",
            f"| Scale Factor | {metrics.get('scale_factor_B', 0):.2f}x | - |",
        ]
        markdown = "\n".join(md_lines)
        
        # Save
        path = self.output_dir / "vector_metrics.md"
        with open(path, "w") as f:
            f.write(markdown)
        logger.info(f"Saved vector metrics to {path}")
        
        return markdown, ""

    def plot_steering_trajectory_pca(
        self,
        centroids: Dict[str, List[float]],
        vectors: Dict[str, List[float]],
        title: str = "Steering Trajectories in Representation Space (PCA)"
    ) -> plt.Figure:
        """
        Create PCA plot showing how steering vectors move representations.
        
        Shows:
        - English safe/unsafe centroids
        - Bengali safe/unsafe centroids  
        - Steering trajectories from Bengali unsafe toward different targets
        
        Args:
            centroids: Dict with en_safe, en_unsafe, bn_safe, bn_unsafe arrays
            vectors: Dict with steering vectors (v_steer_scaled, v_safety_mono_scaled)
        
        Returns:
            matplotlib Figure
        """
        # Convert to numpy
        en_safe = np.array(centroids["en_safe"])
        en_unsafe = np.array(centroids["en_unsafe"])
        bn_safe = np.array(centroids["bn_safe"])
        bn_unsafe = np.array(centroids["bn_unsafe"])
        
        v_steer_scaled = np.array(vectors.get("v_steer_scaled", np.zeros_like(en_safe)))
        v_safety_mono = np.array(vectors.get("v_safety_mono_scaled", np.zeros_like(en_safe)))
        
        # Compute steered positions
        bn_steered_A = bn_unsafe + v_steer_scaled  # Solution A destination
        bn_steered_B = bn_unsafe + v_safety_mono   # Solution B destination
        
        # Stack all points for PCA
        all_points = np.vstack([
            en_safe, en_unsafe, bn_safe, bn_unsafe,
            bn_steered_A, bn_steered_B
        ])
        
        # Fit PCA
        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(all_points)
        
        # Extract 2D coordinates
        en_safe_2d = points_2d[0]
        en_unsafe_2d = points_2d[1]
        bn_safe_2d = points_2d[2]
        bn_unsafe_2d = points_2d[3]
        bn_steered_A_2d = points_2d[4]
        bn_steered_B_2d = points_2d[5]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw cluster ellipses (approximate)
        from matplotlib.patches import Ellipse
        
        # English cluster (green ellipse)
        en_center = (en_safe_2d + en_unsafe_2d) / 2
        en_radius = np.linalg.norm(en_safe_2d - en_unsafe_2d) * 0.8
        english_ellipse = Ellipse(
            en_center, en_radius * 1.5, en_radius,
            alpha=0.15, color=self.colors["en_safe"],
            label="English Manifold"
        )
        ax.add_patch(english_ellipse)
        
        # Bengali cluster (blue ellipse)
        bn_center = (bn_safe_2d + bn_unsafe_2d) / 2
        bn_radius = np.linalg.norm(bn_safe_2d - bn_unsafe_2d) * 0.8
        bengali_ellipse = Ellipse(
            bn_center, bn_radius * 1.5, bn_radius,
            alpha=0.15, color=self.colors["bn_safe"],
            label="Bengali Manifold"
        )
        ax.add_patch(bengali_ellipse)
        
        # Plot centroids
        ax.scatter(*en_safe_2d, c=self.colors["en_safe"], s=200, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5, label="EN Safe")
        ax.scatter(*en_unsafe_2d, c=self.colors["en_unsafe"], s=200, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=5, label="EN Unsafe")
        ax.scatter(*bn_safe_2d, c=self.colors["bn_safe"], s=200, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN Safe")
        ax.scatter(*bn_unsafe_2d, c=self.colors["bn_unsafe"], s=200, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN Unsafe")
        
        # Plot steered positions
        ax.scatter(*bn_steered_A_2d, c=self.colors["bn_steered_A"], s=150, marker='s',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN + Sol. A")
        ax.scatter(*bn_steered_B_2d, c=self.colors["bn_steered_B"], s=150, marker='D',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN + Sol. B")
        
        # Draw steering arrows
        # Solution A arrow (Orange - moves toward English)
        ax.annotate('', xy=bn_steered_A_2d, xytext=bn_unsafe_2d,
                    arrowprops=dict(arrowstyle='->', color=self.colors["bn_steered_A"],
                                    lw=2.5, mutation_scale=15))
        
        # Solution B arrow (Teal - stays in Bengali region)
        ax.annotate('', xy=bn_steered_B_2d, xytext=bn_unsafe_2d,
                    arrowprops=dict(arrowstyle='->', color=self.colors["bn_steered_B"],
                                    lw=2.5, mutation_scale=15))
        
        # Add text labels
        offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.annotate("EN Safe", en_safe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("EN Unsafe", en_unsafe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("BN Safe", bn_safe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("BN Unsafe\n(Start)", bn_unsafe_2d, xytext=(5, -15), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["en_safe"],
                   markersize=10, markeredgecolor='black', label='EN Safe'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors["en_unsafe"],
                   markersize=10, markeredgecolor='black', label='EN Unsafe'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["bn_safe"],
                   markersize=10, markeredgecolor='black', label='BN Safe'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors["bn_unsafe"],
                   markersize=10, markeredgecolor='black', label='BN Unsafe'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors["bn_steered_A"],
                   markersize=10, markeredgecolor='black', label='Sol. A (Cross-lingual)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=self.colors["bn_steered_B"],
                   markersize=10, markeredgecolor='black', label='Sol. B (Monolingual)'),
            mpatches.Patch(color=self.colors["en_safe"], alpha=0.3, label='English Manifold'),
            mpatches.Patch(color=self.colors["bn_safe"], alpha=0.3, label='Bengali Manifold'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
                  framealpha=0.95, edgecolor='gray')
        
        # Add annotation box
        textstr = ("Solution A: Crosses into English cluster\n"
                   "(causes language bleed)\n\n"
                   "Solution B: Stays in Bengali manifold\n"
                   "(preserves language)")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"steering_trajectory_pca.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA plot to {self.output_dir}/steering_trajectory_pca.[png|pdf]")
        
        return fig

    def plot_steering_trajectory_tsne(
        self,
        centroids: Dict[str, List[float]],
        vectors: Dict[str, List[float]],
        title: str = "Steering Trajectories in Representation Space (t-SNE)"
    ) -> plt.Figure:
        """
        Create t-SNE plot showing steering trajectories.
        Similar to PCA but with non-linear dimensionality reduction.
        """
        # Convert to numpy
        en_safe = np.array(centroids["en_safe"])
        en_unsafe = np.array(centroids["en_unsafe"])
        bn_safe = np.array(centroids["bn_safe"])
        bn_unsafe = np.array(centroids["bn_unsafe"])
        
        v_steer_scaled = np.array(vectors.get("v_steer_scaled", np.zeros_like(en_safe)))
        v_safety_mono = np.array(vectors.get("v_safety_mono_scaled", np.zeros_like(en_safe)))
        
        # Compute steered positions
        bn_steered_A = bn_unsafe + v_steer_scaled
        bn_steered_B = bn_unsafe + v_safety_mono
        
        # Stack all points
        all_points = np.vstack([
            en_safe, en_unsafe, bn_safe, bn_unsafe,
            bn_steered_A, bn_steered_B
        ])
        
        # Fit t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=3, n_iter=1000)
        points_2d = tsne.fit_transform(all_points)
        
        # Extract 2D coordinates
        en_safe_2d = points_2d[0]
        en_unsafe_2d = points_2d[1]
        bn_safe_2d = points_2d[2]
        bn_unsafe_2d = points_2d[3]
        bn_steered_A_2d = points_2d[4]
        bn_steered_B_2d = points_2d[5]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw cluster ellipses
        from matplotlib.patches import Ellipse
        
        en_center = (en_safe_2d + en_unsafe_2d) / 2
        en_radius = np.linalg.norm(en_safe_2d - en_unsafe_2d) * 0.7
        english_ellipse = Ellipse(
            en_center, en_radius * 2, en_radius * 1.5,
            alpha=0.15, color=self.colors["en_safe"]
        )
        ax.add_patch(english_ellipse)
        
        bn_center = (bn_safe_2d + bn_unsafe_2d) / 2
        bn_radius = np.linalg.norm(bn_safe_2d - bn_unsafe_2d) * 0.7
        bengali_ellipse = Ellipse(
            bn_center, bn_radius * 2, bn_radius * 1.5,
            alpha=0.15, color=self.colors["bn_safe"]
        )
        ax.add_patch(bengali_ellipse)
        
        # Plot centroids
        ax.scatter(*en_safe_2d, c=self.colors["en_safe"], s=200, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5, label="EN Safe")
        ax.scatter(*en_unsafe_2d, c=self.colors["en_unsafe"], s=200, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=5, label="EN Unsafe")
        ax.scatter(*bn_safe_2d, c=self.colors["bn_safe"], s=200, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN Safe")
        ax.scatter(*bn_unsafe_2d, c=self.colors["bn_unsafe"], s=200, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN Unsafe")
        
        # Plot steered positions
        ax.scatter(*bn_steered_A_2d, c=self.colors["bn_steered_A"], s=150, marker='s',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN + Sol. A")
        ax.scatter(*bn_steered_B_2d, c=self.colors["bn_steered_B"], s=150, marker='D',
                   edgecolors='black', linewidths=1.5, zorder=5, label="BN + Sol. B")
        
        # Draw arrows
        ax.annotate('', xy=bn_steered_A_2d, xytext=bn_unsafe_2d,
                    arrowprops=dict(arrowstyle='->', color=self.colors["bn_steered_A"],
                                    lw=2.5, mutation_scale=15))
        ax.annotate('', xy=bn_steered_B_2d, xytext=bn_unsafe_2d,
                    arrowprops=dict(arrowstyle='->', color=self.colors["bn_steered_B"],
                                    lw=2.5, mutation_scale=15))
        
        # Labels
        ax.annotate("EN Safe", en_safe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("EN Unsafe", en_unsafe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("BN Safe", bn_safe_2d, xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        ax.annotate("BN Unsafe", bn_unsafe_2d, xytext=(5, -15), textcoords='offset points',
                    fontsize=9, fontweight='bold')
        
        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["en_safe"],
                   markersize=10, markeredgecolor='black', label='EN Safe'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors["en_unsafe"],
                   markersize=10, markeredgecolor='black', label='EN Unsafe'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors["bn_safe"],
                   markersize=10, markeredgecolor='black', label='BN Safe'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors["bn_unsafe"],
                   markersize=10, markeredgecolor='black', label='BN Unsafe'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors["bn_steered_A"],
                   markersize=10, markeredgecolor='black', label='Sol. A (Cross-lingual)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor=self.colors["bn_steered_B"],
                   markersize=10, markeredgecolor='black', label='Sol. B (Monolingual)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"steering_trajectory_tsne.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot to {self.output_dir}/steering_trajectory_tsne.[png|pdf]")
        
        return fig

    def plot_comparison_bar_chart(
        self,
        results: Dict[str, Any],
        title: str = "Steering Methods Comparison"
    ) -> plt.Figure:
        """
        Create grouped bar chart comparing all steering methods.
        """
        summary = results["summary"]
        
        methods = ['Baseline', 'Raw\n(A)', 'Scaled\n(A)', 'Mono\n(B)', 'Mono Scaled\n(B)']
        refusal_rates = [
            summary["baseline_refusal_rate"] * 100,
            summary["raw_refusal_rate"] * 100,
            summary["scaled_refusal_rate"] * 100,
            summary["mono_refusal_rate"] * 100,
            summary["mono_scaled_refusal_rate"] * 100,
        ]
        english_rates = [
            summary["baseline_english_rate"] * 100,
            summary["raw_english_rate"] * 100,
            summary["scaled_english_rate"] * 100,
            summary["mono_english_rate"] * 100,
            summary["mono_scaled_english_rate"] * 100,
        ]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, refusal_rates, width, label='Refusal Rate (%)',
                       color='#27ae60', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, english_rates, width, label='English Response Rate (%)',
                       color='#e74c3c', edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add "Language Bleed" annotation for high English rates
        for i, (method, eng_rate) in enumerate(zip(methods, english_rates)):
            if eng_rate > 50:
                ax.annotate('Language\nBleed!', 
                           xy=(x[i] + width/2, eng_rate + 5),
                           fontsize=8, color='darkred', fontweight='bold',
                           ha='center')
        
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_xlabel('Steering Method', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, max(max(refusal_rates), max(english_rates)) + 15)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add "Goal" annotation
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target Refusal')
        ax.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Target English')
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"comparison_bar_chart.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bar chart to {self.output_dir}/comparison_bar_chart.[png|pdf]")
        
        return fig

    def plot_vector_similarity_heatmap(
        self,
        vectors: Dict[str, List[float]],
        title: str = "Vector Cosine Similarity Matrix"
    ) -> plt.Figure:
        """
        Create heatmap showing cosine similarities between all vectors.
        """
        import torch.nn.functional as F
        import torch
        
        vector_names = ['v_lang', 'v_steer_raw', 'v_steer_scaled', 
                        'v_safety_mono', 'v_safety_mono_scaled']
        display_names = ['Language\nAxis', 'Raw\n(A)', 'Scaled\n(A)', 
                         'Mono\n(B)', 'Mono Scaled\n(B)']
        
        # Build similarity matrix
        n = len(vector_names)
        sim_matrix = np.zeros((n, n))
        
        for i, name_i in enumerate(vector_names):
            for j, name_j in enumerate(vector_names):
                if vectors.get(name_i) is not None and vectors.get(name_j) is not None:
                    v_i = torch.tensor(vectors[name_i]).float()
                    v_j = torch.tensor(vectors[name_j]).float()
                    sim = F.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0)).item()
                    sim_matrix[i, j] = sim
                else:
                    sim_matrix[i, j] = np.nan
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create heatmap
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom", fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(display_names, fontsize=10)
        ax.set_yticklabels(display_names, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if not np.isnan(sim_matrix[i, j]):
                    text_color = "white" if abs(sim_matrix[i, j]) > 0.5 else "black"
                    text = ax.text(j, i, f'{sim_matrix[i, j]:.3f}',
                                   ha="center", va="center", color=text_color,
                                   fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Highlight orthogonal pairs (near 0)
        for i in range(n):
            for j in range(n):
                if not np.isnan(sim_matrix[i, j]) and abs(sim_matrix[i, j]) < 0.01:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                          edgecolor='blue', linewidth=3)
                    ax.add_patch(rect)
        
        plt.tight_layout()
        
        # Save
        for fmt in ['png', 'pdf']:
            path = self.output_dir / f"vector_similarity_heatmap.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {self.output_dir}/vector_similarity_heatmap.[png|pdf]")
        
        return fig

    def generate_all(
        self,
        results_path: Path = None,
        vectors_path: Path = None,
        results: Dict = None,
        vectors_data: Dict = None
    ) -> None:
        """
        Generate all visualizations from saved results or provided data.
        
        Args:
            results_path: Path to orthogonal_steering_results.json
            vectors_path: Path to orthogonal_vectors.json
            results: Pre-loaded results dict
            vectors_data: Pre-loaded vectors dict
        """
        # Load data if paths provided
        if results is None and results_path:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        if vectors_data is None and vectors_path:
            with open(vectors_path, 'r') as f:
                vectors_data = json.load(f)
        
        if results is None or vectors_data is None:
            raise ValueError("Must provide either paths or data dicts")
        
        metrics = vectors_data.get("metrics", {})
        centroids = vectors_data.get("centroids", {})
        vectors = vectors_data.get("vectors", {})
        
        logger.info("=" * 60)
        logger.info("GENERATING PHASE 4 VISUALIZATIONS")
        logger.info("=" * 60)
        
        # 1. Evaluation tables
        logger.info("\n1. Generating evaluation tables...")
        self.generate_evaluation_table(results, metrics)
        self.generate_vector_metrics_table(metrics)
        
        # 2. PCA plot
        logger.info("\n2. Generating PCA trajectory plot...")
        try:
            self.plot_steering_trajectory_pca(centroids, vectors)
        except Exception as e:
            logger.warning(f"Could not generate PCA plot: {e}")
        
        # 3. t-SNE plot
        logger.info("\n3. Generating t-SNE trajectory plot...")
        try:
            self.plot_steering_trajectory_tsne(centroids, vectors)
        except Exception as e:
            logger.warning(f"Could not generate t-SNE plot: {e}")
        
        # 4. Comparison bar chart
        logger.info("\n4. Generating comparison bar chart...")
        self.plot_comparison_bar_chart(results)
        
        # 5. Similarity heatmap
        logger.info("\n5. Generating vector similarity heatmap...")
        try:
            self.plot_vector_similarity_heatmap(vectors)
        except Exception as e:
            logger.warning(f"Could not generate heatmap: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"All visualizations saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        # Close all figures to free memory
        plt.close('all')


def generate_phase4_visualizations(
    run_dir: Path,
    output_dir: Path = None
) -> None:
    """
    Convenience function to generate all Phase 4 visualizations.
    
    Args:
        run_dir: Path to Phase 4 run directory containing results files
        output_dir: Optional output directory (defaults to run_dir/charts)
    """
    run_dir = Path(run_dir)
    
    results_path = run_dir / "orthogonal_steering_results.json"
    vectors_path = run_dir / "orthogonal_vectors.json"
    
    if not results_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(f"Results files not found in {run_dir}")
    
    if output_dir is None:
        output_dir = run_dir / "charts"
    
    visualizer = Phase4Visualizer(output_dir=output_dir)
    visualizer.generate_all(results_path=results_path, vectors_path=vectors_path)
