"""Visualization for safety analysis (Phase 1 & Phase 2)."""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from .analyzers.combined import SafetyResult
from .report import DriftReport


class Visualizer:
    """Creates charts for Phase 1 analysis results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_errors(self, report: DriftReport) -> Path:
        """Bar chart of error categories."""
        data = [
            ("Refusal\nMismatch", report.refusal_mismatch.percentage),
            ("Token\nFragmentation", report.token_fragmentation.percentage),
            ("Faux\nRefusal", report.faux_refusal.percentage),
            ("Confident\nHallucination", report.confident_hallucination.percentage),
            ("Incoherent\nResponse", report.incoherent_response.percentage),
        ]
        names, vals = zip(*data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#e67e22', '#9b59b6']
        bars = ax.bar(names, vals, color=colors, edgecolor='black')
        
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Safety Error Distribution (Bengali vs English)', fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.3 if vals else 10)
        
        plt.tight_layout()
        path = self.output_dir / "error_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    
    def plot_severity(self, report: DriftReport) -> Path:
        """Pie chart of severity."""
        colors = {"low": "#2ecc71", "medium": "#f1c40f", "high": "#e67e22", "critical": "#e74c3c"}
        labels, sizes, cols = [], [], []
        
        for sev in ["low", "medium", "high", "critical"]:
            cnt = report.severity_dist.get(sev, 0)
            if cnt > 0:
                labels.append(f"{sev.capitalize()}\n({cnt})")
                sizes.append(cnt)
                cols.append(colors[sev])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        if sizes:
            ax.pie(sizes, labels=labels, colors=cols, autopct='%1.1f%%', startangle=90, explode=[0.02]*len(sizes))
        ax.set_title('Severity Distribution', fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / "severity.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    
    def plot_histogram(self, results: List[SafetyResult]) -> Path:
        """Histogram of safety scores."""
        scores = [r.safety_score for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(scores, bins=20, edgecolor='black')
        
        for patch, left in zip(patches, bins[:-1]):
            if left < 0.3: patch.set_facecolor('#e74c3c')
            elif left < 0.6: patch.set_facecolor('#f39c12')
            elif left < 0.8: patch.set_facecolor('#f1c40f')
            else: patch.set_facecolor('#2ecc71')
        
        mean = np.mean(scores)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
        
        ax.set_xlabel('Safety Score')
        ax.set_ylabel('Count')
        ax.set_title('Safety Score Distribution', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        path = self.output_dir / "scores.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    
    def plot_comparison(self, results: List[SafetyResult]) -> Path:
        """EN vs BN comparison."""
        en_ref = bn_ref = 0
        for r in results:
            if r.refusal.get("english", {}).get("type") in ["clean_refusal", "partial_refusal"]:
                en_ref += 1
            if r.refusal.get("bengali", {}).get("type") in ["clean_refusal", "partial_refusal"]:
                bn_ref += 1
        
        total = len(results) or 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [en_ref/total*100, (total-en_ref)/total*100], width, label='English', color='#3498db')
        ax.bar(x + width/2, [bn_ref/total*100, (total-bn_ref)/total*100], width, label='Bengali', color='#e74c3c')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('English vs Bengali Response Behavior', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Refusals', 'Compliance'])
        ax.legend()
        
        plt.tight_layout()
        path = self.output_dir / "comparison.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path
    
    def generate_all(self, results: List[SafetyResult], report: DriftReport) -> List[Path]:
        """Generate all Phase 1 charts."""
        paths = []
        for fn in [lambda: self.plot_errors(report), lambda: self.plot_severity(report),
                   lambda: self.plot_histogram(results), lambda: self.plot_comparison(results)]:
            try:
                paths.append(fn())
            except Exception as e:
                print(f"Chart error: {e}")
        return paths


class Phase2Visualizer:
    """Creates charts for Phase 2 hidden state analysis results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib for publication-quality figures."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (8, 5),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.family': 'serif'
        })
    
    def load_results(self, base_path: Path) -> tuple:
        """Load Phase 2 analysis results from JSON files."""
        with open(base_path / "internal_boundary_collapse" / "results.json") as f:
            collapse_data = json.load(f)
        with open(base_path / "cross_lingual_drift" / "results.json") as f:
            drift_data = json.load(f)
        return collapse_data, drift_data
    
    def extract_metrics(self, collapse_data: Dict, drift_data: Dict) -> Dict[str, Any]:
        """Extract metrics by layer from result data."""
        layers = [0, 6, 13, 19, 25]
        layer_labels = ['0', '6', '13', '19', '25']
        
        en_safe_unsafe, bn_safe_unsafe, collapse_scores = [], [], []
        en_bn_unsafe, drift_scores = [], []
        
        for layer in layers:
            layer_str = str(layer)
            # Collapse metrics
            en_safe_unsafe.append(collapse_data['layer_results'][layer_str]['en_safe_vs_unsafe']['mean'])
            bn_safe_unsafe.append(collapse_data['layer_results'][layer_str]['bn_safe_vs_unsafe']['mean'])
            collapse_scores.append(collapse_data['layer_results'][layer_str]['collapse_score'])
            # Drift metrics
            en_bn_unsafe.append(drift_data['layer_results'][layer_str]['en_unsafe_vs_bn_unsafe']['mean'])
            drift_scores.append(drift_data['layer_results'][layer_str]['drift_score'])
        
        return {
            'layers': layers,
            'layer_labels': layer_labels,
            'en_safe_unsafe': en_safe_unsafe,
            'bn_safe_unsafe': bn_safe_unsafe,
            'collapse_scores': collapse_scores,
            'en_bn_unsafe': en_bn_unsafe,
            'drift_scores': drift_scores
        }
    
    def plot_collapse_scores(self, metrics: Dict) -> Path:
        """Figure 3: Layer-wise collapse scores."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        x = np.arange(len(metrics['layers']))
        colors = ['#2ecc71' if s < 0 else '#e74c3c' for s in metrics['collapse_scores']]
        
        ax.bar(x, metrics['collapse_scores'], color=colors, edgecolor='black', linewidth=0.8, width=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Collapse Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics['layer_labels'])
        ax.set_ylim(-0.035, 0.02)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='black', label='Negative (No Collapse)'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Positive (Collapse)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
        
        plt.tight_layout()
        path = self.output_dir / 'figure3_collapse_scores.png'
        fig.savefig(path)
        fig.savefig(self.output_dir / 'figure3_collapse_scores.pdf')
        plt.close(fig)
        return path
    
    def plot_drift_scores(self, metrics: Dict) -> Path:
        """Figure 4: Layer-wise drift scores."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        x = np.arange(len(metrics['layers']))
        colors = []
        for d in metrics['drift_scores']:
            if d < 0.1:
                colors.append('#2ecc71')
            elif d < 0.3:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')
        
        ax.bar(x, metrics['drift_scores'], color=colors, edgecolor='black', linewidth=0.8, width=0.6)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Drift Score')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics['layer_labels'])
        ax.set_ylim(0, 0.6)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='black', label='Low (<0.1)'),
            Patch(facecolor='#f39c12', edgecolor='black', label='Moderate (0.1–0.3)'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Severe (>0.3)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        path = self.output_dir / 'figure4_drift_scores.png'
        fig.savefig(path)
        fig.savefig(self.output_dir / 'figure4_drift_scores.pdf')
        plt.close(fig)
        return path
    
    def plot_similarity_comparison(self, metrics: Dict) -> Path:
        """Figure 5: EN vs BN safe-unsafe similarity comparison."""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        x = np.arange(len(metrics['layers']))
        width = 0.35
        
        ax.bar(x - width/2, metrics['en_safe_unsafe'], width, 
               label='English', color='#3498db', edgecolor='black', linewidth=0.8)
        ax.bar(x + width/2, metrics['bn_safe_unsafe'], width,
               label='Bengali', color='#e74c3c', edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity (Safe vs Unsafe)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics['layer_labels'])
        ax.set_ylim(0.88, 0.98)
        ax.legend(loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        path = self.output_dir / 'figure5_similarity_comparison.png'
        fig.savefig(path)
        fig.savefig(self.output_dir / 'figure5_similarity_comparison.pdf')
        plt.close(fig)
        return path
    
    def plot_combined_analysis(self, metrics: Dict) -> Path:
        """Figure 6: Combined collapse and drift on dual y-axis."""
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(metrics['layers']))
        
        color1 = '#3498db'
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Collapse Score', color=color1)
        line1 = ax1.plot(x, metrics['collapse_scores'], 'o-', color=color1, 
                         linewidth=2, markersize=8, label='Collapse Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.axhline(y=0, color=color1, linestyle='--', alpha=0.4)
        
        ax2 = ax1.twinx()
        color2 = '#e74c3c'
        ax2.set_ylabel('Drift Score', color=color2)
        line2 = ax2.plot(x, metrics['drift_scores'], 's--', color=color2,
                         linewidth=2, markersize=8, label='Drift Score')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics['layer_labels'])
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        path = self.output_dir / 'figure6_combined_analysis.png'
        fig.savefig(path)
        fig.savefig(self.output_dir / 'figure6_combined_analysis.pdf')
        plt.close(fig)
        return path
    
    def generate_all(self, results_base_path: Path) -> List[Path]:
        """Generate all Phase 2 charts from result files."""
        collapse_data, drift_data = self.load_results(results_base_path)
        metrics = self.extract_metrics(collapse_data, drift_data)
        
        paths = []
        for fn in [
            lambda: self.plot_collapse_scores(metrics),
            lambda: self.plot_drift_scores(metrics),
            lambda: self.plot_similarity_comparison(metrics),
            lambda: self.plot_combined_analysis(metrics)
        ]:
            try:
                paths.append(fn())
            except Exception as e:
                print(f"Phase 2 chart error: {e}")
        
        return paths
    
    def print_summary_table(self, results_base_path: Path):
        """Print summary table for paper."""
        collapse_data, drift_data = self.load_results(results_base_path)
        metrics = self.extract_metrics(collapse_data, drift_data)
        
        print("\n" + "=" * 75)
        print("SUMMARY TABLE FOR PAPER")
        print("=" * 75)
        print("\nLayer | EN safe-unsafe | BN safe-unsafe | Collapse | EN-BN unsafe | Drift")
        print("-" * 75)
        for i, layer in enumerate(metrics['layers']):
            print(f"  {layer:2d}  |     {metrics['en_safe_unsafe'][i]:.3f}      |"
                  f"     {metrics['bn_safe_unsafe'][i]:.3f}      |"
                  f" {metrics['collapse_scores'][i]:+.3f}  |"
                  f"    {metrics['en_bn_unsafe'][i]:.3f}     |"
                  f" {metrics['drift_scores'][i]:.3f}")
