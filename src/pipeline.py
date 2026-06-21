"""Main evaluation pipeline."""
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from loguru import logger

from .config import Config, get_config
from .model_loader import ModelLoader
from .data_loader import DatasetLoader
from .inference import InferenceEngine
from .analyzers import SafetyAnalyzer
from .report import DriftCalculator
from .visualizer import Visualizer


class Pipeline:
    """Complete safety evaluation pipeline."""
    
    def __init__(self, config: Optional[Config] = None, max_samples: Optional[int] = None):
        self.config = config or get_config()
        if max_samples:
            self.config.max_samples = max_samples
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.config.results_dir / "Phase1_Behavioral_Profiling" / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = ModelLoader(self.config)
        self.data = DatasetLoader(self.config)
        self.analyzer = SafetyAnalyzer(self.config)
        self.drift = DriftCalculator()
        self.viz = Visualizer(self.run_dir / "charts")
        
        # Cache options
        self.use_cache = True
        self.refresh_cache = False
    
    def _save_data_files(self, prompts: List[Dict], results: List) -> None:
        """Save baseline prompts and failure cohort to data/ directory."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save 150 baseline prompts
        baseline_prompts = []
        for p in prompts:
            baseline_prompts.append({
                "id": p["id"],
                "en": p["en"],
                "bn": p["bn"],
                "category": p.get("category", "unknown")
            })
        
        baseline_path = data_dir / "150_baseline_prompts.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_prompts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(baseline_prompts)} baseline prompts to {baseline_path}")
        
        # Extract failure cohort (fragmentation issues)
        failure_ids = []
        for r in results:
            if "token_fragmentation" in r.issues:
                failure_ids.append(r.prompt_id)
        
        failure_cohort = []
        for p in prompts:
            if p["id"] in failure_ids:
                failure_cohort.append({
                    "id": p["id"],
                    "en": p["en"],
                    "bn": p["bn"],
                    "category": p.get("category", "unknown")
                })
        
        cohort_path = data_dir / "35_failure_cohort.json"
        with open(cohort_path, "w", encoding="utf-8") as f:
            json.dump(failure_cohort, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(failure_cohort)} failure prompts to {cohort_path}")
    
    def run(self) -> Dict[str, Any]:
        """Execute full pipeline."""
        logger.info("=" * 60)
        logger.info("MULTILINGUAL SAFETY EVALUATION PIPELINE")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info("=" * 60)
        
        start = datetime.now()
        responses = []
        results = []
        report = None
        charts = []
        
        try:
            # Load model
            logger.info("Loading model...")
            model, tokenizer = self.loader.load()
            engine = InferenceEngine(model, tokenizer, self.config)
            
            # Load data (with caching)
            logger.info("Loading dataset...")
            prompts = self.data.get_parallel_prompts(
                use_cache=self.use_cache,
                refresh_cache=self.refresh_cache
            )
            
            # Generate responses
            logger.info("Generating responses...")
            responses = engine.generate_parallel(prompts)
            
            # Save responses immediately after generation
            resp_file = self.run_dir / "responses.json"
            with open(resp_file, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved responses to {resp_file}")
            
            # Analyze
            logger.info("Analyzing safety...")
            results = self.analyzer.analyze_batch(responses)
            
            # Calculate drift
            logger.info("Calculating drift...")
            report = self.drift.calculate(results)
            
            # Export report immediately
            self.drift.export_json(report, results, self.run_dir / "report.json")
            logger.info("Saved report.json")
            
            # Save data files (baseline + failure cohort)
            self._save_data_files(prompts, results)
            
            # Save text report
            formatted = self.drift.format_report(report)
            with open(self.run_dir / "report.txt", "w") as f:
                f.write(formatted)
            logger.info("Saved report.txt")
            
            # Generate visualizations (in try block - non-critical)
            try:
                logger.info("Generating visualizations...")
                charts = self.viz.generate_all(results, report)
                logger.info(f"Generated {len(charts)} charts")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
            
            # Print report
            print("\n" + formatted)
            
            duration = (datetime.now() - start).total_seconds()
            
            logger.info("=" * 60)
            logger.info(f"Complete in {duration:.1f}s")
            logger.info(f"Results: {self.run_dir}")
            logger.info("=" * 60)
            
            return {
                "run_id": self.run_id,
                "duration": duration,
                "samples": len(prompts),
                "report": report,
                "results": results,
                "charts": charts,
                "output_dir": str(self.run_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Try to save whatever we have
            if responses:
                try:
                    with open(self.run_dir / "responses_partial.json", "w", encoding="utf-8") as f:
                        json.dump(responses, f, indent=2, ensure_ascii=False)
                    logger.info("Saved partial responses")
                except:
                    pass
            raise
            
        finally:
            self.loader.unload()
    
    def analyze_only(self, responses_file: Path) -> Dict[str, Any]:
        """Run analysis on existing responses."""
        logger.info(f"Loading responses from {responses_file}")
        
        with open(responses_file, "r", encoding="utf-8") as f:
            responses = json.load(f)
        
        logger.info(f"Analyzing {len(responses)} responses...")
        results = self.analyzer.analyze_batch(responses)
        
        report = self.drift.calculate(results)
        charts = self.viz.generate_all(results, report)
        
        self.drift.export_json(report, results, self.run_dir / "report.json")
        
        formatted = self.drift.format_report(report)
        print("\n" + formatted)
        
        with open(self.run_dir / "report.txt", "w") as f:
            f.write(formatted)
        
        return {
            "run_id": self.run_id,
            "report": report,
            "results": results,
            "charts": charts,
            "output_dir": str(self.run_dir)
        }
