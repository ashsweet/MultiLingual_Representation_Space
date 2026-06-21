#!/usr/bin/env python3
"""
Multilingual Safety Analysis Pipeline

A three-phase pipeline analyzing safety drift between English and Bengali 
LLM responses using the google/gemma-2-2b-it model.

Phase 1: Baseline Behavioral Profiling
Phase 2: Hidden State Analysis  
Phase 3: Latent Steering Intervention
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def run_phase1(config, max_samples: int = None, use_cache: bool = True, 
               refresh_cache: bool = False) -> dict:
    """Phase 1: Baseline Behavioral Profiling."""
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE BEHAVIORAL PROFILING")
    print("=" * 70 + "\n")
    
    from src.pipeline import Pipeline
    
    pipeline = Pipeline(config, max_samples=max_samples)
    pipeline.use_cache = use_cache
    pipeline.refresh_cache = refresh_cache
    
    result = pipeline.run()
    
    print("-" * 70)
    print(f"Phase 1 Complete - Output: {result['output_dir']}")
    print(f"Fragmentation failures: {result['report'].token_fragmentation.count}")
    print("-" * 70)
    
    return result


def run_phase2(config, num_prompts: int = 35, responses_path: Path = None) -> dict:
    """Phase 2: Hidden State Analysis."""
    print("\n" + "=" * 70)
    print("PHASE 2: HIDDEN STATE ANALYSIS")
    print("=" * 70)
    print("Method A: Internal Boundary Collapse")
    print("Method B: Cross-Lingual Drift")
    print("=" * 70 + "\n")
    
    from src.model_loader import ModelLoader
    from src.hidden_state_analyzer import HiddenStateAnalyzer, load_phase1_prompts
    from src.visualizer import Phase2Visualizer
    
    logger.info("Loading model for hidden state extraction")
    loader = ModelLoader(config)
    model, tokenizer = loader.load()
    
    analyzer = HiddenStateAnalyzer(model, tokenizer, config)
    safe_prompts, unsafe_prompts = load_phase1_prompts(responses_path, num_prompts)
    
    print(f"Loaded {len(safe_prompts)} safe prompts, {len(unsafe_prompts)} unsafe prompts\n")
    
    results = analyzer.run_full_analysis(
        safe_prompts=safe_prompts,
        unsafe_prompts=unsafe_prompts,
        show_progress=True
    )
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/Phase2_Hidden_Analysis") / f"run_{run_id}"
    analyzer.save_results(results, output_dir)
    
    charts_dir = output_dir / "charts"
    visualizer = Phase2Visualizer(output_dir=charts_dir)
    visualizer.generate_all(results_base_path=output_dir)
    visualizer.print_summary_table(results_base_path=output_dir)
    
    collapse = results["internal_boundary_collapse"]["summary"]
    drift = results["cross_lingual_drift"]["summary"]
    
    print("-" * 70)
    print(f"Phase 2 Complete - Output: {output_dir}")
    print(f"Max collapse at layer: {collapse['max_collapse_layer']}")
    print(f"Max drift at layer: {drift['max_drift_layer']}")
    print("-" * 70)
    
    loader.unload()
    
    return {
        "output_dir": output_dir,
        "results": results,
        "collapse_summary": collapse,
        "drift_summary": drift
    }


def run_phase3(config, target_layer: int = 13, alpha: float = 1.5, 
               num_prompts: int = 35) -> dict:
    """Phase 3: Latent Steering Intervention."""
    print("\n" + "=" * 70)
    print("PHASE 3: LATENT STEERING INTERVENTION")
    print("=" * 70)
    print(f"Target Layer: {target_layer}, Alpha: {alpha}")
    print("=" * 70 + "\n")
    
    from src.model_loader import ModelLoader
    from src.latent_steering import LatentSteering
    from src.hidden_state_analyzer import load_phase1_prompts
    
    logger.info("Loading model for steering")
    loader = ModelLoader(config)
    model, tokenizer = loader.load()
    
    steerer = LatentSteering(model, tokenizer, config, target_layer=target_layer)
    _, unsafe_prompts = load_phase1_prompts(max_prompts=num_prompts)
    
    print(f"Loaded {len(unsafe_prompts)} failure prompts\n")
    
    print("Computing steering vector...")
    steerer.compute_steering_vector(unsafe_prompts, show_progress=True)
    
    print("\nTesting steering...")
    refusal_patterns = config.refusal_patterns_en + config.refusal_patterns_bn
    
    results = steerer.test_steering(
        bn_prompts=unsafe_prompts,
        refusal_patterns=refusal_patterns,
        alpha=alpha,
        show_progress=True
    )
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/Phase3_Latent_Steering") / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    steerer.save_steering_vector(output_dir / "steering_vector.json")
    steerer.save_results(results, output_dir / "steering_results.json")
    
    improvement = results['refusal_rate_after'] - results['refusal_rate_before']
    
    print("\n" + "-" * 70)
    print("STEERING RESULTS")
    print("-" * 70)
    print(f"Refusals before: {results['refusals_before']}/{results['num_prompts']} "
          f"({results['refusal_rate_before']*100:.1f}%)")
    print(f"Refusals after:  {results['refusals_after']}/{results['num_prompts']} "
          f"({results['refusal_rate_after']*100:.1f}%)")
    print(f"Improvement: +{improvement*100:.1f}%")
    
    successes = [d for d in results["details"] if d["steering_worked"]]
    if successes:
        print("\nExample transformations:")
        for i, ex in enumerate(successes[:3]):
            print(f"  [{i+1}] ID={ex['id']}: {ex['response_before'][:60]}... -> refused")
    
    print("-" * 70)
    print(f"Phase 3 Complete - Output: {output_dir}")
    print("-" * 70)
    
    steerer.remove_steering()
    loader.unload()
    
    return {
        "output_dir": output_dir,
        "results": results,
        "improvement": improvement
    }


def main():
    parser = argparse.ArgumentParser(description="Multilingual Safety Analysis Pipeline")
    
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], 
                        help="Run specific phase only (default: all)")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=35)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=1.5)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("MULTILINGUAL SAFETY ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Mode: {'Phase ' + str(args.phase) if args.phase else 'All phases'}")
    print("=" * 70)
    
    try:
        from src.config import Config
        
        config = Config()
        config.model_name = args.model
        
        results = {}
        phases_to_run = [args.phase] if args.phase else [1, 2, 3]
        
        if 1 in phases_to_run:
            results["phase1"] = run_phase1(
                config,
                max_samples=args.max_samples,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache
            )
        
        if 2 in phases_to_run:
            responses_path = None
            if "phase1" in results:
                responses_path = Path(results["phase1"]["output_dir"]) / "responses.json"
            
            results["phase2"] = run_phase2(
                config,
                num_prompts=args.num_prompts,
                responses_path=responses_path
            )
        
        if 3 in phases_to_run:
            results["phase3"] = run_phase3(
                config,
                target_layer=args.layer,
                alpha=args.alpha,
                num_prompts=args.num_prompts
            )
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        for phase, res in results.items():
            print(f"{phase}: {res.get('output_dir', 'N/A')}")
        print("=" * 70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
