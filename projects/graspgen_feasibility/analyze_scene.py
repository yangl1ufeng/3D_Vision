"""One command: load scene -> segment -> point cloud -> grasp -> collision -> export."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import traceback
from offline_core.config import ROOT, load_config
from offline_core.export import write_json
from offline_core.official_sample import prepare_official_sample
from offline_core.pipeline import analyze
from offline_core.profiling import Profiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--official-sample', action='store_true', help='Use the already downloaded official bowl scene')
    inputs.add_argument('--input', type=Path, help='Aligned RGB/depth or RGB/XYZ scene manifest')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--candidates', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--approach', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable continuous straight approach collision checking')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--observation', action=argparse.BooleanOptionalAction, default=None,
                        help='Require calibrated depth evidence for the entire approach sweep')
    args = parser.parse_args()
    profiler = Profiler()
    output = (args.output or ROOT / 'runs/offline' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config(args.config, args.candidates, args.seed)
        if args.approach is not None:
            cfg['approach']['enabled'] = args.approach
            if not args.approach and args.observation is None:
                cfg['observation']['enabled'] = False
        if args.observation is not None:
            cfg['observation']['enabled'] = args.observation
        if cfg['observation']['enabled'] and not cfg['approach']['enabled']:
            raise ValueError('observation checking requires approach geometry; enable --approach or disable --observation')
        if args.official_sample:
            with profiler.stage('official_sample_conversion'):
                manifest = prepare_official_sample()
        else:
            manifest = args.input.resolve()
        result = analyze(manifest, output, cfg, profiler)
        counts = result['counts']
        print(f"OK: raw={counts['raw']}, eligible={counts['eligible_for_collision']}, accepted={counts['accepted']}")
        insufficient = counts['exclusion_reasons'].get('insufficient_observation', 0)
        missing_calibration = counts['exclusion_reasons'].get('observation_missing_calibration', 0)
        if insufficient:
            print(f'Insufficient observation: {insufficient} candidates need additional depth views.')
        if missing_calibration:
            print(f'Missing calibrated depth: {missing_calibration} candidates cannot be visibility-verified.')
        print(f"Compute: {result['performance']['compute_stage_seconds']:.3f}s; CUDA peak: {result['performance']['peak_allocated_mib']:.1f} MiB")
        print(f'Output: {output}')
        return 0
    except Exception as exc:
        write_json(output / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                                          'traceback': traceback.format_exc(), 'performance': profiler.summary()})
        print(f'FAILED: {exc}\nDetails: {output / "failure.json"}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
