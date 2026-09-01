#!/usr/bin/env python3
"""Process calibration results and save to CSV format."""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

def main():
    project_root = Path(__file__).resolve().parent
    
    # Load calibration results
    calibration_path = project_root / "reports" / "calibration" / "calibration_result.json"
    with open(calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    # Create results directory
    results_dir = project_root / "results" / "calibration"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create before/after comparison data
    comparison_data = []
    
    for split_name in ['val', 'test_seen', 'test_unseen']:
        before_data = calibration_data['before_calibration'].get(split_name, {})
        after_data = calibration_data['after_calibration'].get(split_name, {})
        
        if before_data.get('support', 0) > 0:
            row = {
                'split': split_name,
                'support': before_data.get('support'),
                'before_ece': before_data.get('ece'),
                'after_ece': after_data.get('ece'),
                'ece_change': after_data.get('ece') - before_data.get('ece') if before_data.get('ece') and after_data.get('ece') else None,
                'before_brier': before_data.get('brier_score'),
                'after_brier': after_data.get('brier_score'),
                'brier_change': after_data.get('brier_score') - before_data.get('brier_score') if before_data.get('brier_score') and after_data.get('brier_score') else None,
                'before_nll': before_data.get('nll'),
                'after_nll': after_data.get('nll'),
                'nll_change': after_data.get('nll') - before_data.get('nll') if before_data.get('nll') and after_data.get('nll') else None,
                'before_f1': before_data.get('f1'),
                'after_f1': after_data.get('f1'),
                'f1_change': after_data.get('f1') - before_data.get('f1') if before_data.get('f1') and after_data.get('f1') else None,
                'temperature': calibration_data['temperature'],
                'calibration_note': 'T=1.0 (single class, no meaningful calibration possible)' if calibration_data['temperature'] == 1.0 else 'Calibrated'
            }
            comparison_data.append(row)
        else:
            # Handle empty splits
            row = {
                'split': split_name,
                'support': 0,
                'before_ece': None,
                'after_ece': None,
                'ece_change': None,
                'before_brier': None,
                'after_brier': None,
                'brier_change': None,
                'before_nll': None,
                'after_nll': None,
                'nll_change': None,
                'before_f1': None,
                'after_f1': None,
                'f1_change': None,
                'temperature': calibration_data['temperature'],
                'calibration_note': before_data.get('note', 'empty_split')
            }
            comparison_data.append(row)
    
    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = results_dir / "calibration_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Calibration comparison saved to {comparison_path}")
    
    # Create detailed metrics CSV
    detailed_data = []
    for phase in ['before_calibration', 'after_calibration']:
        for split_name in ['val', 'test_seen', 'test_unseen']:
            split_data = calibration_data[phase].get(split_name, {})
            if split_data.get('support', 0) > 0:
                row = {
                    'phase': phase,
                    'split': split_name,
                    'support': split_data.get('support'),
                    'ece': split_data.get('ece'),
                    'brier_score': split_data.get('brier_score'),
                    'nll': split_data.get('nll'),
                    'roc_auc': split_data.get('roc_auc'),
                    'f1': split_data.get('f1')
                }
                detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = results_dir / "calibration_detailed_metrics.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed metrics saved to {detailed_path}")
    
    # Create modality status
    modality_status = {
        'modality': ['IMAGE', 'VIDEO', 'AUDIO'],
        'model_available': ['Yes', 'No', 'No'],
        'calibration_evaluated': ['Yes', 'No', 'No'],
        'temperature_fit': ['1.0 (single class)', 'N/A', 'N/A'],
        'test_seen_support': [10, 0, 0],
        'test_unseen_support': [0, 0, 0],
        'calibration_effective': ['No (single class)', 'N/A', 'N/A']
    }
    
    status_df = pd.DataFrame(modality_status)
    status_path = results_dir / "modality_calibration_status.csv"
    status_df.to_csv(status_path, index=False)
    print(f"Modality status saved to {status_path}")
    
    # Copy original JSON to results directory
    import shutil
    json_dest = results_dir / "calibration_results.json"
    shutil.copy(calibration_path, json_dest)
    print(f"Original calibration results copied to {json_dest}")
    
    # Create summary
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'modality': 'IMAGE',
        'model': 'EfficientNet-B4',
        'checkpoint': str(calibration_data['checkpoint_path']),
        'temperature': calibration_data['temperature'],
        'calibration_method': 'Temperature Scaling',
        'test_seen_support': 10,
        'test_unseen_support': 0,
        'calibration_effective': False,
        'reason': 'Single class in validation set prevents meaningful temperature fitting',
        'before_calibration_ece': calibration_data['before_calibration']['test_seen']['ece'],
        'after_calibration_ece': calibration_data['after_calibration']['test_seen']['ece'],
        'ece_improvement': 0.0,  # No change when T=1.0
        'video_audio_status': 'No trained models available for calibration'
    }
    
    summary_path = results_dir / "calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Calibration summary saved to {summary_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())