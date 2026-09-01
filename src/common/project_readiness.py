"""
AEGIS Project Readiness Verification System

This module provides machine-checkable verification of the AEGIS project's
scientific readiness. It validates data pipeline integrity, split consistency,
model availability, and experimental reproducibility.

The system fails loudly rather than silently continuing with missing data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum


class ReadinessStatus(Enum):
    """Readiness status levels"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    """Result of a single readiness check"""
    check_name: str
    status: ReadinessStatus
    message: str
    details: Dict[str, Any]
    critical: bool = False


class ProjectReadinessChecker:
    """
    Comprehensive project readiness verification system.
    
    Validates dataset integrity, split consistency, model availability,
    and experimental reproducibility.
    """
    
    def __init__(self, project_root: str = None):
        """Initialize the checker with project root path."""
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)
        
        self.results: List[CheckResult] = []
        self._setup_paths()
    
    def _find_project_root(self) -> Path:
        """Auto-discover project root from current working directory."""
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            if (parent / "src").exists() and (parent / "configs").exists():
                return parent
        raise FileNotFoundError("Cannot find AEGIS project root")
    
    def _setup_paths(self):
        """Setup standard project paths."""
        self.paths = {
            "data_raw": self.project_root / "data" / "raw",
            "data_processed": self.project_root / "data" / "processed",
            "models": self.project_root / "models",
            "results": self.project_root / "results",
            "reports": self.project_root / "reports",
            "configs": self.project_root / "configs",
            "mlruns": self.project_root / "mlruns",
        }
        
        # Modality-specific paths
        for modality in ["image", "audio", "video"]:
            self.paths[f"data_raw_{modality}"] = self.paths["data_raw"] / modality
            self.paths[f"data_processed_{modality}"] = self.paths["data_processed"] / modality
            self.paths[f"models_{modality}"] = self.paths["models"] / modality
            self.paths[f"splits_{modality}"] = self.paths[f"data_processed_{modality}"] / "splits"
            self.paths[f"preprocessing_{modality}"] = self.paths[f"data_processed_{modality}"] / "preprocessing"
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all readiness checks and return comprehensive report."""
        print("Running AEGIS Project Readiness Checks...")
        print("=" * 60)
        
        # Dataset structure checks
        self._check_dataset_directories()
        self._check_processed_samples()
        
        # Split integrity checks
        self._check_split_files_exist()
        self._check_split_files_nonempty()
        self._check_unseen_generator_splits()
        
        # Data consistency checks
        self._check_preprocessing_metadata_match()
        self._check_generator_distributions()
        self._check_identity_leakage()
        self._check_generator_leakage()
        
        # Model and artifact checks
        self._check_model_checkpoints()
        self._check_calibration_files()
        self._check_evaluation_results()
        self._check_artifact_consistency()
        
        # Reproducibility checks
        self._check_experiment_configuration()
        self._check_random_seeds()
        self._check_mlflow_experiments()
        
        # Generate summary
        return self._generate_report()
    
    def _add_result(self, check_name: str, status: ReadinessStatus, 
                    message: str, details: Dict[str, Any] = None, 
                    critical: bool = False):
        """Add a check result to the results list."""
        self.results.append(CheckResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            critical=critical
        ))
        
        # Print immediate feedback for critical failures
        if critical and status != ReadinessStatus.PASS:
            print(f"[CRITICAL] {check_name} - {message}")
    
    def _check_dataset_directories(self):
        """Check that required dataset directories exist."""
        print("\n[CHECK] Checking dataset directories...")
        
        for modality in ["image", "audio", "video"]:
            raw_dir = self.paths[f"data_raw_{modality}"]
            processed_dir = self.paths[f"data_processed_{modality}"]
            
            # Check raw directory
            if raw_dir.exists():
                self._add_result(
                    f"raw_{modality}_dir_exists",
                    ReadinessStatus.PASS,
                    f"Raw {modality} directory exists",
                    {"path": str(raw_dir)}
                )
            else:
                self._add_result(
                    f"raw_{modality}_dir_exists",
                    ReadinessStatus.FAIL,
                    f"Raw {modality} directory missing",
                    {"expected_path": str(raw_dir)},
                    critical=True
                )
            
            # Check processed directory
            if processed_dir.exists():
                self._add_result(
                    f"processed_{modality}_dir_exists",
                    ReadinessStatus.PASS,
                    f"Processed {modality} directory exists",
                    {"path": str(processed_dir)}
                )
            else:
                self._add_result(
                    f"processed_{modality}_dir_exists",
                    ReadinessStatus.FAIL,
                    f"Processed {modality} directory missing",
                    {"expected_path": str(processed_dir)},
                    critical=True
                )
    
    def _check_processed_samples(self):
        """Check that pipeline infrastructure is working."""
        print("\n[CHECK] Checking processed pipeline infrastructure...")
        
        for modality in ["image", "audio", "video"]:
            preprocessing_dir = self.paths[f"preprocessing_{modality}"]
            
            if not preprocessing_dir.exists():
                self._add_result(
                    f"{modality}_preprocessing_dir_exists",
                    ReadinessStatus.FAIL,
                    f"{modality} preprocessing directory missing",
                    {},
                    critical=True
                )
                continue
            
            metadata_file = preprocessing_dir / "metadata.csv"
            if metadata_file.exists():
                try:
                    df = pd.read_csv(metadata_file)
                    sample_count = len(df)
                    
                    if sample_count > 0:
                        self._add_result(
                            f"{modality}_pipeline_active",
                            ReadinessStatus.PASS,
                            f"{modality} pipeline active ({sample_count} processed samples)",
                            {"count": sample_count}
                        )
                    else:
                        self._add_result(
                            f"{modality}_pipeline_active",
                            ReadinessStatus.WARNING,
                            f"{modality} pipeline empty (0 processed samples)",
                            {"count": 0}
                        )
                        
                    # Research readiness check (coverage > 90%)
                    registry_file = self.paths[f"data_processed_{modality}"] / "sample_registry.csv"
                    if registry_file.exists():
                        reg = pd.read_csv(registry_file, low_memory=False)
                        total_raw = len(reg)
                        processed = len(reg[reg['status'] == 'PROCESSED'])
                        coverage = processed / total_raw if total_raw > 0 else 0
                        
                        if coverage >= 0.90:
                            self._add_result(
                                f"{modality}_research_ready",
                                ReadinessStatus.PASS,
                                f"{modality} research ready (coverage: {coverage:.1%})",
                                {"coverage": coverage}
                            )
                        else:
                            self._add_result(
                                f"{modality}_research_ready",
                                ReadinessStatus.FAIL,
                                f"{modality} research NOT ready (coverage: {coverage:.1%} < 90%)",
                                {"coverage": coverage},
                                critical=True
                            )
                            
                except Exception as e:
                    self._add_result(
                        f"{modality}_processed_samples_readable",
                        ReadinessStatus.FAIL,
                        f"Cannot read {modality} preprocessing metadata: {e}",
                        {"error": str(e)},
                        critical=True
                    )
            else:
                self._add_result(
                    f"{modality}_preprocessing_metadata_exists",
                    ReadinessStatus.FAIL,
                    f"{modality} preprocessing metadata missing",
                    {"expected_path": str(metadata_file)},
                    critical=True
                )
    
    def _check_split_files_exist(self):
        """Check that split CSV files exist."""
        print("\n[CHECK] Checking split files...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            
            if not splits_dir.exists():
                self._add_result(
                    f"{modality}_splits_dir_exists",
                    ReadinessStatus.FAIL,
                    f"{modality} splits directory missing",
                    {},
                    critical=True
                )
                continue
            
            required_splits = ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]
            for split_file in required_splits:
                split_path = splits_dir / split_file
                if split_path.exists():
                    self._add_result(
                        f"{modality}_{split_file.replace('.csv', '')}_exists",
                        ReadinessStatus.PASS,
                        f"{modality} {split_file} exists",
                        {"path": str(split_path)}
                    )
                else:
                    self._add_result(
                        f"{modality}_{split_file.replace('.csv', '')}_exists",
                        ReadinessStatus.FAIL,
                        f"{modality} {split_file} missing",
                        {"expected_path": str(split_path)},
                        critical=(split_file == "test_unseen.csv")
                    )
    
    def _check_split_files_nonempty(self):
        """Check that split CSV files are non-empty (beyond header)."""
        print("\n[CHECK] Checking split file contents...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            
            if not splits_dir.exists():
                continue
            
            required_splits = ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]
            for split_file in required_splits:
                split_path = splits_dir / split_file
                if not split_path.exists():
                    continue
                
                try:
                    df = pd.read_csv(split_path)
                    sample_count = len(df)  # pandas already removes header
                    actual_samples = sample_count
                    
                    if actual_samples > 0:
                        self._add_result(
                            f"{modality}_{split_file.replace('.csv', '')}_nonempty",
                            ReadinessStatus.PASS,
                            f"{modality} {split_file} has {actual_samples} samples",
                            {"sample_count": actual_samples}
                        )
                    else:
                        status = ReadinessStatus.FAIL if split_file == "test_unseen.csv" else ReadinessStatus.WARNING
                        self._add_result(
                            f"{modality}_{split_file.replace('.csv', '')}_nonempty",
                            status,
                            f"{modality} {split_file} is empty",
                            {"sample_count": 0},
                            critical=(split_file == "test_unseen.csv")
                        )
                except Exception as e:
                    self._add_result(
                        f"{modality}_{split_file.replace('.csv', '')}_readable",
                        ReadinessStatus.FAIL,
                        f"Cannot read {modality} {split_file}: {e}",
                        {"error": str(e)}
                    )
    
    def _check_unseen_generator_splits(self):
        """Check that unseen-generator test sets are non-empty."""
        print("\n[CHECK] Checking unseen generator splits...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            test_unseen_path = splits_dir / "test_unseen.csv"
            
            if not test_unseen_path.exists():
                self._add_result(
                    f"{modality}_test_unseen_available",
                    ReadinessStatus.FAIL,
                    f"{modality} test_unseen split does not exist",
                    {},
                    critical=True
                )
                continue
            
            try:
                df = pd.read_csv(test_unseen_path)
                sample_count = len(df)  # pandas already removes header
                
                if sample_count > 0:
                    self._add_result(
                        f"{modality}_test_unseen_nonempty",
                        ReadinessStatus.PASS,
                        f"{modality} test_unseen has {sample_count} samples",
                        {"sample_count": sample_count}
                    )
                else:
                    self._add_result(
                        f"{modality}_test_unseen_nonempty",
                        ReadinessStatus.FAIL,
                        f"{modality} test_unseen is empty - CRITICAL BLOCKER",
                        {"sample_count": 0},
                        critical=True
                    )
            except Exception as e:
                self._add_result(
                    f"{modality}_test_unseen_readable",
                    ReadinessStatus.FAIL,
                    f"Cannot read {modality} test_unseen: {e}",
                    {"error": str(e)},
                    critical=True
                )
    
    def _check_preprocessing_metadata_match(self):
        """Check that preprocessing metadata matches actual processed files."""
        print("\n[CHECK] Checking preprocessing metadata consistency...")
        
        for modality in ["image", "audio", "video"]:
            preprocessing_dir = self.paths[f"preprocessing_{modality}"]
            metadata_file = preprocessing_dir / "metadata.csv"
            
            if not metadata_file.exists():
                continue
            
            try:
                df = pd.read_csv(metadata_file)
                metadata_samples = len(df)
                
                # Check if processed files match metadata
                if modality == "image":
                    crops_dir = preprocessing_dir / "crops"
                    normalized_dir = preprocessing_dir / "normalized"
                    
                    if crops_dir.exists():
                        crop_files = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
                        crop_count = len(crop_files)
                        
                        if abs(crop_count - metadata_samples) <= 1:  # Allow for header
                            self._add_result(
                                f"{modality}_crops_match_metadata",
                                ReadinessStatus.PASS,
                                f"{modality} crop files match metadata",
                                {"crop_count": crop_count, "metadata_count": metadata_samples}
                            )
                        else:
                            self._add_result(
                                f"{modality}_crops_match_metadata",
                                ReadinessStatus.WARNING,
                                f"{modality} crop count mismatch: {crop_count} vs {metadata_samples}",
                                {"crop_count": crop_count, "metadata_count": metadata_samples}
                            )
                
            except Exception as e:
                self._add_result(
                    f"{modality}_preprocessing_consistency",
                    ReadinessStatus.WARNING,
                    f"Cannot verify {modality} preprocessing consistency: {e}",
                    {"error": str(e)}
                )
    
    def _check_generator_distributions(self):
        """Check that generator distributions are valid."""
        print("\n[CHECK] Checking generator distributions...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            
            if not splits_dir.exists():
                continue
            
            for split_name in ["train.csv", "val.csv", "test_seen.csv"]:
                split_path = splits_dir / split_name
                if not split_path.exists():
                    continue
                
                try:
                    df = pd.read_csv(split_path)
                    
                    # Check for generator column
                    if 'generator' in df.columns:
                        generators = df['generator'].unique()
                        generator_count = len(generators)
                        
                        if generator_count >= 2:
                            self._add_result(
                                f"{modality}_{split_name.replace('.csv', '')}_generator_diversity",
                                ReadinessStatus.PASS,
                                f"{modality} {split_name} has {generator_count} generators",
                                {"generators": list(generators), "count": generator_count}
                            )
                        else:
                            self._add_result(
                                f"{modality}_{split_name.replace('.csv', '')}_generator_diversity",
                                ReadinessStatus.WARNING,
                                f"{modality} {split_name} has only {generator_count} generator(s)",
                                {"generators": list(generators), "count": generator_count}
                            )
                    else:
                        self._add_result(
                            f"{modality}_{split_name.replace('.csv', '')}_generator_column",
                            ReadinessStatus.WARNING,
                            f"{modality} {split_name} missing generator column",
                            {}
                        )
                except Exception as e:
                    self._add_result(
                        f"{modality}_{split_name.replace('.csv', '')}_generator_check",
                        ReadinessStatus.WARNING,
                        f"Cannot check {modality} {split_name} generators: {e}",
                        {"error": str(e)}
                    )
    
    def _check_identity_leakage(self):
        """Check that train/validation/test identities do not overlap."""
        print("\n[CHECK] Checking identity leakage...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            
            if not splits_dir.exists():
                continue
            
            try:
                # Load all splits
                splits_data = {}
                for split_name in ["train.csv", "val.csv", "test_seen.csv", "test_unseen.csv"]:
                    split_path = splits_dir / split_name
                    if split_path.exists():
                        df = pd.read_csv(split_path)
                        splits_data[split_name.replace('.csv', '')] = df
                
                # Check for identity column
                if not splits_data:
                    continue
                
                first_df = list(splits_data.values())[0]
                if 'identity_key' not in first_df.columns and 'identity_id' not in first_df.columns:
                    self._add_result(
                        f"{modality}_identity_column_available",
                        ReadinessStatus.WARNING,
                        f"{modality} splits missing identity column",
                        {}
                    )
                    continue
                
                identity_col = 'identity_key' if 'identity_key' in first_df.columns else 'identity_id'
                
                # Check for overlap
                all_identities = {}
                for split_name, df in splits_data.items():
                    identities = set(df[identity_col].dropna().unique())
                    all_identities[split_name] = identities
                
                # Check pairwise overlaps
                overlap_found = False
                for split1 in all_identities:
                    for split2 in all_identities:
                        if split1 >= split2:  # Avoid duplicate checks
                            continue
                        
                        overlap = all_identities[split1] & all_identities[split2]
                        if overlap:
                            overlap_found = True
                            self._add_result(
                                f"{modality}_identity_leakage_{split1}_{split2}",
                                ReadinessStatus.FAIL,
                                f"{modality} identity overlap between {split1} and {split2}: {len(overlap)} subjects",
                                {"overlap_count": len(overlap)},
                                critical=True
                            )
                
                if not overlap_found and all_identities:
                    self._add_result(
                        f"{modality}_identity_leakage",
                        ReadinessStatus.PASS,
                        f"{modality} no identity leakage detected",
                        {}
                    )
                
            except Exception as e:
                self._add_result(
                    f"{modality}_identity_leakage_check",
                    ReadinessStatus.WARNING,
                    f"Cannot check {modality} identity leakage: {e}",
                    {"error": str(e)}
                )
    
    def _check_generator_leakage(self):
        """Check that unseen generators do not appear in training."""
        print("\n[CHECK] Checking generator leakage...")
        
        for modality in ["image", "audio", "video"]:
            splits_dir = self.paths[f"splits_{modality}"]
            
            if not splits_dir.exists():
                continue
            
            try:
                # Load train and test_unseen
                train_path = splits_dir / "train.csv"
                test_unseen_path = splits_dir / "test_unseen.csv"
                
                if not train_path.exists() or not test_unseen_path.exists():
                    continue
                
                train_df = pd.read_csv(train_path)
                test_unseen_df = pd.read_csv(test_unseen_path)
                
                # Check for generator column
                if 'generator' not in train_df.columns or 'generator' not in test_unseen_df.columns:
                    self._add_result(
                        f"{modality}_generator_leakage_check",
                        ReadinessStatus.WARNING,
                        f"{modality} missing generator column",
                        {}
                    )
                    continue
                
                train_generators = set(train_df['generator'].dropna().unique())
                test_unseen_generators = set(test_unseen_df['generator'].dropna().unique())
                
                # Check for overlap
                overlap = train_generators & test_unseen_generators
                if overlap:
                    self._add_result(
                        f"{modality}_generator_leakage",
                        ReadinessStatus.FAIL,
                        f"{modality} generators in train also in test_unseen: {overlap}",
                        {"leaking_generators": list(overlap)},
                        critical=True
                    )
                else:
                    if test_unseen_generators:
                        self._add_result(
                            f"{modality}_generator_leakage",
                            ReadinessStatus.PASS,
                            f"{modality} no generator leakage detected",
                            {"train_generators": len(train_generators), "unseen_generators": len(test_unseen_generators)}
                        )
                
            except Exception as e:
                self._add_result(
                    f"{modality}_generator_leakage_check",
                    ReadinessStatus.WARNING,
                    f"Cannot check {modality} generator leakage: {e}",
                    {"error": str(e)}
                )
    
    def _check_model_checkpoints(self):
        """Check that model checkpoints exist before evaluation is attempted."""
        print("\n[CHECK] Checking model checkpoints...")
        
        for modality in ["image", "audio", "video"]:
            models_dir = self.paths[f"models_{modality}"]
            checkpoint_path = models_dir / "baseline_best.pt"
            
            if checkpoint_path.exists():
                file_size = checkpoint_path.stat().st_size
                self._add_result(
                    f"{modality}_checkpoint_exists",
                    ReadinessStatus.PASS,
                    f"{modality} checkpoint exists ({file_size / 1024 / 1024:.1f} MB)",
                    {"path": str(checkpoint_path), "size_bytes": file_size}
                )
            else:
                self._add_result(
                    f"{modality}_checkpoint_exists",
                    ReadinessStatus.FAIL,
                    f"{modality} checkpoint missing",
                    {"expected_path": str(checkpoint_path)},
                    critical=(modality == "image")  # Image has data, so checkpoint is critical
                )
    
    def _check_calibration_files(self):
        """Check that calibration files correspond to the correct checkpoint."""
        print("\n[CHECK] Checking calibration files...")
        
        for modality in ["image", "audio", "video"]:
            models_dir = self.paths[f"models_{modality}"]
            calibration_path = models_dir / "calibration.json"
            checkpoint_path = models_dir / "baseline_best.pt"
            
            if not checkpoint_path.exists():
                continue  # Skip calibration check if no checkpoint
            
            if calibration_path.exists():
                try:
                    with open(calibration_path, 'r') as f:
                        calib_data = json.load(f)
                    
                    # Check if calibration references correct checkpoint
                    if 'checkpoint_path' in calib_data:
                        if checkpoint_path in calib_data['checkpoint_path']:
                            self._add_result(
                                f"{modality}_calibration_match",
                                ReadinessStatus.PASS,
                                f"{modality} calibration matches checkpoint",
                                {}
                            )
                        else:
                            self._add_result(
                                f"{modality}_calibration_match",
                                ReadinessStatus.WARNING,
                                f"{modality} calibration references different checkpoint",
                                {"calibration_checkpoint": calib_data['checkpoint_path']}
                            )
                    else:
                        self._add_result(
                            f"{modality}_calibration_complete",
                            ReadinessStatus.WARNING,
                            f"{modality} calibration missing checkpoint reference",
                            {}
                        )
                except Exception as e:
                    self._add_result(
                        f"{modality}_calibration_readable",
                        ReadinessStatus.FAIL,
                        f"Cannot read {modality} calibration: {e}",
                        {"error": str(e)}
                    )
            else:
                self._add_result(
                    f"{modality}_calibration_exists",
                    ReadinessStatus.WARNING,
                    f"{modality} calibration file missing",
                    {"expected_path": str(calibration_path)}
                )
    
    def _check_evaluation_results(self):
        """Check that evaluation results correspond to the correct dataset split."""
        print("\n[CHECK] Checking evaluation results...")
        
        for modality in ["image", "audio", "video"]:
            results_dir = self.paths["results"] / modality
            generalization_results = results_dir / "generalization_results.json"
            
            if not generalization_results.exists():
                if modality == "image":
                    self._add_result(
                        f"{modality}_evaluation_results_exist",
                        ReadinessStatus.WARNING,
                        f"{modality} evaluation results missing",
                        {"expected_path": str(generalization_results)}
                    )
                continue
            
            try:
                with open(generalization_results, 'r') as f:
                    results_data = json.load(f)
                
                # Check if results reference correct splits
                if 'splits' in results_data:
                    splits_checked = 0
                    for split_name in ['val', 'test_seen', 'test_unseen']:
                        if split_name in results_data['splits']:
                            splits_checked += 1
                    
                    if splits_checked > 0:
                        self._add_result(
                            f"{modality}_evaluation_results_complete",
                            ReadinessStatus.PASS,
                            f"{modality} evaluation results cover {splits_checked} splits",
                            {"splits_checked": splits_checked}
                        )
                
                # Check test_unseen support
                if 'splits' in results_data and 'test_unseen' in results_data['splits']:
                    test_unseen_support = results_data['splits']['test_unseen'].get('support', 0)
                    if test_unseen_support == 0:
                        self._add_result(
                            f"{modality}_test_unseen_evaluated",
                            ReadinessStatus.FAIL,
                            f"{modality} test_unseen has zero support in evaluation",
                            {"support": test_unseen_support},
                            critical=True
                        )
                    else:
                        self._add_result(
                            f"{modality}_test_unseen_evaluated",
                            ReadinessStatus.PASS,
                            f"{modality} test_unseen evaluated with {test_unseen_support} samples",
                            {"support": test_unseen_support}
                        )
                
            except Exception as e:
                self._add_result(
                    f"{modality}_evaluation_results_readable",
                    ReadinessStatus.FAIL,
                    f"Cannot read {modality} evaluation results: {e}",
                    {"error": str(e)}
                )
    
    def _check_artifact_consistency(self):
        """Check overall consistency between artifacts."""
        print("\n[CHECK] Checking artifact consistency...")
        
        # Check that results directory structure is consistent
        for modality in ["image", "audio", "video"]:
            results_dir = self.paths["results"] / modality
            
            if not results_dir.exists():
                continue
            
            expected_files = [
                "generalization_results.json",
                "split_metrics.csv",
                "experiment_config.csv"
            ]
            
            files_present = 0
            for expected_file in expected_files:
                if (results_dir / expected_file).exists():
                    files_present += 1
            
            if files_present == len(expected_files):
                self._add_result(
                    f"{modality}_results_complete",
                    ReadinessStatus.PASS,
                    f"{modality} results directory complete",
                    {"files_present": files_present}
                )
            elif files_present > 0:
                self._add_result(
                    f"{modality}_results_complete",
                    ReadinessStatus.WARNING,
                    f"{modality} results directory partial ({files_present}/{len(expected_files)} files)",
                    {"files_present": files_present, "expected": len(expected_files)}
                )
    
    def _check_experiment_configuration(self):
        """Check that experiment configuration is reproducible."""
        print("\n[CHECK] Checking experiment configuration...")
        
        # Check for config files
        configs_dir = self.paths["configs"]
        expected_configs = [
            "image_baseline.yaml",
            "audio_baseline.yaml", 
            "video_baseline.yaml"
        ]
        
        for config_file in expected_configs:
            config_path = configs_dir / config_file
            if config_path.exists():
                self._add_result(
                    f"config_{config_file.replace('.yaml', '')}_exists",
                    ReadinessStatus.PASS,
                    f"Configuration file {config_file} exists",
                    {"path": str(config_path)}
                )
            else:
                self._add_result(
                    f"config_{config_file.replace('.yaml', '')}_exists",
                    ReadinessStatus.WARNING,
                    f"Configuration file {config_file} missing",
                    {"expected_path": str(config_path)}
                )
    
    def _check_random_seeds(self):
        """Check that random seeds are recorded."""
        print("\n[CHECK] Checking random seeds...")
        
        for modality in ["image", "audio", "video"]:
            results_dir = self.paths["results"] / modality
            generalization_results = results_dir / "generalization_results.json"
            
            if not generalization_results.exists():
                continue
            
            try:
                with open(generalization_results, 'r') as f:
                    results_data = json.load(f)
                
                if 'random_seed' in results_data:
                    seed = results_data['random_seed']
                    self._add_result(
                        f"{modality}_random_seed_recorded",
                        ReadinessStatus.PASS,
                        f"{modality} random seed recorded: {seed}",
                        {"seed": seed}
                    )
                else:
                    self._add_result(
                        f"{modality}_random_seed_recorded",
                        ReadinessStatus.WARNING,
                        f"{modality} random seed not recorded in results",
                        {}
                    )
                
            except Exception as e:
                self._add_result(
                    f"{modality}_random_seed_check",
                    ReadinessStatus.WARNING,
                    f"Cannot check {modality} random seed: {e}",
                    {"error": str(e)}
                )
    
    def _check_mlflow_experiments(self):
        """Check MLflow experiment tracking status."""
        print("\n[CHECK] Checking MLflow experiments...")
        
        mlruns_dir = self.paths["mlruns"]
        
        if mlruns_dir.exists():
            # Count experiments
            experiment_count = 0
            if (mlruns_dir / ".metadata").exists():
                try:
                    with open(mlruns_dir / ".metadata", 'r') as f:
                        metadata = json.load(f)
                        experiment_count = len(metadata.get('experiments', []))
                except:
                    pass
            
            if experiment_count > 0:
                self._add_result(
                    "mlflow_active",
                    ReadinessStatus.PASS,
                    f"MLflow active with {experiment_count} experiment(s)",
                    {"experiment_count": experiment_count}
                )
            else:
                self._add_result(
                    "mlflow_active",
                    ReadinessStatus.WARNING,
                    "MLflow directory exists but no experiments found",
                    {}
                )
        else:
            self._add_result(
                "mlflow_active",
                ReadinessStatus.WARNING,
                "MLflow not active (no mlruns directory)",
                {}
            )
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive readiness report."""
        print("\n" + "=" * 60)
        print("Generating Readiness Report...")
        
        # Count results by status
        pass_count = sum(1 for r in self.results if r.status == ReadinessStatus.PASS)
        warning_count = sum(1 for r in self.results if r.status == ReadinessStatus.WARNING)
        fail_count = sum(1 for r in self.results if r.status == ReadinessStatus.FAIL)
        critical_fail_count = sum(1 for r in self.results if r.critical and r.status == ReadinessStatus.FAIL)
        
        # Determine overall status
        if critical_fail_count > 0:
            overall_status = ReadinessStatus.FAIL
            overall_message = f"CRITICAL: {critical_fail_count} critical failure(s) blocking scientific validity"
        elif fail_count > 0:
            overall_status = ReadinessStatus.FAIL
            overall_message = f"FAIL: {fail_count} failure(s) need resolution"
        elif warning_count > 0:
            overall_status = ReadinessStatus.WARNING
            overall_message = f"WARNING: {warning_count} warning(s) should be addressed"
        else:
            overall_status = ReadinessStatus.PASS
            overall_message = "PASS: All checks passed"
        
        # Build report
        report = {
            "overall_status": overall_status.value,
            "overall_message": overall_message,
            "summary": {
                "total_checks": len(self.results),
                "pass": pass_count,
                "warning": warning_count,
                "fail": fail_count,
                "critical_failures": critical_fail_count
            },
            "critical_failures": [
                {
                    "check": r.check_name,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results if r.critical and r.status == ReadinessStatus.FAIL
            ],
            "all_results": [
                {
                    "check": r.check_name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "critical": r.critical
                }
                for r in self.results
            ],
            "timestamp": pd.Timestamp.now().isoformat(),
            "project_root": str(self.project_root)
        }
        
        # Print summary
        print(f"\n[SUMMARY] READINESS SUMMARY:")
        print(f"   Total Checks: {len(self.results)}")
        print(f"   PASS: {pass_count}")
        print(f"   WARNING: {warning_count}")
        print(f"   FAIL: {fail_count}")
        print(f"   CRITICAL: {critical_fail_count}")
        print(f"\n   Overall Status: {overall_status.value}")
        print(f"   {overall_message}")
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = None):
        """Save readiness report to JSON file."""
        if output_path is None:
            output_path = self.project_root / "reports" / "project_readiness.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[SAVE] Report saved to: {output_path}")
        
        # Exit with error code if critical failures
        if report["summary"]["critical_failures"] > 0:
            print("\n[CRITICAL] CRITICAL FAILURES DETECTED - Project not scientifically ready")
            sys.exit(1)
        elif report["summary"]["fail"] > 0:
            print("\n[WARNING] FAILURES DETECTED - Address failures before proceeding")
            sys.exit(1)
        else:
            print("\n[PASS] Project readiness checks passed")
            sys.exit(0)


def main():
    """Main entry point for project readiness checking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEGIS Project Readiness Verification")
    parser.add_argument("--project-root", type=str, help="Path to AEGIS project root")
    parser.add_argument("--output", type=str, help="Output path for readiness report")
    parser.add_argument("--fail-on-warning", action="store_true", 
                       help="Treat warnings as failures")
    
    args = parser.parse_args()
    
    try:
        checker = ProjectReadinessChecker(args.project_root)
        report = checker.run_all_checks()
        checker.save_report(report, args.output)
    except Exception as e:
        print(f"[ERROR] Error running readiness checks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()