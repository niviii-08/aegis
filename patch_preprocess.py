import re
from pathlib import Path

src = Path(r'c:\Users\Neevetha N\Downloads\AEGIS\src\image\preprocessing\preprocess.py')
content = src.read_text(encoding='utf-8')

# 1. Imports
content = content.replace(
    'import tempfile\nimport time\n',
    'import tempfile\nimport time\nimport shutil\nimport os\n'
)

# 2. Config fields
config_fields = """    config_path: Path
    preprocessing_version: str = "image_facecrop_v1"
    preprocessing_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_version: str = "1.0.0"
    configuration_hash: str = ""
    registry_path: Path = None
    dry_run: bool = False
    batch_size: int = 1000"""
content = re.sub(r'    config_path: Path\n    preprocessing_version.*?configuration_hash: str = ""', config_fields, content, flags=re.DOTALL)

# 3. Config initialization
config_init = """        config_path=config_path.resolve(),
        preprocessing_version="image_facecrop_v1",
        preprocessing_timestamp=datetime.now(timezone.utc).isoformat(),
        code_version="1.0.0",
        configuration_hash=config_hash_str,
        registry_path=(root / "data/processed/image/sample_registry.csv").resolve(),
    )"""
content = re.sub(r'        config_path=config_path\.resolve\(\),\n        preprocessing_version.*?configuration_hash=config_hash_str,\n    \)', config_init, content, flags=re.DOTALL)

# 4. write_outputs (atomic tmp)
old_write = """    if config.save_crop_jpeg:
        crop_path = config.crops_dir / f"{basename}.jpg"
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]):
            raise OSError(f"Failed to write crop JPEG: {crop_path}")
        crop_rel = relative_project_path(crop_path, config.project_root)

    if config.save_normalized_npy:
        norm_path = config.normalized_dir / f"{basename}.npy"
        np.save(norm_path, normalized_chw)
        norm_rel = relative_project_path(norm_path, config.project_root)"""
        
new_write = """    if config.save_crop_jpeg:
        crop_path = config.crops_dir / f"{basename}.jpg"
        crop_tmp = crop_path.with_suffix('.jpg.tmp')
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_tmp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]):
            raise OSError(f"Failed to write crop JPEG: {crop_path}")
        os.replace(str(crop_tmp), str(crop_path))
        crop_rel = relative_project_path(crop_path, config.project_root)

    if config.save_normalized_npy:
        norm_path = config.normalized_dir / f"{basename}.npy"
        norm_tmp = norm_path.with_suffix('.npy.tmp')
        np.save(norm_tmp, normalized_chw)
        os.replace(str(norm_tmp), str(norm_path))
        norm_rel = relative_project_path(norm_path, config.project_root)"""
content = content.replace(old_write, new_write)

# 5. run_preprocessing (driver)
new_run_preprocessing = '''
def run_preprocessing(config: PreprocessConfig) -> PreprocessSummary:
    import pandas as pd
    import shutil
    run_started = time.perf_counter()
    
    # Load registry
    registry_df = pd.read_csv(config.registry_path, low_memory=False)
    total_samples = len(registry_df)
    
    # Find already processed
    already_processed = registry_df[registry_df["status"] == "PROCESSED"]
    num_processed = len(already_processed)
    
    # Filter to eligible
    if config.retry_failures:
        eligible_df = registry_df[registry_df["status"].isin(["RAW", "FAILED"])]
    else:
        eligible_df = registry_df[registry_df["status"] == "RAW"]
        
    eligible_rows = eligible_df.to_dict('records')
    
    if config.max_images is not None:
        eligible_rows = eligible_rows[: config.max_images]

    num_eligible = len(eligible_rows)
    
    # Resource checking
    estimated_bytes = num_eligible * (20*1024 + 602*1024)  # approx 622KB per sample
    free_bytes = shutil.disk_usage(config.crops_dir.parent if config.crops_dir.exists() else config.project_root).free
    
    if config.dry_run:
        logger.info("================ DRY RUN MODE ================")
        logger.info(f"Total registry samples: {total_samples:,}")
        logger.info(f"Already PROCESSED:      {num_processed:,}")
        logger.info(f"Eligible to process:    {num_eligible:,}")
        logger.info(f"Estimated storage req:  {estimated_bytes/1e9:.2f} GB")
        logger.info(f"Available free space:   {free_bytes/1e9:.2f} GB")
        logger.info("==============================================")
        if estimated_bytes > free_bytes:
            logger.error("INSUFFICIENT DISK SPACE FOR FULL RUN")
        return PreprocessSummary(
            preprocessing_version=config.preprocessing_version,
            config_path=str(config.config_path),
            manifest_path=str(config.registry_path),
            detector=config.detector_name,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            total_images=num_eligible
        )
        
    if estimated_bytes > free_bytes:
        raise OSError(f"Insufficient disk space. Need {estimated_bytes/1e9:.2f}GB, have {free_bytes/1e9:.2f}GB")

    existing = load_existing_metadata(config.metadata_path)
    detector = create_face_detector(
        config.detector_name,
        min_face_size=config.min_face_size,
        min_confidence=config.min_confidence,
    )
    cropper = FaceCropper(
        crop_config=CropConfig(
            margin_factor=config.margin_factor,
            output_size=config.output_size,
            align_faces=config.align_faces,
        ),
        normalization=config.normalization,
    )

    summary = PreprocessSummary(
        preprocessing_version=config.preprocessing_version,
        config_path=str(config.config_path),
        manifest_path=str(config.registry_path),
        detector=config.detector_name,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_images=num_eligible,
        preprocessing_version_tag=config.preprocessing_version,
        code_version=config.code_version,
        configuration_hash=config.configuration_hash,
    )

    merged_metadata: dict[str, dict[str, str]] = dict(existing)
    face_areas: list[int] = []

    checkpoint_interval = config.batch_size
    
    # We will update registry_df in place and flush it
    registry_idx_map = {row["sample_id"]: idx for idx, row in registry_df.iterrows()}

    def flush_checkpoint():
        ordered_rows = [merged_metadata[sid] for sid in merged_metadata]
        write_csv_atomic(ordered_rows, config.metadata_path, METADATA_COLUMNS)
        write_failures_csv(ordered_rows, config.failures_path)
        
        # Flush registry atomically
        registry_out = config.registry_path.with_suffix('.csv.tmp')
        registry_df.to_csv(registry_out, index=False)
        os.replace(str(registry_out), str(config.registry_path))

    for index, row in enumerate(eligible_rows, start=1):
        sample_id = row["sample_id"]
        
        # Fake manifest row for process_sample
        manifest_row = {"sample_id": sample_id, "path": row["raw_path"]}
        
        outcome = process_sample(manifest_row, config=config, detector=detector, cropper=cropper)
        merged_metadata[sample_id] = outcome.metadata_row
        summary.processed_this_run += 1

        reg_idx = registry_idx_map[sample_id]
        if outcome.is_success:
            summary.successful_crops += 1
            if outcome.face_area is not None:
                face_areas.append(outcome.face_area)
                
            # Compute hash of crop
            crop_path = config.project_root / outcome.metadata_row["processed_crop_path"]
            h = hashlib.sha256()
            with open(crop_path, 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            
            # Update registry
            registry_df.at[reg_idx, "status"] = "PROCESSED"
            registry_df.at[reg_idx, "crop_path"] = outcome.metadata_row["processed_crop_path"]
            registry_df.at[reg_idx, "processed_path"] = outcome.metadata_row["processed_normalized_path"]
            registry_df.at[reg_idx, "file_hash"] = h.hexdigest()
            registry_df.at[reg_idx, "preprocessing_version"] = config.preprocessing_version
            registry_df.at[reg_idx, "preprocessing_timestamp"] = outcome.metadata_row["processed_at"]
            registry_df.at[reg_idx, "code_version"] = config.code_version
            registry_df.at[reg_idx, "configuration_hash"] = config.configuration_hash
            registry_df.at[reg_idx, "failure_reason"] = ""
        else:
            summary.failed_crops += 1
            registry_df.at[reg_idx, "status"] = "FAILED"
            registry_df.at[reg_idx, "failure_reason"] = outcome.metadata_row.get("status", "unknown")
            
        if outcome.is_no_face:
            summary.no_face_cases += 1
        if outcome.is_multi_face:
            summary.multi_face_cases += 1

        if index % config.log_every == 0 or index == num_eligible:
            elapsed = time.perf_counter() - run_started
            throughput = index / elapsed if elapsed > 0 else 0
            remaining = num_eligible - index
            eta = remaining / throughput if throughput > 0 else 0
            logger.info(
                "Progress %s/%s | success=%s failed=%s | throughput=%.1f img/s eta=%.1fs (%.1f%%)",
                index,
                num_eligible,
                summary.successful_crops,
                summary.failed_crops,
                throughput,
                eta,
                (index / num_eligible) * 100
            )

        # Periodic checkpoint
        if index % checkpoint_interval == 0 or index == num_eligible:
            flush_checkpoint()

    flush_checkpoint()

    summary.successful_crops = sum(1 for row in merged_metadata.values() if row.get("status") == SUCCESS_STATUS)
    summary.failed_crops = sum(1 for row in merged_metadata.values() if row.get("status") != SUCCESS_STATUS)
    summary.no_face_cases = sum(1 for row in merged_metadata.values() if row.get("status") == "no_face")
    summary.multi_face_cases = sum(
        1 for row in merged_metadata.values() if row.get("status") == SUCCESS_STATUS and int(row.get("face_count") or 0) > 1
    )

    all_face_areas = [
        int(row["bbox_w"]) * int(row["bbox_h"])
        for row in merged_metadata.values()
        if row.get("status") == SUCCESS_STATUS and row.get("bbox_w") and row.get("bbox_h")
    ]
    if all_face_areas:
        summary.average_face_size_pixels = float(sum(all_face_areas) / len(all_face_areas))
    elif face_areas:
        summary.average_face_size_pixels = float(sum(face_areas) / len(face_areas))
    summary.total_processing_time_seconds = time.perf_counter() - run_started

    if summary.failed_crops:
        summary.notes.append(f"Recorded {summary.failed_crops} failures in {config.failures_path.name}.")
    else:
        summary.notes.append("No preprocessing failures in this run.")

    write_summary_json(summary, config.summary_path)
    return summary
'''
content = re.sub(r'def run_preprocessing\(config: PreprocessConfig\) -> PreprocessSummary:.*?def parse_args', new_run_preprocessing + '\n\ndef parse_args', content, flags=re.DOTALL)

# Add CLI arguments
new_cli = '''    parser.add_argument("--dry-run", action="store_true", help="Report only, do not process.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Registry flush interval.")
    return parser.parse_args(argv)'''
content = content.replace('    return parser.parse_args(argv)', new_cli)

new_main = '''    if args.limit is not None:
        config.max_images = args.limit
        logger.warning("DEBUG MODE: limiting to %d samples", args.limit)
        
    config.dry_run = args.dry_run
    config.batch_size = args.batch_size'''
content = content.replace('''    if args.limit is not None:
        config.max_images = args.limit
        logger.warning("DEBUG MODE: limiting to %d samples", args.limit)''', new_main)

src.write_text(content, encoding='utf-8')
print("Successfully patched preprocess.py")
