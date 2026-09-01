import re
from pathlib import Path

# Paths
image_dir = Path('src/image/training')
audio_dir = Path('src/audio/training')

# 1. Evaluate.py
eval_txt = (image_dir / 'evaluate.py').read_text(encoding='utf-8')
eval_txt = eval_txt.replace('image.data_audit', 'audio.data_audit')
eval_txt = eval_txt.replace('image.models.factory', 'audio.models.factory')
eval_txt = eval_txt.replace('image.training.', 'audio.training.')
eval_txt = eval_txt.replace('image', 'audio')
eval_txt = eval_txt.replace('FaceCropDataset', 'AudioDataset')
eval_txt = eval_txt.replace('input_source=config.input_source', 'feature_mode=config.feature_mode')
eval_txt = eval_txt.replace('for batch_x, batch_y, batch_ids in loader:', 'for batch_x, batch_y in loader:')
eval_txt = eval_txt.replace('sample_ids.extend(batch_ids)', '')
eval_txt = eval_txt.replace('AudioDataset(samples, feature_mode=config.feature_mode)', 'AudioDataset(samples, feature_mode=config.feature_mode, is_training=False)')
eval_txt = eval_txt.replace('metrics["sample_ids"] = sample_ids', '')
eval_txt = eval_txt.replace('sample_ids: list[str] = []', '')

(audio_dir / 'evaluate.py').write_text(eval_txt, encoding='utf-8')

# 2. Train.py
train_txt = (image_dir / 'train.py').read_text(encoding='utf-8')
train_txt = train_txt.replace('image.data_audit', 'audio.data_audit')
train_txt = train_txt.replace('image.models.factory', 'audio.models.factory')
train_txt = train_txt.replace('image.training.', 'audio.training.')
train_txt = train_txt.replace('image', 'audio')
train_txt = train_txt.replace('FaceCropDataset', 'AudioDataset')

train_txt = train_txt.replace('input_source=config.input_source,  # type: ignore[arg-type]', 'feature_mode=config.feature_mode,')

train_txt = train_txt.replace(
    'AudioDataset(train_samples, feature_mode=config.feature_mode,)',
    'AudioDataset(train_samples, feature_mode=config.feature_mode, is_training=True),'
)
train_txt = train_txt.replace(
    'AudioDataset(val_samples, feature_mode=config.feature_mode,)',
    'AudioDataset(val_samples, feature_mode=config.feature_mode, is_training=False),'
)
# Note: In train.py it was: FaceCropDataset(train_samples, input_source=config.input_source),  # type: ignore[arg-type]
# So after first replaces it became: AudioDataset(train_samples, feature_mode=config.feature_mode,)
# Wait, let's just do regex or safer string replace.
train_txt = re.sub(r'AudioDataset\(train_samples, feature_mode=config\.feature_mode\),.*', 'AudioDataset(train_samples, feature_mode=config.feature_mode, is_training=True),', train_txt)
train_txt = re.sub(r'AudioDataset\(val_samples, feature_mode=config\.feature_mode\),.*', 'AudioDataset(val_samples, feature_mode=config.feature_mode, is_training=False),', train_txt)

train_txt = train_txt.replace('for batch_x, batch_y, _sample_ids in loader:', 'for batch_x, batch_y in loader:')

(audio_dir / 'train.py').write_text(train_txt, encoding='utf-8')

# 3. Utils.py
utils_txt = (image_dir / 'utils.py').read_text(encoding='utf-8')
utils_txt = utils_txt.replace('image.data_audit', 'audio.data_audit')
utils_txt = utils_txt.replace('image.models.baseline', 'audio.models.baseline')
utils_txt = utils_txt.replace('image_baseline', 'audio_baseline')
utils_txt = utils_txt.replace('models/image', 'models/audio')
utils_txt = utils_txt.replace('reports/image', 'reports/audio')
utils_txt = utils_txt.replace('aegis-image-baseline', 'aegis-audio-baseline')
utils_txt = utils_txt.replace('data/processed/image/splits', 'data/processed/audio/splits')

utils_txt = utils_txt.replace('input_source: str', 'feature_mode: str')
utils_txt = utils_txt.replace('input_source=str(data_cfg.get("input_source", "normalized_npy"))', 'feature_mode=str(data_cfg.get("feature_mode", "wav2vec2"))')
utils_txt = utils_txt.replace('"input_source": config.input_source,', '"feature_mode": config.feature_mode,')

utils_txt = re.sub(
    r'metadata_path=\(root / split_cfg\.get\("metadata_path", "[^"]+"\)\)\.resolve\(\),',
    'metadata_path=(root / "reports" / "audio" / f"preprocessing_metadata_{str(data_cfg.get(\'feature_mode\', \'wav2vec2\'))}.csv").resolve(),',
    utils_txt
)

(audio_dir / 'utils.py').write_text(utils_txt, encoding='utf-8')

print("Success")
