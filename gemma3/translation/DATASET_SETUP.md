# NLLB-200 Local Dataset Setup

## Overview

To avoid connection issues and slow streaming when training, you can download the NLLB-200 dataset locally. This downloads 1M translation pairs for English↔Hebrew, English↔Sanskrit, and English↔Tibetan.

## Quick Start

### 1. Download the Dataset

```bash
cd /home/orrz/gpufs/projects/gemma3/translation

# Download 1M samples per language pair (default)
python download_nllb_data.py

# Or specify custom number of samples
python download_nllb_data.py --samples 500000
```

**This will take 30-60 minutes** depending on your connection speed.

### 2. Output Structure

Data will be saved to `/home/orrz/gpufs/projects/gemma3/data/nllb_translation/`:

```
nllb_translation/
├── eng_heb.jsonl          # English ↔ Hebrew (1M pairs, ~400 MB)
├── eng_san.jsonl          # English ↔ Sanskrit (1M pairs, ~400 MB)
├── eng_bod.jsonl          # English ↔ Tibetan (1M pairs, ~400 MB)
├── all_pairs.jsonl        # Combined file (3M pairs, ~1.2 GB)
└── dataset_info.json      # Metadata
```

### 3. Use in Training

The config is already set up to use local dataset:

```yaml
translation:
  local_dataset_path: "/home/orrz/gpufs/projects/gemma3/data/nllb_translation/all_pairs.jsonl"
```

Now when you run training, it will **load from local file** instead of streaming:

```bash
python train_translation.py --config configs/train_config.yaml
```

## Language Pairs Included

The download script automatically downloads these pairs:

| Source | Target | NLLB Codes | File |
|--------|--------|------------|------|
| English | Hebrew | eng_Latn → heb_Hebr | eng_heb.jsonl |
| English | Sanskrit | eng_Latn → san_Deva | eng_san.jsonl |
| English | Tibetan | eng_Latn → bod_Tibt | eng_bod.jsonl |

## JSONL Format

Each line in the JSONL files contains:

```json
{
  "source": "Hello, how are you?",
  "target": "שלום, מה שלומך?",
  "source_lang": "English",
  "target_lang": "Hebrew",
  "source_code": "eng_Latn",
  "target_code": "heb_Hebr"
}
```

## Advanced Usage

### Download Different Number of Samples

```bash
# Download 100K samples per pair (faster, for testing)
python download_nllb_data.py --samples 100000

# Download 2M samples per pair (more data, better quality)
python download_nllb_data.py --samples 2000000
```

### Custom Output Directory

```bash
python download_nllb_data.py --output-dir /path/to/custom/location
```

Then update your config:
```yaml
translation:
  local_dataset_path: "/path/to/custom/location/all_pairs.jsonl"
```

### Use Specific Language Pair File

If you only need one language pair, point to specific file:

```yaml
translation:
  source_lang: "English"
  target_lang: "Hebrew"
  local_dataset_path: "/home/orrz/gpufs/projects/gemma3/data/nllb_translation/eng_heb.jsonl"
```

## Fallback to Streaming

If local dataset is not available, the training code automatically falls back to HuggingFace streaming:

```yaml
translation:
  # Comment out or remove to use streaming
  # local_dataset_path: "/home/orrz/gpufs/projects/gemma3/data/nllb_translation/all_pairs.jsonl"
```

## Benefits of Local Dataset

✅ **Fast loading**: No waiting for downloads during training
✅ **Reliable**: No connection issues or timeouts
✅ **Reproducible**: Same exact samples every time with same seed
✅ **Offline**: Train without internet connection
✅ **Efficient**: Load only what you need from local file

## Troubleshooting

### Download Fails

If download fails midway, just run the script again. It will restart the download.

### File Size Too Large

If disk space is limited, download fewer samples:

```bash
python download_nllb_data.py --samples 100000
```

100K samples per pair ≈ 40 MB per file.

### Wrong Language Pair

Edit `download_nllb_data.py` to add/remove language pairs in the `LANGUAGE_PAIRS` list:

```python
LANGUAGE_PAIRS = [
    {
        "source_lang": "English",
        "target_lang": "Hebrew",
        "source_code": "eng_Latn",
        "target_code": "heb_Hebr",
        "output_file": "eng_heb.jsonl"
    },
    # Add more pairs here...
]
```

### Check Dataset Info

View dataset metadata:

```bash
cat /home/orrz/gpufs/projects/gemma3/data/nllb_translation/dataset_info.json
```

## Dataset Statistics

After downloading 1M samples per pair:

- **English-Hebrew**: ~400 MB, 1,000,000 pairs
- **English-Sanskrit**: ~400 MB, 1,000,000 pairs
- **English-Tibetan**: ~400 MB, 1,000,000 pairs
- **Combined**: ~1.2 GB, 3,000,000 pairs

## Next Steps

After downloading the local dataset:

1. ✅ Dataset is ready at `/home/orrz/gpufs/projects/gemma3/data/nllb_translation/`
2. ✅ Config already points to `all_pairs.jsonl`
3. ✅ Run training: `python train_translation.py --config configs/train_config.yaml`
4. ✅ Training will automatically use local dataset (much faster!)

---

**Note**: The download script uses streaming mode to download data, so it doesn't require downloading the entire NLLB-200 dataset (which is huge). It only downloads the specific language pairs and number of samples you need.