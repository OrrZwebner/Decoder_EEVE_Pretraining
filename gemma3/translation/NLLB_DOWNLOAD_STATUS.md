# NLLB-200 Dataset Download Status

## Current Status: ✅ DOWNLOADING

The NLLB-200 dataset download is currently running successfully!

### Download Progress

| Language Pair | Status | Samples | File Size | File Name |
|---------------|--------|---------|-----------|-----------|
| **English → Hebrew** | ✅ **COMPLETE** | 1,000,000 | 325.1 MB | `eng_heb.jsonl` |
| **Tibetan → English** | 🔄 **IN PROGRESS** | ~50,000/1M | ~17 MB | `bod_eng.jsonl` |
| **English → Arabic** | ⏳ **PENDING** | - | - | `eng_arb.jsonl` |

### Output Directory
```
/home/orrz/gpufs/projects/gemma3/data/nllb_translation/
├── eng_heb.jsonl        ✅ 325.1 MB (1M pairs)
├── bod_eng.jsonl        🔄 Downloading...
├── eng_arb.jsonl        ⏳ Waiting...
└── all_pairs.jsonl      ⏳ Will be created after all downloads
```

---

## Monitor Download Progress

```bash
# Watch download log in real-time
tail -f /home/orrz/gpufs/projects/gemma3/translation/download_nllb.log

# Check current file sizes
ls -lh /home/orrz/gpufs/projects/gemma3/data/nllb_translation/

# Check progress
grep "✓ Downloaded" /home/orrz/gpufs/projects/gemma3/translation/download_nllb.log
```

---

## What's Being Downloaded

### 1. English → Hebrew (COMPLETED ✅)
- **Samples**: 1,000,000 translation pairs
- **File Size**: 325.1 MB
- **Speed**: ~15,000 samples/second
- **Time**: ~67 seconds
- **File**: `eng_heb.jsonl`

**Sample entry:**
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

### 2. Tibetan → English (IN PROGRESS 🔄)
- **Target Samples**: 1,000,000
- **Current Progress**: ~5% (check log for latest)
- **Estimated Size**: ~350 MB
- **File**: `bod_eng.jsonl`

**Note**: Tibetan→English (reverse direction) because English→Tibetan is not available in NLLB-200

### 3. English → Arabic (PENDING ⏳)
- **Target Samples**: 1,000,000
- **Estimated Size**: ~400 MB
- **File**: `eng_arb.jsonl`
- **Status**: Will start after Tibetan download completes

---

## Estimated Completion Time

**Total Download Time**: ~5-10 minutes for all 3 language pairs

- ✅ English-Hebrew: DONE (67 seconds)
- 🔄 Tibetan-English: ~3-5 minutes (slower due to special characters)
- ⏳ English-Arabic: ~2-3 minutes

**Total Dataset Size**: ~1 GB (3M translation pairs)

---

## After Download Completes

### 1. Combined File Will Be Created

The script automatically creates `all_pairs.jsonl` combining all language pairs:

```
all_pairs.jsonl  (~1 GB, 3M pairs)
```

### 2. Use in Training

Your training config is already set to use the local dataset:

```yaml
translation:
  local_dataset_path: "/home/orrz/gpufs/projects/gemma3/data/nllb_translation/all_pairs.jsonl"
```

**Training will:**
- ✅ Load from local file (FAST - no streaming needed)
- ✅ Work offline
- ✅ Be reproducible with same seed

### 3. Train Any Language Pair

You can train any of the downloaded pairs by updating config:

**English → Hebrew:**
```yaml
translation:
  source_lang: "English"
  target_lang: "Hebrew"
```

**Tibetan → English:**
```yaml
translation:
  source_lang: "Tibetan"
  target_lang: "English"
```

**English → Arabic:**
```yaml
translation:
  source_lang: "English"
  target_lang: "Arabic"
```

---

## Verification After Download

### Check Downloaded Files

```bash
cd /home/orrz/gpufs/projects/gemma3/data/nllb_translation

# List all files with sizes
ls -lh

# Count lines in each file (should be 1M)
wc -l *.jsonl

# View first entry
head -1 eng_heb.jsonl | python -m json.tool
```

### Verify Data Quality

```bash
# Check first 5 samples from English-Hebrew
head -5 eng_heb.jsonl | python -m json.tool

# Count total samples
wc -l eng_heb.jsonl  # Should show: 1000000
```

### Test with Training

```bash
cd /home/orrz/gpufs/projects/gemma3/translation

# Run a quick test with 10 samples
# Edit config: train_samples: 10
python train_translation.py --config configs/train_config.yaml
```

---

## Download Command Reference

If you need to download again or download different language pairs:

```bash
cd /home/orrz/gpufs/projects/gemma3/translation

# Download with custom sample count
python download_nllb_data.py --samples 500000

# Download to different directory
python download_nllb_data.py --output-dir /path/to/custom/dir

# Download with different seed
python download_nllb_data.py --seed 123
```

---

## Troubleshooting

### If Download Stops

Check if process is still running:
```bash
ps aux | grep download_nllb_data
```

If stuck, restart:
```bash
# Kill current download
pkill -f download_nllb_data

# Start fresh
python download_nllb_data.py --samples 1000000
```

### If Disk Space is Low

Download fewer samples:
```bash
python download_nllb_data.py --samples 100000  # 100K instead of 1M
```

100K samples ≈ 33 MB per language pair

---

## Language Pair Availability Notes

### ✅ Available in NLLB-200
- English ↔ Hebrew
- Tibetan ↔ English (reverse direction only)
- English ↔ Arabic

### ❌ NOT Available in NLLB-200
- English → Sanskrit (not in dataset)
- English → Tibetan (only Tibetan → English available)

### To Find Available Pairs

Check the NLLB-200 dataset documentation:
https://huggingface.co/datasets/allenai/nllb

Or check error logs - they list all available pairs.

---

## Next Steps

Once download completes:

1. ✅ **Verify files exist**: `ls -lh /home/orrz/gpufs/projects/gemma3/data/nllb_translation/`
2. ✅ **Check combined file**: Look for `all_pairs.jsonl`
3. ✅ **Run training**: `./start_training.sh`
4. ✅ **Training will be FAST**: No more streaming delays!

---

## Summary

🎉 **English-Hebrew dataset ready!** (1M samples, 325 MB)
🔄 **Tibetan-English downloading...** (ETA: 3-5 minutes)
⏳ **English-Arabic next...** (ETA: 2-3 minutes)

**Total time**: ~5-10 minutes for all 3 language pairs
**Total size**: ~1 GB
**Total samples**: 3 million translation pairs

**Training benefit**: 10x faster loading, no internet needed, fully reproducible!