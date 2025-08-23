# Sanskrit Tokenizer Optimization Research

A research tool that optimizes modern language model tokenizers for Sanskrit text processing through systematic algorithm comparison and vocabulary enhancement.

## Overview

Modern language models like LLaMA and Gemma use tokenizers optimized for English and common European languages. When processing Sanskrit text, these tokenizers are highly inefficient, requiring 2-3x more tokens than necessary. This research tool addresses this inefficiency by:

1. **Training specialized tokenizers** on Sanskrit corpora using different algorithms
2. **Extracting the most effective Sanskrit-specific tokens** 
3. **Adding exactly N new tokens** to existing model vocabularies
4. **Measuring compression improvements** and efficiency gains

**Key Innovation**: Smart duplicate filtering ensures exactly 128 new unique tokens are added to each model, accounting for Sanskrit substrings already present in large modern vocabularies (Gemma has 256K tokens, many overlapping with Sanskrit).

## Methodology

### Core Research Question
*"Which tokenization algorithm most effectively learns Sanskrit-specific subword units that improve compression ratios when added to existing LLM vocabularies?"*

### Experimental Design

**Training Phase**:
1. Sample Sanskrit texts from corpus (configurable: full dataset or N samples)
2. Train 5 different tokenization algorithms on Sanskrit data
3. Extract learned subword tokens from each algorithm
4. Filter duplicates against target model vocabularies
5. Select exactly 128 unique Sanskrit tokens per algorithm

**Evaluation Phase**:
1. Add learned tokens to original model tokenizers
2. Measure compression ratio improvements (chars/token)
3. Test on representative Sanskrit texts
4. Compare algorithm effectiveness across models

**Controlled Variables**:
- Exact same training data across all algorithms
- Identical number of added tokens (128) per model
- Same evaluation texts and metrics
- Consistent model architectures (LLaMA 3.2-1B, Gemma 2-2B)

### Tokenization Algorithms Tested

#### BPE (Byte Pair Encoding)
**Methodology**: Iteratively merges most frequent character pairs
- **Sanskrit Relevance**: Learns common conjuncts like "क्ष", "त्र", "ज्ञ"
- **Training**: Deterministic, uses Whitespace pre-tokenization
- **Processing**: Direct compatibility with LLaMA's BPE architecture

#### WordPiece  
**Methodology**: Likelihood-based subword segmentation with continuation markers
- **Sanskrit Relevance**: Handles morphological complexity through subword boundaries
- **Training**: Uses "##" prefixes for non-initial subwords
- **Processing**: Prefixes removed for LLaMA compatibility

#### Unigram
**Methodology**: Probabilistic tokenization maintaining multiple segmentation candidates
- **Sanskrit Relevance**: Flexible morpheme learning for complex Sanskrit morphology
- **Challenge**: Non-deterministic - same training parameters yield different vocabulary sizes
- **Solution**: Iterative heuristic training with binary search to achieve exact token count

#### SentencePiece BPE (Gemma)
**Methodology**: Google's unified framework implementing BPE with Unicode handling
- **Sanskrit Relevance**: Designed for multilingual use, excellent Unicode support
- **Training**: Uses "▁" prefixes for word boundaries
- **Processing**: Native compatibility with Gemma's SentencePiece tokenizer

#### SentencePiece Unigram (Gemma)
**Methodology**: SentencePiece implementation of Unigram algorithm
- **Sanskrit Relevance**: Combines SentencePiece Unicode handling with Unigram flexibility
- **Training**: File-based training eliminates iterator warnings
- **Processing**: Maintains ▁ prefixes for Gemma compatibility

### Smart Duplicate Filtering

**Problem**: Modern tokenizers already contain many Sanskrit substrings
- LLaMA (128K vocab): ~15-20% overlap with common Sanskrit sequences
- Gemma (256K vocab): ~40-50% overlap with Sanskrit characters/syllables

**Solution**: Progressive buffer training
1. Train with 2x target vocabulary size
2. Filter against existing model vocabulary  
3. If insufficient unique tokens, increase to 3x, 4x, 5x, 6x
4. For Unigram: Use iterative approach with binary search
5. Return exactly 128 unique tokens per algorithm

### Model-Specific Processing

#### LLaMA Processing
- **Space Handling**: Converts Ġ tokens to standard spaces
- **Prefix Removal**: Strips ## (WordPiece) and ▁ (Unigram) prefixes
- **Format**: BPE-compatible token format

#### Gemma Processing  
- **SentencePiece Native**: Keeps ▁ prefixes for word boundaries
- **Cross-Algorithm**: Creates both ▁prefixed and base versions
- **Format**: SentencePiece-compatible token format

### Evaluation Metrics

#### Primary: Compression Ratio
- **Formula**: `total_characters / total_tokens`
- **Interpretation**: Higher values = better compression
- **Measurement**: 1000 randomly sampled texts

#### Secondary: Token Count Reduction
- **Method**: Before/after tokenization of standard Sanskrit texts
- **Tests**: Bhagavad Gita verses, Upanishad passages
- **Analysis**: Token-level improvement breakdown

#### Efficiency: Improvement per Token Added
- **Formula**: `improvement_percentage / tokens_added`
- **Purpose**: Identify most efficient algorithms
- **Application**: Guide optimal token budget allocation

## Authentication Setup

Both models require HuggingFace authentication:

```bash
# Method 1: Interactive login
huggingface-cli login

# Method 2: Environment variable
export HF_TOKEN="your_huggingface_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

## Running the Experiment

### Quick Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your data path in sanskrit_config.yaml
# 3. Set up HuggingFace authentication
# 4. Run the experiment
```

### Execution
```bash
# From llama3/ directory
nohup python tokenizers/main.py > logs/tokenizers/experiment.log 2>&1 &

# Monitor progress
tail -f logs/tokenizers/experiment.log
```

### Configuration (sanskrit_config.yaml)
```yaml
data:
  path: "sanskrit_data/train.jsonl"  # Your Sanskrit dataset

training:
  num_samples: null      # Use full dataset
  num_new_tokens: 128    # Tokens to add per model

models:
  llama:
    model_name: "meta-llama/Llama-3.2-1B"
    algorithms: ["bpe", "unigram", "wordpiece"]
  
  gemma:
    model_name: "google/gemma-2-2b" 
    algorithms: ["sentencepiece_bpe", "sentencepiece_unigram"]
```

## Expected Research Outcomes

### Quantitative Results
- **Compression ratio improvements**: 10-25% expected for Sanskrit text
- **Algorithm ranking**: Efficiency comparison across tokenization methods
- **Model compatibility**: LLaMA vs Gemma adaptation effectiveness

### Qualitative Insights  
- **Morphological capture**: Which algorithms best learn Sanskrit morphemes
- **Unicode handling**: Effectiveness across different Unicode normalization
- **Scalability**: Performance with varying dataset sizes

### Research Applications
- **Cross-lingual tokenization** optimization
- **Low-resource language** model adaptation
- **Morphologically rich language** processing
- **Domain-specific vocabulary** enhancement

## Output Analysis

### Automated Reports
- **Compression ratio comparison** across algorithms
- **Statistical significance** testing
- **Efficiency rankings** by improvement per token
- **Token usage verification** in actual text processing

### Visualizations
- **4-panel comparison plots**: Ratios, improvements, tokens added, efficiency
- **Algorithm performance** across different model architectures
- **Before/after tokenization** examples with detailed analysis

## Technical Notes

### Reproducibility
- **Fixed random seeds** for consistent sampling
- **Deterministic training** where possible (BPE, WordPiece)
- **Heuristic convergence** criteria for non-deterministic algorithms

### Scalability
- **Memory management** for large datasets (>500K texts)
- **Progressive training** with automatic buffer adjustment
- **Cleanup procedures** for temporary files and models

### Error Handling
- **Authentication validation** with helpful error messages
- **Data format verification** across multiple file types
- **Graceful algorithm failure** handling with detailed logging

This research tool enables systematic comparison of tokenization algorithms for Sanskrit optimization, providing quantitative evidence for optimal vocabulary enhancement strategies in modern language models.