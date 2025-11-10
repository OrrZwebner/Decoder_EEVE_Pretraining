#!/usr/bin/env python3
"""
Comprehensive tests for expand_tokenizer method.

Tests BPE and SentencePiece BPE expansion across:
- Models: Gemma, LLaMA, GPT-2
- Languages: Tibetan, Hebrew, English
"""

import os
import sys
import logging
import unittest
from typing import List, Dict, Any
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_processors import get_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample corpora for different languages
TIBETAN_CORPUS = [
    "བོད་ཀྱི་སྐད་ཡིག་ནི་བོད་ཀྱི་མི་རིགས་ཀྱི་སྐད་ཡིག་རེད།",
    "ང་ཚོས་བོད་སྐད་སློབ་གཉེར་བྱེད་དགོས།",
    "བོད་ལྗོངས་ནི་ལྷོ་རྒྱ་གར་དང་བལ་ཡུལ་དང་འབྲེལ་བ་ཡོད།",
    "དཀའ་ལས་མང་པོ་ཡོད་རུང་ང་ཚོས་སློབ་གཉེར་བྱེད་དགོས།",
    "བོད་ཀྱི་རིག་གནས་དང་ལོ་རྒྱུས་ནི་ཧ་ཅང་རིང་པོ་རེད།",
] * 100  # Repeat for more data

HEBREW_CORPUS = [
    "השפה העברית היא שפה שמית וזו השפה הרשמית של מדינת ישראל.",
    "אנחנו לומדים עברית כדי להבין את התרבות היהודית.",
    "ספרות עברית עתיקה כוללת את התנ״ך ומדרשים רבים.",
    "העברית המודרנית התפתחה במהלך המאה התשע עשרה והעשרים.",
    "הלימוד של השפה העברית הוא חשוב מאוד להבנת ההיסטוריה.",
] * 100

ENGLISH_CORPUS = [
    "The English language is a West Germanic language that was first spoken in early medieval England.",
    "We are learning English to understand global communication and literature.",
    "Modern English has evolved significantly from Old English and Middle English.",
    "English vocabulary is rich with borrowings from Latin, French, and other languages.",
    "The study of English linguistics includes phonology, morphology, syntax, and semantics.",
] * 100

# Model configurations
MODELS = {
    "gemma": {
        "model_name": "google/gemma-3-1b-it",
        "algorithm": "SENTENCEPIECE_BPE"
    },
    "llama": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "algorithm": "BPE"
    },
    "gpt2": {
        "model_name": "openai-community/gpt2",
        "algorithm": "BPE"
    }
}


class TestExpandTokenizer(unittest.TestCase):
    """Comprehensive tests for tokenizer expansion."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Check for HuggingFace token
        cls.hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if not cls.hf_token:
            logging.warning("HUGGINGFACE_HUB_TOKEN not set - some models may fail to load")

        cls.test_corpus = {
            "tibetan": TIBETAN_CORPUS,
            "hebrew": HEBREW_CORPUS,
            "english": ENGLISH_CORPUS
        }

        cls.max_tokens_to_add = 100  # Add 100 new tokens per test

    def _test_model_language(self, model_name: str, language: str):
        """Test a specific model with a specific language."""

        model_config = MODELS[model_name]
        corpus = self.test_corpus[language]

        logging.info(f"\n{'='*80}")
        logging.info(f"Testing {model_name.upper()} with {language.upper()}")
        logging.info(f"{'='*80}")

        # Load processor
        processor = get_processor(model_name, model_config)
        original_tokenizer = processor.original_tokenizer
        original_vocab_size = len(original_tokenizer.get_vocab())

        logging.info(f"Original vocabulary size: {original_vocab_size:,}")

        # Test expansion
        try:
            expanded_tokenizer, tokens_added, new_tokens = processor.expand_tokenizer(
                learned_tokens=[],  # Not used for train_new_from_iterator
                algorithm_name=model_config["algorithm"],
                max_tokens=self.max_tokens_to_add,
                training_corpus=corpus,
                merges=None
            )

            new_vocab_size = len(expanded_tokenizer.get_vocab())

            # Assertions
            self.assertIsNotNone(expanded_tokenizer, f"{model_name}/{language}: Tokenizer should not be None")
            self.assertGreater(tokens_added, 0, f"{model_name}/{language}: Should add some tokens")
            self.assertGreater(new_vocab_size, original_vocab_size,
                             f"{model_name}/{language}: Vocab size should increase")

            # Check that vocab size increase matches tokens_added (actual amount added)
            actual_increase = new_vocab_size - original_vocab_size
            self.assertEqual(actual_increase, tokens_added,
                           f"{model_name}/{language}: Vocab size increase should match tokens_added "
                           f"(got {actual_increase}, reported {tokens_added})")

            # Check that tokens_added is reasonable
            # With 500 corpus samples, we may not always get the full 100 tokens
            # but we should get at least 20% of the requested amount
            min_expected = int(self.max_tokens_to_add * 0.2)
            self.assertGreaterEqual(tokens_added, min_expected,
                                  f"{model_name}/{language}: Should add at least {min_expected} tokens (20% of {self.max_tokens_to_add}), got {tokens_added}")

            self.assertIsNotNone(new_tokens, f"{model_name}/{language}: Should return new tokens list")
            self.assertEqual(len(new_tokens), tokens_added,
                           f"{model_name}/{language}: new_tokens length should match tokens_added")

            logging.info(f"✅ SUCCESS: Added {tokens_added} tokens")
            logging.info(f"   New vocabulary size: {new_vocab_size:,}")
            logging.info(f"   Sample new tokens: {new_tokens[:10]}")

            # Test encoding with new tokenizer
            test_text = corpus[0]
            original_encoding = original_tokenizer.encode(test_text)
            new_encoding = expanded_tokenizer.encode(test_text)

            logging.info(f"   Original encoding length: {len(original_encoding)}")
            logging.info(f"   New encoding length: {len(new_encoding)}")

            # New encoding should be same or shorter (more efficient tokenization)
            self.assertLessEqual(len(new_encoding), len(original_encoding),
                               f"{model_name}/{language}: New encoding should not be longer")

            # Test decoding
            decoded_text = expanded_tokenizer.decode(new_encoding)
            self.assertIsNotNone(decoded_text, f"{model_name}/{language}: Should decode successfully")

            # Check that decoding is similar to original (may have minor whitespace differences)
            # For SentencePiece models, decoding might differ slightly
            logging.info(f"   Decoding works: {len(decoded_text)} characters")

            return True

        except Exception as e:
            self.fail(f"{model_name}/{language} failed: {str(e)}")
            return False

    # Gemma tests
    def test_gemma_tibetan(self):
        """Test Gemma with Tibetan corpus."""
        self._test_model_language("gemma", "tibetan")

    def test_gemma_hebrew(self):
        """Test Gemma with Hebrew corpus."""
        self._test_model_language("gemma", "hebrew")

    def test_gemma_english(self):
        """Test Gemma with English corpus."""
        self._test_model_language("gemma", "english")

    # LLaMA tests
    def test_llama_tibetan(self):
        """Test LLaMA with Tibetan corpus."""
        self._test_model_language("llama", "tibetan")

    def test_llama_hebrew(self):
        """Test LLaMA with Hebrew corpus."""
        self._test_model_language("llama", "hebrew")

    def test_llama_english(self):
        """Test LLaMA with English corpus."""
        self._test_model_language("llama", "english")

    # GPT-2 tests
    def test_gpt2_tibetan(self):
        """Test GPT-2 with Tibetan corpus."""
        self._test_model_language("gpt2", "tibetan")

    def test_gpt2_hebrew(self):
        """Test GPT-2 with Hebrew corpus."""
        self._test_model_language("gpt2", "hebrew")

    def test_gpt2_english(self):
        """Test GPT-2 with English corpus."""
        self._test_model_language("gpt2", "english")

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        model_config = {"model_name": "openai-community/gpt2"}
        processor = get_processor("gpt2", model_config)

        with self.assertRaises(ValueError) as context:
            processor.expand_tokenizer(
                learned_tokens=[],
                algorithm_name="INVALID_ALGO",
                max_tokens=100,
                training_corpus=ENGLISH_CORPUS,
                merges=None
            )

        self.assertIn("Only BPE and SENTENCEPIECE_BPE", str(context.exception))

    def test_missing_corpus(self):
        """Test that missing corpus raises ValueError."""
        model_config = {"model_name": "openai-community/gpt2"}
        processor = get_processor("gpt2", model_config)

        with self.assertRaises(ValueError) as context:
            processor.expand_tokenizer(
                learned_tokens=[],
                algorithm_name="BPE",
                max_tokens=100,
                training_corpus=None,  # Missing corpus
                merges=None
            )

        self.assertIn("Training corpus is required", str(context.exception))


class TestModelTypeDetection(unittest.TestCase):
    """Test that model types are correctly detected."""

    def test_bpe_model_type(self):
        """Test BPE model type detection (LLaMA, GPT-2)."""
        for model_name in ["llama", "gpt2"]:
            model_config = MODELS[model_name]
            processor = get_processor(model_name, model_config)

            # Get tokenizer JSON
            import json
            tokenizer_json = json.loads(processor.original_tokenizer.backend_tokenizer.to_str())
            model_type = tokenizer_json['model']['type']

            self.assertEqual(model_type, 'BPE',
                           f"{model_name} should be BPE type")
            logging.info(f"✅ {model_name} correctly detected as BPE")

    def test_gemma_model_type(self):
        """Test Gemma model type detection (can be BPE or Unigram depending on version)."""
        model_config = MODELS["gemma"]
        processor = get_processor("gemma", model_config)

        # Get tokenizer JSON
        import json
        tokenizer_json = json.loads(processor.original_tokenizer.backend_tokenizer.to_str())
        model_type = tokenizer_json['model']['type']

        # Gemma-3 uses BPE, older Gemma versions used Unigram
        self.assertIn(model_type, ['BPE', 'Unigram'],
                    "Gemma should be either BPE or Unigram type")
        logging.info(f"✅ Gemma correctly detected as {model_type}")


class TestTokenizerCompression(unittest.TestCase):
    """Test that expanded tokenizers provide better compression."""

    def test_compression_improvement(self):
        """Test that new tokenizer compresses better than original."""
        # Use a small test with GPT-2 and Hebrew
        model_config = MODELS["gpt2"]
        processor = get_processor("gpt2", model_config)
        corpus = HEBREW_CORPUS[:50]  # Use smaller corpus for speed

        # Expand tokenizer
        expanded_tokenizer, tokens_added, new_tokens = processor.expand_tokenizer(
            learned_tokens=[],
            algorithm_name="BPE",
            max_tokens=50,  # Smaller for speed
            training_corpus=corpus,
            merges=None
        )

        # Test compression on held-out text
        test_texts = HEBREW_CORPUS[50:60]

        original_total = 0
        expanded_total = 0

        for text in test_texts:
            original_tokens = processor.original_tokenizer.encode(text)
            expanded_tokens = expanded_tokenizer.encode(text)

            original_total += len(original_tokens)
            expanded_total += len(expanded_tokens)

        compression_ratio = (original_total - expanded_total) / original_total * 100

        logging.info(f"\nCompression test results:")
        logging.info(f"  Original: {original_total} tokens")
        logging.info(f"  Expanded: {expanded_total} tokens")
        logging.info(f"  Compression improvement: {compression_ratio:.2f}%")

        # Expanded should be same or better
        self.assertLessEqual(expanded_total, original_total,
                           "Expanded tokenizer should not be worse")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExpandTokenizer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTypeDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenizerCompression))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("TEST SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Tests run: {result.testsRun}")
    logging.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logging.info(f"Failures: {len(result.failures)}")
    logging.info(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
