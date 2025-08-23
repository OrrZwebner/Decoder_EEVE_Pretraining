# log_score_evaluator.py

import pandas as pd
import numpy as np
import math
import logging
from collections import Counter
from typing import List

class LogScoreEvaluator:
    """
    Calculates the log-score for a set of new tokens.
    
    The log-score measures the effectiveness of new vocabulary by balancing a token's
    frequency with its ability to compress multiple sub-tokens.
    
    Formula: log_score(token) = P(token) * log2(1 + |SegmentationLength|)
    """
    
    def __init__(self, original_tokenizer, expanded_tokenizer, new_tokens: List[str]):
        """
        Initializes the evaluator.
        
        Args:
            original_tokenizer: The tokenizer before vocabulary expansion.
            expanded_tokenizer: The tokenizer after adding the new tokens.
            new_tokens (List[str]): The list of new, processed tokens that were added.
        """
        self.original_tokenizer = original_tokenizer
        self.expanded_tokenizer = expanded_tokenizer
        self.vocabulary = pd.DataFrame({'token': list(set(new_tokens))}) # Ensure unique tokens
        self.logger = logging.getLogger(__name__)

    def _calculate_segmentation_lengths(self):
        """Calculates how many sub-tokens each new token replaces."""
        self.logger.info("Calculating segmentation lengths for new tokens...")
        
        self.vocabulary['segmentation_length'] = self.vocabulary['token'].apply(
            lambda token: len(self.original_tokenizer.encode(token, add_special_tokens=False))
        )
        self.logger.info("Segmentation lengths calculated.")

    def calculate_score(self, corpus: List[str]) -> float:
        """
        Calculates the total log-score for the new vocabulary.
        
        Args:
            corpus (List[str]): The list of text documents to use for frequency analysis.
            
        Returns:
            float: The total log-score for the entire new vocabulary.
        """
        self.logger.info("Starting log-score calculation...")
        
        # 1. Calculate segmentation lengths
        self._calculate_segmentation_lengths()
        
        # 2. Tokenize corpus with the *expanded* tokenizer to get frequencies
        self.logger.info(f"Tokenizing corpus of {len(corpus)} documents with expanded tokenizer...")
        all_corpus_text = " ".join(corpus)
        corpus_tokens = self.expanded_tokenizer.tokenize(all_corpus_text)
        total_tokens_in_corpus = len(corpus_tokens)
        
        if total_tokens_in_corpus == 0:
            self.logger.warning("Corpus tokenization resulted in 0 tokens. Cannot calculate score.")
            return 0.0

        # 3. Count token occurrences
        self.logger.info(f"Counting {total_tokens_in_corpus:,} tokens for frequency analysis...")
        token_counts = Counter(corpus_tokens)
        
        # 4. Map counts to our new vocabulary DataFrame
        self.vocabulary['count'] = self.vocabulary['token'].map(token_counts).fillna(0).astype(int)
        
        # 5. Calculate unigram probability
        self.vocabulary['probability'] = self.vocabulary['count'] / total_tokens_in_corpus
        
        # 6. Calculate the log-score for each token
        self.vocabulary['log_score'] = self.vocabulary.apply(
            lambda row: row['probability'] * math.log2(1 + row['segmentation_length']),
            axis=1
        )
        
        # 7. Sum the scores for the total
        total_log_score = self.vocabulary['log_score'].sum()
        
        self.logger.info(f"✅ Log-score calculated: {total_log_score:.4f}")
        return total_log_score
    
    def get_vocabulary(self) -> pd.DataFrame:
        """
        Returns the vocabulary DataFrame with calculated scores.
        
        Returns:
            pd.DataFrame: The DataFrame containing tokens, counts, probabilities, segmentation lengths, and log scores.
        """
        # if the vocabulary has not been calculated yet, we can call calculate_score to ensure it's ready
        if 'log_score' not in self.vocabulary.columns:
            self.logger.info("Vocabulary scores not calculated yet. Calculating now...")
            # self.calculate_score([])
        return self.vocabulary.copy()