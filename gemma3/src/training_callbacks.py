#!/usr/bin/env python3
"""
Custom training callbacks for Gemma-3 Sanskrit training
"""

import logging
import torch
import math
import csv
import os
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from transformers import TrainerCallback


class SampleLoggingCallback(TrainerCallback):
    """
    Callback to log sample predictions during training
    """
    
    def __init__(self, tokenizer, log_every_n_steps=500):
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.last_logged_step = -1  # Start at -1 so first step logs
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when trainer logs metrics (happens every logging_steps)
        This is more reliable than on_step_end for getting training info
        """
        # Only log at specified intervals
        if state.global_step - self.last_logged_step < self.log_every_n_steps:
            return
        
        # Don't log if we haven't actually trained yet
        if state.global_step == 0:
            return
        
        self.last_logged_step = state.global_step
        
        try:
            logging.info("\n" + "="*80)
            logging.info(f"SAMPLE LOGGING AT STEP {state.global_step}")
            logging.info("="*80)
            
            # Log current metrics
            if logs:
                logging.info("Current metrics:")
                for key, value in logs.items():
                    if isinstance(value, float):
                        logging.info(f"  {key}: {value:.4f}")
                    else:
                        logging.info(f"  {key}: {value}")
            
            # Get model from kwargs
            model = kwargs.get('model')
            
            if model is not None:
                # Generate a sample prediction
                logging.info("\nGenerating sample prediction...")
                
                # Create a simple prompt
                # prompt = "שלום, "  # Simple test prompt
                prompt = "היום אני"
                
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        model.eval()
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        model.train()

                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logging.info(f"Generated: {generated_text[:200]}")

                    # Clean up GPU memory after generation
                    # This prevents memory accumulation from repeated inference calls
                    del inputs, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logging.warning(f"Could not generate sample: {e}")
            
            logging.info("="*80 + "\n")
            
        except Exception as e:
            logging.warning(f"Sample logging failed: {e}")


class DataInspectionCallback(TrainerCallback):
    """
    Callback to inspect data quality at the start of training
    """
    
    def __init__(self, tokenizer, num_samples=3):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.has_logged = False
    
    def on_train_begin(self, args, state, control, train_dataloader=None, **kwargs):
        """
        Called at the beginning of training
        """
        if self.has_logged or train_dataloader is None:
            return
        
        self.has_logged = True
        
        try:
            logging.info("\n" + "="*80)
            logging.info("DATA INSPECTION - FIRST TRAINING BATCHES")
            logging.info("="*80)
            
            # Get first few batches
            dataloader_iter = iter(train_dataloader)
            
            for i in range(min(self.num_samples, len(train_dataloader))):
                batch = next(dataloader_iter)
                
                logging.info(f"\n--- Batch {i+1} ---")
                logging.info(f"Batch keys: {batch.keys()}")
                logging.info(f"Input IDs shape: {batch['input_ids'].shape}")
                logging.info(f"Attention mask shape: {batch['attention_mask'].shape}")
                
                if 'labels' in batch:
                    logging.info(f"Labels shape: {batch['labels'].shape}")
                    
                    # Count padding in labels
                    labels = batch['labels']
                    total_tokens = labels.numel()
                    masked_tokens = (labels == -100).sum().item()
                    actual_tokens = total_tokens - masked_tokens
                    
                    logging.info(f"  Total label tokens: {total_tokens}")
                    logging.info(f"  Masked tokens (-100): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
                    logging.info(f"  Actual tokens: {actual_tokens} ({actual_tokens/total_tokens*100:.1f}%)")
                
                # Decode first sample
                first_input = batch['input_ids'][0]
                decoded = self.tokenizer.decode(first_input, skip_special_tokens=True)
                logging.info(f"\nFirst sample decoded (first 300 chars):")
                logging.info(f"  {decoded[:300]}...")
                
                # Show padding statistics
                attention_mask = batch['attention_mask'][0]
                seq_length = len(attention_mask)
                non_padding = attention_mask.sum().item()
                padding = seq_length - non_padding
                
                logging.info(f"\nPadding statistics:")
                logging.info(f"  Sequence length: {seq_length}")
                logging.info(f"  Actual tokens: {non_padding} ({non_padding/seq_length*100:.1f}%)")
                logging.info(f"  Padding tokens: {padding} ({padding/seq_length*100:.1f}%)")
            
            logging.info("="*80 + "\n")
            
        except Exception as e:
            logging.error(f"Error during data inspection: {e}")



class EpochBasedStoppingCallback(TrainerCallback):
    """
    Stop training after a specified number of epochs, even with max_steps set.
    This is necessary for streaming datasets where max_steps is required but
    we want epochs to control the actual training duration.
    """
    
    def __init__(self, num_epochs: int):
        if num_epochs is None or num_epochs <= 0:
            logging.warning(f"Invalid num_epochs: {num_epochs}, defaulting to 1")
            num_epochs = 1
        self.num_epochs = num_epochs
        self.epoch_count = 0
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Called at the end of each epoch
        """
        self.epoch_count += 1
        logging.info(f"✅ Completed epoch {self.epoch_count}/{self.num_epochs}")
        
        if self.epoch_count >= self.num_epochs:
            logging.info(f"🛑 Reached {self.num_epochs} epochs - stopping training")
            control.should_training_stop = True

        return control


class EnhancedMetricsCallback(TrainerCallback):
    """
    Enhanced metrics tracking callback that logs:
    - Perplexity (exp(loss))
    - Token usage statistics for custom tokens
    - Gradient norms
    - Learning rate
    - Memory usage
    - Loss trends (train vs eval)
    """

    def __init__(self, tokenizer, config, base_vocab_size=None):
        self.tokenizer = tokenizer
        self.config = config
        self.base_vocab_size = base_vocab_size or config.get('model', {}).get('vocab_size', len(tokenizer))

        # Track metrics over time
        self.train_losses = []
        self.eval_losses = []
        self.train_perplexities = []
        self.eval_perplexities = []
        self.learning_rates = []
        self.steps = []

        # Token usage tracking
        self.custom_token_start_idx = self.base_vocab_size
        self.custom_token_count = len(tokenizer) - self.base_vocab_size
        self.custom_token_usage = defaultdict(int)

        logging.info(f"✅ Enhanced Metrics Callback initialized:")
        logging.info(f"   Base vocab size: {self.base_vocab_size:,}")
        logging.info(f"   Total vocab size: {len(tokenizer):,}")
        logging.info(f"   Custom tokens added: {self.custom_token_count:,}")
        logging.info(f"   Custom token range: [{self.custom_token_start_idx}:{len(tokenizer)}]")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track metrics whenever trainer logs"""
        if logs is None:
            return

        step = state.global_step
        self.steps.append(step)

        # Track training loss and perplexity
        if 'loss' in logs:
            train_loss = logs['loss']
            self.train_losses.append(train_loss)
            train_perplexity = math.exp(train_loss) if train_loss < 20 else float('inf')
            self.train_perplexities.append(train_perplexity)

            # Log perplexity
            logs['perplexity'] = train_perplexity
            logging.info(f"📊 Step {step} - Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")

        # Track eval loss and perplexity
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            self.eval_losses.append(eval_loss)
            eval_perplexity = math.exp(eval_loss) if eval_loss < 20 else float('inf')
            self.eval_perplexities.append(eval_perplexity)

            # Log eval perplexity
            logs['eval_perplexity'] = eval_perplexity
            logging.info(f"📊 Step {step} - Eval Loss: {eval_loss:.4f}, Eval Perplexity: {eval_perplexity:.2f}")

            # Analyze train/eval gap
            if self.train_losses:
                latest_train_loss = self.train_losses[-1]
                gap = eval_loss - latest_train_loss
                gap_pct = (gap / latest_train_loss) * 100 if latest_train_loss > 0 else 0

                logging.info(f"📈 Loss Gap: {gap:.4f} ({gap_pct:+.1f}%)")
                if gap_pct > 20:
                    logging.warning("⚠️  Large train/eval gap - possible overfitting!")
                elif gap_pct < -10:
                    logging.warning("⚠️  Eval loss lower than train loss - check for data leakage!")

        # Track learning rate
        if 'learning_rate' in logs:
            lr = logs['learning_rate']
            self.learning_rates.append(lr)
            logging.info(f"📚 Learning Rate: {lr:.2e}")

        # Track gradient norm
        if 'grad_norm' in logs:
            grad_norm = logs['grad_norm']
            logging.info(f"📐 Gradient Norm: {grad_norm:.4f}")

            # Warn about gradient issues
            if grad_norm > 10.0:
                logging.warning(f"⚠️  High gradient norm ({grad_norm:.2f}) - may indicate instability!")
            elif grad_norm < 0.0001:
                logging.warning(f"⚠️  Very low gradient norm ({grad_norm:.2e}) - may indicate vanishing gradients!")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Add custom metrics during evaluation"""
        if metrics is None:
            return

        # Add perplexity if not already added
        if 'eval_loss' in metrics and 'eval_perplexity' not in metrics:
            eval_loss = metrics['eval_loss']
            metrics['eval_perplexity'] = math.exp(eval_loss) if eval_loss < 20 else float('inf')

        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logging.info(f"💾 GPU {i} - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")

    def on_train_end(self, args, state, control, **kwargs):
        """Generate summary report at end of training"""
        logging.info("\n" + "="*80)
        logging.info("TRAINING METRICS SUMMARY")
        logging.info("="*80)

        if self.train_losses:
            logging.info(f"Training Loss: {self.train_losses[0]:.4f} → {self.train_losses[-1]:.4f}")
            logging.info(f"Training Perplexity: {self.train_perplexities[0]:.2f} → {self.train_perplexities[-1]:.2f}")

        if self.eval_losses:
            logging.info(f"Eval Loss: {self.eval_losses[0]:.4f} → {self.eval_losses[-1]:.4f}")
            logging.info(f"Eval Perplexity: {self.eval_perplexities[0]:.2f} → {self.eval_perplexities[-1]:.2f}")

        if self.learning_rates:
            logging.info(f"Learning Rate: {self.learning_rates[0]:.2e} → {self.learning_rates[-1]:.2e}")

        logging.info("="*80 + "\n")


class TokenUsageCallback(TrainerCallback):
    """
    Track usage statistics for custom tokens during training.
    Answers: "Are new tokens being used?"
    """

    def __init__(self, tokenizer, base_vocab_size, log_every_n_steps=500):
        self.tokenizer = tokenizer
        self.base_vocab_size = base_vocab_size
        self.custom_token_start_idx = base_vocab_size
        self.custom_token_count = len(tokenizer) - base_vocab_size
        self.log_every_n_steps = log_every_n_steps

        # Track token usage
        self.total_tokens_seen = 0
        self.custom_tokens_seen = 0
        self.token_frequency = defaultdict(int)

        self.last_logged_step = -1

        logging.info(f"✅ Token Usage Callback initialized:")
        logging.info(f"   Tracking {self.custom_token_count} custom tokens")
        logging.info(f"   Custom token range: [{self.custom_token_start_idx}:{len(tokenizer)}]")

    def on_step_end(self, args, state, control, **kwargs):
        """Track token usage during training"""
        # Only analyze periodically to avoid slowdown
        if state.global_step - self.last_logged_step < self.log_every_n_steps:
            return

        self.last_logged_step = state.global_step

        # Get current batch from kwargs
        if 'inputs' in kwargs:
            inputs = kwargs['inputs']

            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']

                # Analyze token usage
                self._analyze_tokens(input_ids, state.global_step)

    def _analyze_tokens(self, input_ids, step):
        """Analyze token usage in a batch"""
        # Flatten and move to CPU
        tokens = input_ids.flatten().cpu().tolist()

        # Count custom token usage
        custom_token_count = 0
        for token_id in tokens:
            self.total_tokens_seen += 1
            self.token_frequency[token_id] += 1

            if token_id >= self.custom_token_start_idx and token_id < len(self.tokenizer):
                custom_token_count += 1
                self.custom_tokens_seen += 1

        # Calculate usage percentage
        usage_pct = (custom_token_count / len(tokens)) * 100 if tokens else 0
        overall_usage_pct = (self.custom_tokens_seen / self.total_tokens_seen) * 100 if self.total_tokens_seen > 0 else 0

        # Log results
        logging.info("\n" + "="*80)
        logging.info(f"CUSTOM TOKEN USAGE - Step {step}")
        logging.info("="*80)
        logging.info(f"📊 Current batch:")
        logging.info(f"   Total tokens: {len(tokens)}")
        logging.info(f"   Custom tokens: {custom_token_count} ({usage_pct:.2f}%)")
        logging.info(f"📊 Overall statistics:")
        logging.info(f"   Total tokens seen: {self.total_tokens_seen:,}")
        logging.info(f"   Custom tokens seen: {self.custom_tokens_seen:,} ({overall_usage_pct:.2f}%)")

        # Find most used custom tokens
        custom_token_freq = {k: v for k, v in self.token_frequency.items()
                            if k >= self.custom_token_start_idx and k < len(self.tokenizer)}

        if custom_token_freq:
            top_custom = sorted(custom_token_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            logging.info(f"\n📌 Top 5 custom tokens:")
            for token_id, count in top_custom:
                try:
                    token_str = self.tokenizer.decode([token_id])
                    logging.info(f"   {token_id}: '{token_str}' - {count:,} times")
                except:
                    logging.info(f"   {token_id}: (decode error) - {count:,} times")
        else:
            logging.warning("⚠️  NO CUSTOM TOKENS USED YET!")
            logging.warning("   This may indicate:")
            logging.warning("   - Custom tokens are not in the training data")
            logging.warning("   - Tokenization is using original tokens instead")
            logging.warning("   - Vocabulary expansion did not work correctly")

        logging.info("="*80 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Final report on token usage"""
        overall_usage_pct = (self.custom_tokens_seen / self.total_tokens_seen) * 100 if self.total_tokens_seen > 0 else 0

        logging.info("\n" + "="*80)
        logging.info("FINAL CUSTOM TOKEN USAGE REPORT")
        logging.info("="*80)
        logging.info(f"Total tokens processed: {self.total_tokens_seen:,}")
        logging.info(f"Custom tokens used: {self.custom_tokens_seen:,} ({overall_usage_pct:.2f}%)")
        logging.info(f"Custom tokens available: {self.custom_token_count:,}")

        # Calculate how many of the custom tokens were actually used
        custom_tokens_used = sum(1 for token_id in self.token_frequency.keys()
                                if token_id >= self.custom_token_start_idx)
        logging.info(f"Unique custom tokens used: {custom_tokens_used}/{self.custom_token_count} ({custom_tokens_used/self.custom_token_count*100:.1f}%)")
        logging.info("="*80 + "\n")

class MemoryCleanupCallback(TrainerCallback):
    """
    Periodic GPU memory cleanup callback to prevent fragmentation and OOM errors.
    
    This callback performs:
    - Python garbage collection (gc.collect())
    - CUDA cache clearing (torch.cuda.empty_cache())
    - Memory usage logging
    
    IMPORTANT: This callback does NOT affect training logic, only memory management.
    It runs AFTER training steps are complete, so it cannot impact gradients or weights.
    """
    
    def __init__(self, cleanup_every_n_steps=100, log_memory=True):
        """
        Args:
            cleanup_every_n_steps: How often to perform cleanup (default: 100 steps)
            log_memory: Whether to log memory usage (default: True)
        """
        self.cleanup_every_n_steps = cleanup_every_n_steps
        self.log_memory = log_memory
        self.last_cleanup_step = -1
        
        logging.info(f"✅ Memory Cleanup Callback initialized:")
        logging.info(f"   Cleanup frequency: every {cleanup_every_n_steps} steps")
        logging.info(f"   Memory logging: {log_memory}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step, AFTER gradients have been applied.
        Safe to clean memory here without affecting training.
        """
        # Only cleanup at specified intervals
        if state.global_step - self.last_cleanup_step < self.cleanup_every_n_steps:
            return
        
        self.last_cleanup_step = state.global_step
        
        try:
            if torch.cuda.is_available():
                # Log memory BEFORE cleanup
                if self.log_memory:
                    device = torch.cuda.current_device()
                    mem_before_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_before_reserved = torch.cuda.memory_reserved(device) / 1024**3
                
                # Perform cleanup
                # 1. Python garbage collection (cleans up Python objects)
                gc.collect()
                
                # 2. Clear CUDA cache (frees fragmented memory back to CUDA allocator)
                torch.cuda.empty_cache()
                
                # Log memory AFTER cleanup
                if self.log_memory:
                    mem_after_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_after_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    mem_freed = mem_before_reserved - mem_after_reserved
                    
                    logging.info(f"🧹 Memory cleanup at step {state.global_step}:")
                    logging.info(f"   Before: {mem_before_allocated:.2f} GB allocated, {mem_before_reserved:.2f} GB reserved")
                    logging.info(f"   After:  {mem_after_allocated:.2f} GB allocated, {mem_after_reserved:.2f} GB reserved")
                    if mem_freed > 0:
                        logging.info(f"   Freed:  {mem_freed:.2f} GB")
        
        except Exception as e:
            # Don't crash training if cleanup fails
            logging.warning(f"Memory cleanup failed at step {state.global_step}: {e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Also perform cleanup at the end of each epoch for extra safety.
        """
        try:
            if torch.cuda.is_available():
                logging.info(f"🧹 End-of-epoch memory cleanup")
                gc.collect()
                torch.cuda.empty_cache()
                
                if self.log_memory:
                    device = torch.cuda.current_device()
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    logging.info(f"   Memory after epoch: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
        
        except Exception as e:
            logging.warning(f"End-of-epoch memory cleanup failed: {e}")
