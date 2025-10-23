#!/usr/bin/env python3
"""
Custom training callbacks for Gemma-3 Sanskrit training
"""

import logging
import torch
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