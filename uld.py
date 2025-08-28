import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import sys
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL = "bigscience/bloomz-560m"
ds = load_dataset("ofir408/MedConceptsQA", "icd10cm_easy")['test'].train_test_split(test_size = 0.2)
train_ds = ds['train']
test_ds = ds['test']

teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, device_map = "auto", torch_dtype = "auto")
student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, device_map = "auto", torch_dtype = "auto")
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, padding_side = "right")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, padding_side = "right")

class UniversalLogitDistillationLoss:
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_uld_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len = target_tokens.shape
        
        if attention_mask is None:
            attention_mask = torch.ones_like(target_tokens, dtype=torch.float)
        
        # Compute Cross-Entropy loss (standard language modeling loss)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            target_tokens.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        ce_loss = (ce_loss * attention_mask).sum() / attention_mask.sum()
        
        uld_loss = self._compute_uld_component(
            student_logits, teacher_logits, attention_mask
        )
        
        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * uld_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'uld_loss': uld_loss,
            'alpha': self.alpha
        }
        
        return total_loss, loss_dict
    
    def _compute_uld_component(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Apply temperature scaling
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Sort probabilities in descending order (key insight from ULD paper)
        # This makes the comparison invariant to token ordering differences
        sorted_student_probs, _ = torch.sort(student_probs, descending=True, dim=-1)
        sorted_teacher_probs, _ = torch.sort(teacher_probs, descending=True, dim=-1)
        
        # Handle vocabulary size differences by padding
        student_vocab_size = sorted_student_probs.size(-1)
        teacher_vocab_size = sorted_teacher_probs.size(-1)
        
        if student_vocab_size != teacher_vocab_size:
            vocab_size_gap = student_vocab_size - teacher_vocab_size
            
            if vocab_size_gap > 0:
                # Student vocab is larger, pad teacher
                sorted_teacher_probs = F.pad(
                    sorted_teacher_probs, 
                    (0, vocab_size_gap), 
                    value=0.0
                )
            else:
                # Teacher vocab is larger, pad student  
                sorted_student_probs = F.pad(
                    sorted_student_probs,
                    (0, -vocab_size_gap),
                    value=0.0
                )
        
        # Compute L1 distance between sorted distributions (Wasserstein-1 approximation)
        # This is the optimal transport distance between the two distributions
        position_wise_loss = torch.abs(sorted_student_probs - sorted_teacher_probs).sum(dim=-1)
        
        # Apply attention mask and compute mean
        masked_loss = position_wise_loss * attention_mask
        uld_loss = masked_loss.sum() / attention_mask.sum()
        
        return uld_loss
    


def example_usage():
    """
    Example showing how to use ULD loss for cross-tokenizer distillation
    """
    # Initialize loss function
    uld_loss_fn = UniversalLogitDistillationLoss(temperature=4.0, alpha=0.7)
    
    # Simulate different vocabulary sizes (key advantage of ULD)
    batch_size, seq_len = 2, 10
    student_vocab_size = 32000  # e.g., LLaMA tokenizer
    teacher_vocab_size = 50257  # e.g., GPT tokenizer
    
    # Generate random logits (in practice, these come from your models)
    student_logits = torch.randn(batch_size, seq_len, student_vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, teacher_vocab_size)
    target_tokens = torch.randint(0, student_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Compute ULD loss
    total_loss, loss_dict = uld_loss_fn.compute_uld_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        target_tokens=target_tokens,
        attention_mask=attention_mask
    )
    
    print("ULD Loss Results:")
    print(f"Total Loss: {total_loss:.4f}")
    print(f"CE Loss: {loss_dict['ce_loss']:.4f}")  
    print(f"ULD Loss: {loss_dict['uld_loss']:.4f}")
    print(f"Alpha: {loss_dict['alpha']:.2f}")
    
    return total_loss, loss_dict

if __name__ == "__main__":
    example_usage()