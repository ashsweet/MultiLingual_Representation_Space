"""Model loading with HuggingFace authentication and 4-bit quantization."""
import os
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from loguru import logger

from .config import Config


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (Metal) GPU")
    else:
        device = "cpu"
        logger.warning("No GPU found, using CPU (will be slow)")
    return device


class ModelLoader:
    """Handles model loading with quantization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = get_device()
    
    def authenticate(self) -> None:
        """Authenticate with HuggingFace Hub."""
        logger.info("Authenticating with HuggingFace Hub...")
        os.environ["HF_TOKEN"] = self.config.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = self.config.hf_token
        try:
            login(token=self.config.hf_token, add_to_git_credential=False)
            logger.info("Authentication successful")
        except Exception as e:
            logger.warning(f"Login call failed ({e}), but token is set in env - continuing...")
    
    def load(self) -> Tuple:
        """Load model and tokenizer."""
        if self.model is not None:
            return self.model, self.tokenizer
        
        self.authenticate()
        
        logger.info(f"Loading tokenizer for: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded")
        
        # Quantization config - only for CUDA (bitsandbytes doesn't support MPS)
        quant_config = None
        torch_dtype = torch.float16
        
        if self.device == "cuda" and self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            logger.info("Using 4-bit quantization (CUDA)")
        elif self.device == "mps":
            # MPS works best with float16
            torch_dtype = torch.float16
            logger.info("Using float16 for MPS")
        
        logger.info(f"Loading model weights on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Move to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded on: {self.model.device}")
        return self.model, self.tokenizer
    
    def unload(self) -> None:
        """Free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Model unloaded")
