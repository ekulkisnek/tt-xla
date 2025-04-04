# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Registration utilities for Qwen2.5-7B model.
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def get_model_metadata():
    """Return metadata for the model."""
    return {
        "name": "Qwen2.5-7B",
        "description": "Qwen2.5-7B transformer language model with tensor parallelism",
        "model_type": "causal_lm",
        "parameters": "7B",
        "link": "https://huggingface.co/Qwen/Qwen2.5-7B",
    }

def register_model_factory(registry=None):
    """Register the model factory with the registry."""
    # Placeholder function
    return True

def get_model_example():
    """Return an example of how to use the model."""
    return """
    # Import the model
    from tt_xla.tests.jax.models.qwen2_5 import get_model
    
    # Create a standard model
    model = get_model(model_type="qwen2_5", use_tensor_parallel=False)
    
    # Create a tensor-parallel model
    tp_model = get_model(model_type="qwen2_5", use_tensor_parallel=True, mesh_shape=(1, 8))
    """ 