# Copyright 2024 TensorTrace Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen2.5 implementation for JAX/Flax with tensor parallelism"""

from .configuration_qwen2_5 import Qwen25Config
from .modeling_flax_qwen2_5 import (
    FlaxQwen25Model,
    FlaxQwen25ForCausalLM,
    FlaxQwen25PreTrainedModel,
)
from .tensor_parallel import (
    create_device_mesh,
    get_partition_specs,
    FlaxQwen25WithSharding,
)
from .weight_loading import convert_qwen25_checkpoint

__all__ = [
    "Qwen25Config",
    "FlaxQwen25Model",
    "FlaxQwen25ForCausalLM",
    "FlaxQwen25PreTrainedModel",
    "create_device_mesh",
    "get_partition_specs",
    "FlaxQwen25WithSharding",
    "convert_qwen25_checkpoint",
] 