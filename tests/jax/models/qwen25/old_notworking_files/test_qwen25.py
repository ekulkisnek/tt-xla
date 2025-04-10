# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test file for Qwen2.5-7B model.
"""

import os
import pytest
from pathlib import Path

from infra import Framework, RunMode
from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelTask,
    ModelSource,
    build_model_name,
)

from .tester import Qwen25Tester, Qwen25SmallTester
from .config import supported_mesh_configs

# Find the model path relative to this file
# The model should be at ../../../../qwen2.5-7b relative to this file
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = str(SCRIPT_DIR.parent.parent.parent.parent.parent / "qwen2.5-7b")

# Define model metadata
MODEL_GROUP = ModelGroup.PRIORITY
MODEL_NAME = build_model_name(
    Framework.JAX,
    "qwen2.5",
    "7b",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.CUSTOM,
)

# Test fixtures for different configurations

@pytest.fixture
def standard_tester() -> Qwen25Tester:
    """Fixture for standard non-parallel model testing."""
    return Qwen25Tester(
        model_path=DEFAULT_MODEL_PATH,
        use_tensor_parallel=False
    )

@pytest.fixture
def small_standard_tester() -> Qwen25SmallTester:
    """Fixture for small standard non-parallel model testing."""
    return Qwen25SmallTester(
        use_tensor_parallel=False
    )

@pytest.fixture
def tp_1x8_tester() -> Qwen25Tester:
    """Fixture for tensor-parallel model with 1x8 mesh."""
    return Qwen25Tester(
        model_path=DEFAULT_MODEL_PATH,
        use_tensor_parallel=True,
        mesh_shape=(1, 8)
    )

@pytest.fixture
def small_tp_1x8_tester() -> Qwen25SmallTester:
    """Fixture for small tensor-parallel model with 1x8 mesh."""
    return Qwen25SmallTester(
        use_tensor_parallel=True,
        mesh_shape=(1, 8)
    )

@pytest.fixture
def tp_2x4_tester() -> Qwen25Tester:
    """Fixture for tensor-parallel model with 2x4 mesh."""
    return Qwen25Tester(
        model_path=DEFAULT_MODEL_PATH,
        use_tensor_parallel=True,
        mesh_shape=(2, 4)
    )

@pytest.fixture
def small_tp_2x4_tester() -> Qwen25SmallTester:
    """Fixture for small tensor-parallel model with 2x4 mesh."""
    return Qwen25SmallTester(
        use_tensor_parallel=True,
        mesh_shape=(2, 4)
    )

# Tests for standard model

@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_standard(standard_tester: Qwen25Tester):
    """Test standard (non-parallel) model."""
    standard_tester.test()

@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_small",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_small_standard(small_standard_tester: Qwen25SmallTester):
    """Test small standard (non-parallel) model."""
    small_standard_tester.test()

# Tests for tensor-parallel model with 1x8 mesh

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_tp_1x8",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_tp_1x8(tp_1x8_tester: Qwen25Tester):
    """Test tensor-parallel model with 1x8 mesh."""
    # Set environment variable for device simulation if needed
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    tp_1x8_tester.test()

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_small_tp_1x8",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_small_tp_1x8(small_tp_1x8_tester: Qwen25SmallTester):
    """Test small tensor-parallel model with 1x8 mesh."""
    # Set environment variable for device simulation if needed
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    small_tp_1x8_tester.test()

# Tests for tensor-parallel model with 2x4 mesh

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_tp_2x4",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_tp_2x4(tp_2x4_tester: Qwen25Tester):
    """Test tensor-parallel model with 2x4 mesh."""
    # Set environment variable for device simulation if needed
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    tp_2x4_tester.test()

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_small_tp_2x4",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_small_tp_2x4(small_tp_2x4_tester: Qwen25SmallTester):
    """Test small tensor-parallel model with 2x4 mesh."""
    # Set environment variable for device simulation if needed
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    small_tp_2x4_tester.test()

# Tests for larger mesh shapes (conditionally skipped if not enough devices)

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_tp_1x32",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_tp_1x32():
    """Test tensor-parallel model with 1x32 mesh."""
    # Set environment variable for device simulation
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    
    mesh_shape = (1, 32)
    try:
        tester = Qwen25SmallTester(
            use_tensor_parallel=True,
            mesh_shape=mesh_shape
        )
        tester.test()
    except ValueError as e:
        pytest.skip(f"Skipping test for mesh shape {mesh_shape}: {str(e)}")

@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=f"{MODEL_NAME}_tp_8x4",
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
)
def test_qwen25_tp_8x4():
    """Test tensor-parallel model with 8x4 mesh."""
    # Set environment variable for device simulation
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    
    mesh_shape = (8, 4)
    try:
        tester = Qwen25SmallTester(
            use_tensor_parallel=True,
            mesh_shape=mesh_shape
        )
        tester.test()
    except ValueError as e:
        pytest.skip(f"Skipping test for mesh shape {mesh_shape}: {str(e)}") 