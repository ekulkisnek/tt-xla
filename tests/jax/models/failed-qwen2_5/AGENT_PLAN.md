# Qwen2.5-7B Tensor-Parallel Implementation Plan

## Required Context Files and Directories

- `/Users/lu/Documents/hf-transformers/src/transformers/models/llama/modeling_flax_llama.py` - Base model reference
- `/Users/lu/Documents/hf-transformers/src/transformers/models/qwen2/modeling_qwen2.py` - Qwen2 PyTorch implementation
- `/Users/lu/Documents/hf-transformers/src/transformers/modeling_flax_utils.py` - Base Flax model utilities
- `/Users/lu/Documents/hf-transformers/src/transformers/modeling_flax_pytorch_utils.py` - PyTorch to Flax conversion
- `/Users/lu/Documents/hf-transformers/examples/flax/language-modeling/run_clm_flax.py` - JAX parallelism examples
- `/Users/lu/Documents/tt-bounty-1/qwen2.5-7b/` - Pre-downloaded model files
- Flax GSPMD Guide: https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
- JAX SPMD Guide: https://jax.readthedocs.io/en/latest/spmd.html

## Project Structure

Create the following files in `tt-xla/tests/jax/models/qwen2_5/`:

1. `config.py` - Model configuration
2. `modeling_flax_qwen2.py` - Core model implementation
3. `tensor_parallel.py` - Device mesh and sharding utilities
4. `weight_loading.py` - Weight loading and conversion
5. `gsm8k_eval.py` - GSM8K evaluation
6. `__init__.py` - Package initialization

## Phase 1: Model Implementation

### 1.1 Configuration (config.py)

```python
from transformers.models.llama.configuration_llama import LlamaConfig

class FlaxQwen2Config(LlamaConfig):
    model_type = "qwen2"
    
    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=1000000.0,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_theta=rope_theta,
            **kwargs
        )
```

### 1.2 Core Model (modeling_flax_qwen2.py)

#### 1.2.1 Attention Implementation

```python
class FlaxQwen2Attention(nn.Module):
    config: FlaxQwen2Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    
    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Initialize projections with partitioning annotations
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.o_proj = nn.Dense(
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("heads", "embed"))},
        )
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        deterministic=True,
    ):
        # Logic similar to FlaxLlamaAttention in modeling_flax_llama.py
        # With tensor parallelism constraints added
```

#### 1.2.2 MLP Implementation

```python
class FlaxQwen2MLP(nn.Module):
    config: FlaxQwen2Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.gate_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("mlp", "embed"))},
        )
        
    def __call__(self, x):
        # SwiGLU activation
        x_gate = self.gate_proj(x)
        x_gate = jax.lax.with_sharding_constraint(x_gate, P('batch', None, 'model'))
        x_gate = nn.silu(x_gate)
        
        x_up = self.up_proj(x)
        x_up = jax.lax.with_sharding_constraint(x_up, P('batch', None, 'model'))
        
        x = x_gate * x_up
        x = self.down_proj(x)
        x = jax.lax.with_sharding_constraint(x, P('batch', None, 'model'))
        
        return x
```

#### 1.2.3 Full Model Implementation 

```python
class FlaxQwen2ForCausalLMModule(nn.Module):
    config: FlaxQwen2Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.model = FlaxQwen2Module(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "vocab"))},
        )
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Forward pass with tensor parallelism constraints
```

## Phase 2: Tensor Parallelism Implementation

### 2.1 Device Mesh Utilities (tensor_parallel.py)

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh
        
    Returns:
        Mesh object with named axes ('batch', 'model')
    """
    devices = jax.devices()
    total_devices = mesh_shape[0] * mesh_shape[1]
    
    # Ensure we have enough devices
    if len(devices) < total_devices:
        raise ValueError(f"Requested mesh shape {mesh_shape} requires {total_devices} devices, "
                         f"but only {len(devices)} are available")
    
    # Create the mesh
    device_mesh = jnp.array(devices[:total_devices]).reshape(mesh_shape)
    return Mesh(device_mesh, ('batch', 'model'))

def get_partition_specs():
    """
    Get appropriate partition specs for model parameters.
    
    Returns:
        Dictionary mapping parameter names to partition specs
    """
    return {
        # Embedding partitioned on hidden dimension
        'model.embed_tokens.weight': P(None, 'model'),
        
        # Attention projections
        'model.layers.*.self_attn.q_proj.kernel': P('model', None),
        'model.layers.*.self_attn.k_proj.kernel': P('model', None),
        'model.layers.*.self_attn.v_proj.kernel': P('model', None),
        'model.layers.*.self_attn.o_proj.kernel': P(None, 'model'),
        
        # MLP projections
        'model.layers.*.mlp.gate_proj.kernel': P('model', None),
        'model.layers.*.mlp.up_proj.kernel': P('model', None), 
        'model.layers.*.mlp.down_proj.kernel': P(None, 'model'),
        
        # Layer norms not sharded
        'model.layers.*.input_layernorm.weight': P(None),
        'model.norm.weight': P(None),
        
        # LM head partitioned along vocab dimension
        'lm_head.kernel': P(None, 'model'),
    }
```

### 2.2 Cross-Device Communication Primitives

```python
def cross_mesh_attention(query, key, value, attention_mask, mesh):
    """
    Attention implementation with proper cross-device communication.
    
    Args:
        query, key, value: Attention tensors
        attention_mask: Attention mask tensor
        mesh: Device mesh
        
    Returns:
        Output tensor after attention
    """
    # Apply sharding constraints to ensure proper device placement
    query = jax.lax.with_sharding_constraint(query, P('batch', 'seq', 'model', None))
    key = jax.lax.with_sharding_constraint(key, P('batch', 'seq', 'model', None))
    value = jax.lax.with_sharding_constraint(value, P('batch', 'seq', 'model', None))
    
    # Compute attention scores
    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2))
    
    # Apply mask
    if attention_mask is not None:
        scores = scores + attention_mask
    
    # Apply softmax
    attention_weights = jax.nn.softmax(scores, axis=-1)
    
    # Compute attention output
    attention_output = jnp.matmul(attention_weights, value)
    
    # All-gather results across model dim if needed for output projection
    if attention_output.shape[-2] != value.shape[-2]:
        attention_output = jax.lax.all_gather(attention_output, axis_name='model')
    
    return attention_output
```

## Phase 3: Weight Loading Implementation

### 3.1 Parameter Conversion (weight_loading.py)

```python
from flax.traverse_util import flatten_dict, unflatten_dict

def apply_parameter_partitioning(params, param_specs, mesh):
    """
    Apply partitioning specifications to parameters with pattern matching.
    
    Args:
        params: Parameter dictionary
        param_specs: Dictionary mapping parameter patterns to partition specs
        mesh: Device mesh
        
    Returns:
        Parameters with appropriate partitioning
    """
    import re
    
    # Flatten parameters for easier matching
    flat_params = flatten_dict(params)
    
    # Apply partitioning based on patterns
    sharded_params = {}
    for param_path, param in flat_params.items():
        # Convert path tuple to string for pattern matching
        path_str = '.'.join(param_path)
        
        # Find matching pattern
        matching_spec = None
        for pattern, spec in param_specs.items():
            # Convert * to regex pattern
            pattern_regex = pattern.replace('.', '\\.').replace('*', '.*')
            if re.match(pattern_regex, path_str):
                matching_spec = spec
                break
        
        # Apply partitioning if spec found
        if matching_spec is not None:
            # Apply partitioning using jax.device_put
            param = jax.device_put(param, jax.sharding.NamedSharding(mesh, matching_spec))
        else:
            # Default to replication if no spec found
            param = jax.device_put(param, jax.sharding.NamedSharding(mesh, P(None)))
        
        sharded_params[param_path] = param
    
    # Unflatten parameters back to original structure
    return unflatten_dict(sharded_params)
```

### 3.2 Model Loading Utility

```python
def create_and_load_tensor_parallel_model(
    model_path,
    mesh_shape=(1, 8),
    dtype=jnp.bfloat16,
    from_pt=True
):
    """
    Create and load a tensor-parallel Qwen2.5 model.
    
    Args:
        model_path: Path to the model weights
        mesh_shape: Shape of the device mesh (rows, cols)
        dtype: Data type for model weights
        from_pt: Whether to convert from PyTorch format
        
    Returns:
        Loaded model with tensor parallelism
    """
    # 1. Create device mesh
    mesh = create_device_mesh(mesh_shape)
    
    # 2. Define parameter partitioning
    param_partition_specs = get_partition_specs()
    
    # 3. Create and load model with tensor parallelism
    with mesh:
        # Load configuration
        config = FlaxQwen2Config.from_pretrained(model_path)
        
        # Initialize model without weights
        model = FlaxQwen2ForCausalLM(
            config,
            dtype=dtype,
            _do_init=False,
        )
        
        # Load and shard weights
        if from_pt:
            # Load from PyTorch weights
            params = FlaxQwen2ForCausalLM.from_pretrained(
                model_path,
                from_pt=True,
                _do_init=False,
                dtype=dtype,
            ).params
            
            # Apply sharding to parameters
            params = apply_parameter_partitioning(params, param_partition_specs, mesh)
            model.params = params
        else:
            # Directly load Flax weights with sharding
            model = FlaxQwen2ForCausalLM.from_pretrained(
                model_path,
                _do_init=False,
                dtype=dtype,
                param_partition_specs=param_partition_specs,
            )
    
    return model, mesh
```

## Phase 4: GSM8K Evaluation

### 4.1 Evaluation Function (gsm8k_eval.py)

```python
def evaluate_gsm8k_with_tensor_parallelism(
    model,
    tokenizer,
    dataset,
    mesh,
    max_new_tokens=512,
    batch_size=1
):
    """
    Evaluate model on GSM8K with tensor parallelism.
    
    Args:
        model: Tensor-parallel model
        tokenizer: Tokenizer for the model
        dataset: GSM8K dataset
        mesh: Device mesh
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    # Prepare batched dataset
    batched_dataset = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batched_dataset.append(batch)
    
    # Run evaluation with tensor parallelism
    results = []
    
    def generate_with_tensor_parallelism(input_ids, attention_mask):
        # Apply sharding constraints
        input_ids = jax.lax.with_sharding_constraint(input_ids, P('batch', None))
        attention_mask = jax.lax.with_sharding_constraint(attention_mask, P('batch', None))
        
        # Generate text
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Use greedy decoding
            do_sample=False,
        )
        
        return output.sequences
    
    # Run evaluation
    with mesh:
        for batch in batched_dataset:
            # Tokenize inputs
            inputs = tokenizer(batch["question"], padding=True, truncation=True, return_tensors="np")
            
            # Generate answers
            output_sequences = generate_with_tensor_parallelism(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
            # Decode outputs
            generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            
            # Extract answers and calculate metrics
            for text, reference in zip(generated_texts, batch["answer"]):
                extracted_answer = extract_answer(text)
                is_correct = check_answer(extracted_answer, reference)
                results.append({
                    "generated": text,
                    "answer": extracted_answer,
                    "reference": reference,
                    "correct": is_correct
                })
    
    # Calculate final metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
```

### 4.2 Answer Extraction Utilities

```python
def extract_answer(text):
    """
    Extract the final answer from generated text.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Extracted final answer
    """
    # Look for patterns like "The answer is X" or just the final number
    import re
    
    answer_patterns = [
        r"The answer is\s*(\d+\.?\d*)",
        r"The final answer is\s*(\d+\.?\d*)",
        r"The result is\s*(\d+\.?\d*)",
        r"Therefore, the answer is\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)$",  # Just a number at the end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    
    # If no match found, try to find any number in the last sentence
    sentences = text.split('.')
    last_sentence = sentences[-1]
    numbers = re.findall(r"(\d+\.?\d*)", last_sentence)
    if numbers:
        return float(numbers[-1])
    
    return None

def check_answer(predicted, reference):
    """
    Check if the predicted answer matches the reference.
    
    Args:
        predicted: Extracted answer from model output
        reference: Reference answer from the dataset
        
    Returns:
        Boolean indicating whether the answer is correct
    """
    if predicted is None:
        return False
    
    try:
        # Convert reference to float for numerical comparison
        ref_value = float(reference)
        return abs(predicted - ref_value) < 1e-6
    except (ValueError, TypeError):
        # If reference is not a number, do string comparison
        return str(predicted) == str(reference)
```

## Phase 5: Complete Example Integration

### 5.1 End-to-End Example

```python
def run_end_to_end_tensor_parallel_inference():
    """
    Complete end-to-end example of tensor-parallel inference with Qwen2.5-7B.
    """
    # 1. Define mesh configurations
    mesh_shapes = {
        '2x4': (2, 4),
        '1x8': (1, 8),
        '1x32': (1, 32),
        '8x4': (8, 4),
    }
    
    # 2. Choose configuration based on available devices
    available_devices = len(jax.devices())
    if available_devices >= 32:
        mesh_shape = mesh_shapes['1x32']
    elif available_devices >= 8:
        mesh_shape = mesh_shapes['1x8']
    else:
        raise ValueError(f"Not enough devices available: {available_devices}")
    
    # 3. Create mesh
    mesh = create_device_mesh(mesh_shape)
    
    # 4. Load model with tensor parallelism
    model_path = "/Users/lu/Documents/tt-bounty-1/qwen2.5-7b"
    model, _ = create_and_load_tensor_parallel_model(
        model_path, 
        mesh_shape=mesh_shape,
        dtype=jnp.bfloat16,
        from_pt=True
    )
    
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 6. Define tensor-parallel inference function
    def generate_with_tensor_parallelism(prompt, max_new_tokens=100):
        inputs = tokenizer(prompt, return_tensors="np")
        
        with mesh:
            # Apply sharding constraints
            input_ids = jax.lax.with_sharding_constraint(
                inputs["input_ids"], P('batch', None))
            
            attention_mask = None
            if "attention_mask" in inputs:
                attention_mask = jax.lax.with_sharding_constraint(
                    inputs["attention_mask"], P('batch', None))
            
            # Generate with tensor parallelism
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
            
            # Get generated text
            generated_ids = outputs.sequences
            
            # Apply final sharding constraint
            generated_ids = jax.lax.with_sharding_constraint(
                generated_ids, P('batch', None))
        
        # Decode generated text
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 7. Run inference
    prompt = "Solve the following math problem: If John has 5 apples and buys 3 more, how many does he have?"
    result = generate_with_tensor_parallelism(prompt)
    
    return result
```

## Implementation Timeline and Milestones

1. **Days 1-2: Setup and Core Model Implementation**
   - Set up development environment with JAX/Flax
   - Implement config.py and base model structure
   - Adapt LLaMA Flax code to Qwen2.5 architecture

2. **Days 3-4: Tensor Parallelism Implementation**
   - Implement device mesh utilities
   - Add parameter partitioning specifications
   - Integrate sharding constraints in forward pass

3. **Days 5-6: Weight Loading and Testing**
   - Implement weight conversion utilities
   - Test with pre-downloaded weights
   - Verify tensor shapes and parameter compatibility

4. **Days 7-8: GSM8K Evaluation**
   - Implement GSM8K evaluation logic
   - Test across different mesh configurations
   - Optimize performance and memory usage

5. **Days 9-10: Documentation and Final Testing**
   - Create comprehensive README
   - Document tensor parallelism approach
   - Finalize code and submit

## Key Implementation Checklist

- [ ] Core model structure implementation (config.py, modeling_flax_qwen2.py)
- [ ] Device mesh implementation (tensor_parallel.py)
- [ ] Weight loading utilities (weight_loading.py)
- [ ] Sharding constraints in attention and MLP
- [ ] Cross-device communication primitives
- [ ] GSM8K evaluation (gsm8k_eval.py)
- [ ] Support for all mesh configurations (2x4, 1x8, 1x32, 8x4)
- [ ] Documentation and examples

## Key Technical Insights

- Use Flax's `param_axes` for partitioning information in module definition
- Apply `jax.lax.with_sharding_constraint` to intermediate tensors during computation
- Leverage JAX's SPMD programming model for efficient parallelism
- Use pattern matching for parameter partitioning
- Apply one-time weight conversion from PyTorch to Flax for faster development
- Use appropriate partitioning strategy for different parameter types:
  - Input projections: partitioned on output dimension 
  - Output projections: partitioned on input dimension
  - Embeddings: partitioned on embedding dimension
  - Layer norms: replicated across devices 