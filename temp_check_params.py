from safetensors import safe_open

with safe_open('/Users/lu/Documents/tt-bounty-1/qwen2.5-7b/model-00001-of-00004.safetensors', framework='numpy', device='cpu') as f:
    keys = sorted(f.keys())[:30]
    for key in keys:
        print(key) 