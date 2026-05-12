# Cosmos3 vLLM Plugin

See [Cosmos-Reason2](https://github.com/nvidia-cosmos/cosmos-reason2) for inference examples.

```shell
VLLM_USE_DEEP_GEMM=0 uvx --torch-backend=auto --with-editable ./vllm-cosmos3 vllm@latest serve nvidia/Cosmos3-Nano-Internal \
  --revision spectralflight/vllm-shim \
  --trust-remote-code \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
```
