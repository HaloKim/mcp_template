# mcp_template

# ax

```
python -m vllm.entrypoints.openai.api_server --model skt/A.X-4.0 --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --max-model-len 32000 --gpu-memory-utilization 0.8 --tool-call-parser ax --enable-auto-tool-choice --chat-template /workspace/hr_jd_mcp/vllm/resource/ax.jinja --tool-parser-plugin /workspace/hr_jd_mcp/vllm/resource/ax.py 
```
