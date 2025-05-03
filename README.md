# Qwen3 Project
exploration on latest Qwen3 state oof the art models


#### uv installation
see here https://docs.astral.sh/uv/getting-started/installation/

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### create virtual environment
```
uv venv --seeds
source .venv/bin/activate
```
#### chatbot install dependencies 
```
uv pip install transformers 
uv pip install --upgrade transformers accelerate bitsandbytes
uv pip install torch
uv pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"
```

#### mcp agentic dependencies 

install ollama see here:  https://ollama.com/download

```
ollama pull qwen3:1.7b
```

## Think Mode Usage

For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0
For non-thinking mode, use Temperature=0.7, TopP=0.8, TopK=20, and MinP=0

generation_config.json
(https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/generation_config.json)
```
   {
     "bos_token_id": 151643,
     "do_sample": true,
     "eos_token_id": [
         151645,
         151643
     ],
     "pad_token_id": 151643,
     "temperature": 0.6,
     "top_k": 20,
     "top_p": 0.95,
     "transformers_version": "4.51.0"
     }
```

```
uv run think_mode.py
```
## Chatbot 
```
uv run chatbot_cli.py
```

## MCP Agentic CLI
```
uv run mcp_agentic_cli.py
```

## MCP Agentic Web App
```
uv run mcp_agentic_app.py
```

