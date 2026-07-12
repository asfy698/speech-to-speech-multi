#!/usr/bin/env bash
set -euo pipefail

exec speech-to-speech \
  --mode realtime \
  --ws_host 0.0.0.0 \
  --ws_port 8765 \
  --llm_backend responses-api \
  --model_name ggml-org/gemma-4-E4B-it-GGUF \
  --responses_api_base_url http://llama:8080/v1 \
  --responses_api_api_key "" \
  --init_chat_role system \
  --init_chat_prompt "You are a helpful assistant. Keep responses short and conversational." \
  --stt parakeet-tdt \
  --tts qwen3 \
  --qwen3_tts_language auto
