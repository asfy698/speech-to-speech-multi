#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for the llama model to become ready..."
llama_ready=false
for _attempt in $(seq 1 180); do
  if python -c 'import urllib.request; response = urllib.request.urlopen("http://llama:8080/health", timeout=2); raise SystemExit(0 if response.status == 200 else 1)' >/dev/null 2>&1; then
    llama_ready=true
    break
  fi
  sleep 2
done
if [[ "$llama_ready" != true ]]; then
  echo "Llama did not become ready within 6 minutes." >&2
  exit 1
fi
echo "Llama is ready. Starting the speech pipeline."

exec speech-to-speech \
  --mode realtime \
  --ws_host 0.0.0.0 \
  --ws_port 8765 \
  --llm_backend responses-api \
  --model_name ggml-org/gemma-4-E4B-it-GGUF \
  --responses_api_base_url http://llama:8080/v1 \
  --responses_api_api_key "" \
  --init_chat_role system \
  --init_chat_prompt "You are a helpful assistant. If the user speaks Tamil, reply in Tamil. Keep responses short and conversational." \
  --stt qwen3-asr \
  --tts indic-qwen3 \
  --qwen3_asr_model_name osmapi/tamil-asr-qwen3 \
  --qwen3_asr_language_code auto \
  --min_silence_ms 500 \
  --unanswered_reopen_ms 1500 \
  --indic_qwen3_tts_model_name aguken-ai/Qwen3-TTS-0.6B-LoRA-Finetuned-Indic-Multilingual \
  --indic_qwen3_tts_adapter tamil_female \
  --indic_qwen3_tts_language auto
