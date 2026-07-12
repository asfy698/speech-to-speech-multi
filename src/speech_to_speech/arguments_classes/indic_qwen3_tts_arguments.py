from dataclasses import dataclass, field


@dataclass
class IndicQwen3TTSHandlerArguments:
    indic_qwen3_tts_model_name: str = field(
        default="aguken-ai/Qwen3-TTS-0.6B-LoRA-Finetuned-Indic-Multilingual",
        metadata={"help": "Indic Qwen3-TTS LoRA model repo or local path."},
    )
    indic_qwen3_tts_adapter: str = field(
        default="tamil_female",
        metadata={"help": "Adapter folder under adapters/. For Tamil use 'tamil_female' or 'tamil_male'."},
    )
    indic_qwen3_tts_language: str = field(
        default="Tamil",
        metadata={"help": "Language name passed to Qwen3-TTS generation. Default is 'Tamil'."},
    )
    indic_qwen3_tts_device: str = field(
        default="cuda",
        metadata={"help": "Device for Indic Qwen3-TTS. Options: 'cuda', 'cpu', 'auto'. Default is 'cuda'."},
    )
    indic_qwen3_tts_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype. Options: 'auto', 'float16', 'bfloat16', 'float32'. Default is 'auto'."},
    )
    indic_qwen3_tts_attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation for transformers/qwen-tts. Default is 'eager'."},
    )
    indic_qwen3_tts_lora_scale: float = field(
        default=1.0,
        metadata={"help": "LoRA adapter scale, if supported by the loaded PEFT model. Default is 1.0."},
    )
    indic_qwen3_tts_output_speed: float = field(
        default=1.0,
        metadata={"help": "Optional local playback speed multiplier after synthesis. Default is 1.0."},
    )
