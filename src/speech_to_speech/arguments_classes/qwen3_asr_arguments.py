from dataclasses import dataclass, field


@dataclass
class Qwen3ASRSTTHandlerArguments:
    qwen3_asr_model_name: str = field(
        default="osmapi/tamil-asr-qwen3",
        metadata={"help": "Hugging Face model ID or local path for Qwen3 ASR. Default is the Tamil ASR model."},
    )
    qwen3_asr_device: str = field(
        default="auto",
        metadata={"help": "Device for Qwen3 ASR. Options: 'auto', 'cuda', 'cpu'. Default is 'auto'."},
    )
    qwen3_asr_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype for Qwen3 ASR. Options: 'auto', 'float16', 'bfloat16', 'float32'."},
    )
    qwen3_asr_language_code: str = field(
        default="auto",
        metadata={
            "help": "Language hint for Qwen3 ASR. Use 'auto' for model detection; the value is also attached to transcriptions. Default is 'auto'."
        },
    )
