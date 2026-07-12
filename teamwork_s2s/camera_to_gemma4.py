from __future__ import annotations

import base64
import contextlib
import os
import shutil
import subprocess
import tempfile
import wave
from queue import Queue
from threading import Event
from pathlib import Path

from openai import OpenAI
from PIL import Image


def capture_photo(camera_index: int = 0, save_path: str | None = None) -> bytes:
    """Capture one frame from the webcam and return it as JPEG bytes."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        Picamera2 = None

    if Picamera2 is not None:
        try:
            picam = Picamera2(camera_num=camera_index)
        except TypeError:
            picam = Picamera2()

        try:
            config = picam.create_still_configuration()
            picam.configure(config)
            picam.start()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                picam.capture_file(str(tmp_path), format="jpeg")
                jpeg_bytes = tmp_path.read_bytes()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
            if save_path:
                Path(save_path).write_bytes(jpeg_bytes)
            return jpeg_bytes
        finally:
            with contextlib.suppress(Exception):
                picam.close()

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise SystemExit("Could not read a frame from the camera.")

                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    raise SystemExit("Could not encode the camera frame as JPEG.")

                jpeg_bytes = encoded.tobytes()

                if save_path:
                    Path(save_path).write_bytes(jpeg_bytes)

                return jpeg_bytes
            finally:
                cap.release()

    # Fall back to the Pi camera command line tools when OpenCV is unavailable
    # or when the camera backend cannot be opened from OpenCV.
    for camera_cmd in ("libcamera-still", "rpicam-still"):
        if not shutil.which(camera_cmd):
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "capture.jpg"
            command = [
                camera_cmd,
                "-n",
                "-t",
                "1",
                "--encoding",
                "jpg",
                "--quality",
                "95",
                "-o",
                str(image_path),
            ]
            if camera_index not in (0, None):
                command.extend(["--camera", str(camera_index)])

            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                continue
            if not image_path.exists():
                continue

            jpeg_bytes = image_path.read_bytes()
            if save_path:
                Path(save_path).write_bytes(jpeg_bytes)
            return jpeg_bytes

    raise SystemExit(
        "No camera backend is available. Install OpenCV or Pi camera tools "
        "(libcamera-still or rpicam-still)."
    )


def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    """Convert JPEG bytes to a base64 data URL suitable for multimodal chat input."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_gemma4_with_image(jpeg_bytes: bytes) -> str:
    """Send the captured image to a vision-capable Gemma 4 endpoint."""
    base_url = os.getenv("GEMMA4_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.getenv("GEMMA4_API_KEY", "local")
    model = "gemma-4"

    client = OpenAI(base_url=base_url, api_key=api_key)
    image_url = jpeg_bytes_to_data_url(jpeg_bytes)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful vision assistant. Look at the face in the image and infer the user's emotion "
                    "from visible facial expression, posture, and context. Reply naturally and respond in a way that "
                    "fits the emotion. Do not mention hidden chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze my emotion from this image and answer accordingly."},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ],
            },
        ],
        temperature=0.7,
    )

    message = response.choices[0].message
    return message.content or ""


def detect_emotion(text: str) -> str | None:
    lowered = text.lower()
    emotion_keywords = [
        (
            "negative",
            (
                "negative",
                "uninterested",
                "unimpressed",
                "bored",
                "tired",
                "subdued",
                "fed up",
                "done",
                "over it",
                "not interested",
                "don't care",
                "dont care",
                "unhappy",
                "sad",
                "down",
                "low",
                "depressed",
                "angry",
                "mad",
                "annoyed",
                "frustrated",
                "irritated",
                "afraid",
                "scared",
                "worried",
                "anxious",
                "nervous",
                "terrified",
                "disgusted",
                "grossed out",
                "would rather",
                "rather be",
                "waiting for this to be over",
                "anything else",
                "whatever is happening",
                "not enjoying",
                "not having fun",
                "pissed",
                "fed up",
                "sick of",
                "overwhelmed",
                "stressed",
                "stressing",
                "helpless",
                "hopeless",
                "miserable",
                "mood",
                "blah",
                "meh",
                "blahh",
                "unmotivated",
                "apathetic",
                "withdrawn",
                "flat",
                "blank",
                "drained",
                "exhausted",
                "irritated",
                "annoyed",
                "agitated",
                "dismayed",
                "discouraged",
                "disheartened",
                "discontent",
                "discontented",
                "displeased",
                "dissatisfied",
                "downcast",
                "downhearted",
                "gloomy",
                "grim",
                "grumpy",
                "hollow",
                "hurt",
                "in pain",
                "in distress",
                "jaded",
                "lonely",
                "lost",
                "mellow",
                "melancholy",
                "numb",
                "oppressed",
                "pained",
                "pessimistic",
                "pleading",
                "restless",
                "restless",
                "shut down",
                "silent",
                "sour",
                "spooked",
                "struggling",
                "tense",
                "troubled",
                "uneasy",
                "uneven",
                "unsettled",
                "upset",
                "worn out",
                "worse",
                "worst",
                "weary",
                "withdrawn",
                "withdrawal",
                "broken",
                "crushed",
                "defeated",
                "deflated",
                "despair",
                "despairing",
                "desperate",
                "dull",
                "empty",
                "fed-up",
                "flat",
                "frayed",
                "freaked out",
                "frustrated",
                "glum",
                "heavy",
                "horrible",
                "impatient",
                "insecure",
                "irate",
                "lethargic",
                "miffed",
                "moody",
                "morose",
                "nervous",
                "off",
                "offended",
                "on edge",
                "over it",
                "overwhelmed",
                "pessimism",
                "pink slip",
                "raw",
                "regretful",
                "reject",
                "rejected",
                "resentful",
                "rattled",
                "scared",
                "shaky",
                "shaken",
                "sick",
                "sickened",
                "somber",
                "stale",
                "stiff",
                "strained",
                "stressed",
                "stressing",
                "stressful",
                "stuffed up",
                "swamped",
                "tetchy",
                "threatened",
                "timid",
                "trapped",
                "trembling",
                "triggered",
                "uncomfortable",
                "uncertain",
                "uncooperative",
                "uncared for",
                "unclear",
                "undone",
                "uneasy",
                "unequipped",
                "unfriendly",
                "unhappy",
                "unmotivated",
                "unpleasant",
                "unproductive",
                "unready",
                "unresponsive",
                "up in the air",
                "vulnerable",
                "weak",
                "worried",
                "worrier",
                "wound up",
                "wretched",
                "yucky",
                "zealous",
            ),
        ),
        (
            "sad",
            (
                "sad",
                "upset",
                "unhappy",
                "depressed",
                "tired",
                "bored",
                "subdued",
                "unimpressed",
                "indifferent",
                "down",
                "low",
                "crying",
                "tearful",
                "distressed",
                "drained",
                "exhausted",
                "fed up",
                "done",
                "over it",
                "waiting",
            ),
        ),
        (
            "angry",
            (
                "angry",
                "anger",
                "mad",
                "annoyed",
                "frustrated",
                "irritated",
                "furious",
                "hostile",
                "fed up",
                "fed-up",
            ),
        ),
        (
            "fear",
            (
                "fear",
                "afraid",
                "scared",
                "frightened",
                "anxious",
                "nervous",
                "worried",
                "terrified",
                "panic",
                "panicked",
                "uneasy",
                "apprehensive",
            ),
        ),
        (
            "disgust",
            (
                "disgust",
                "disgusted",
                "grossed out",
                "repulsed",
                "revolted",
                "nauseated",
                "gross",
                "icky",
            ),
        ),
        (
            "happy",
            (
                "happy",
                "joy",
                "joyful",
                "delighted",
                "cheerful",
                "excited",
                "smiling",
                "smile",
                "thrilled",
                "glad",
            ),
        ),
        (
            "surprise",
            (
                "surprise",
                "surprised",
                "astonished",
                "shocked",
                "amazed",
                "startled",
                "stunned",
            ),
        ),
    ]

    for emotion, keywords in emotion_keywords:
        if any(word in lowered for word in keywords):
            return emotion

    negative_signals = (
        "not",
        "no",
        "never",
        "can't",
        "cant",
        "don't",
        "dont",
        "rather",
        "else",
        "over",
        "waiting",
        "boring",
        "bored",
        "tired",
        "uninterested",
        "unimpressed",
        "unhappy",
        "sad",
        "angry",
        "afraid",
        "worried",
        "stressed",
    )
    if any(signal in lowered for signal in negative_signals) and not any(
        positive in lowered for positive in ("happy", "surprise", "surprised", "joy", "smile", "smiling")
    ):
        return "negative"

    return None


def send_whatsapp_alert(reply_text: str) -> bool:
    """Send a WhatsApp alert to the parent for any tracked emotion except happy and surprise."""
    emotion = detect_emotion(reply_text)
    if emotion is None or emotion in {"happy", "surprise"}:
        return False

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "whatsapp:+14155238886").strip()
    to_number = os.getenv("ALERT_PHONE", "").strip()

    if not account_sid or not auth_token or not to_number:
        print("Twilio alert skipped: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and ALERT_PHONE.")
        return False

    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"

    default_body = f"Alert: the camera response suggests child_emotion:{emotion}. Reply: {reply_text}"
    body = os.getenv("SAD_ALERT_MESSAGE", default_body)

    try:
        import httpx

        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        response = httpx.post(
            url,
            auth=(account_sid, auth_token),
            data={"To": to_number, "From": from_number, "Body": body},
            timeout=20.0,
        )
        if response.status_code in (200, 201):
            print("WhatsApp alert sent to parent.")
            return True
        print(f"WhatsApp alert failed: {response.status_code} {response.text}")
        return False
    except Exception as exc:
        print(f"WhatsApp alert error: {exc}")
        return False


def _audio_bytes_from_tts_output(chunk: object) -> bytes:
    """Convert a Qwen3-TTS output chunk to raw int16 PCM bytes."""
    import numpy as np

    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    array = np.asarray(chunk)
    if array.dtype != np.int16:
        array = np.clip(array * 32768, -32768, 32767).astype(np.int16)
    return array.tobytes()


def speak_with_qwen3_tts(text: str) -> None:
    """Synthesize and play text using the best available repo TTS handler."""
    try:
        from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
    except ImportError:
        Qwen3TTSHandler = None  # type: ignore[assignment]

    try:
        from speech_to_speech.pipeline.messages import TTSInput
    except ImportError as exc:
        raise SystemExit("Could not import the TTS message types from the repo.") from exc

    stop_event = Event()
    queue_in: Queue[object] = Queue()
    queue_out: Queue[object] = Queue()
    should_listen = Event()
    should_listen.set()

    default_ref_audio = Path(__file__).resolve().parent.parent / "speech-to-speech-main" / "src" / "speech_to_speech" / "TTS" / "ref_audio.wav"
    ref_audio = os.getenv("QWEN3_TTS_REF_AUDIO", str(default_ref_audio))

    if Qwen3TTSHandler is None:
        raise SystemExit("Qwen 3 TTS is not available in this environment.")

    handler = Qwen3TTSHandler(
        stop_event,
        queue_in=queue_in,  # type: ignore[arg-type]
        queue_out=queue_out,  # type: ignore[arg-type]
        setup_args=(should_listen,),
        setup_kwargs={
            "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "device": "cuda",
            "dtype": "auto",
            "ref_audio": ref_audio if ref_audio else None,
            "speaker": "Aiden",
        },
    )

    tts_input = TTSInput(text=text, language_code="en")
    audio_pcm = bytearray()
    for out in handler.process(tts_input):
        audio_pcm.extend(_audio_bytes_from_tts_output(out))

    handler.cleanup()
    stop_event.set()

    if not audio_pcm:
        raise SystemExit("Qwen3 TTS produced no audio.")

    wav_path = Path(__file__).resolve().with_name("gemma4_reply.wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(bytes(audio_pcm))

    try:
        import winsound

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_SYNC)
    except Exception as exc:
        raise SystemExit(f"Saved audio to {wav_path}, but playback failed: {exc}") from exc


def main() -> int:
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    save_path = os.getenv("CAMERA_SNAPSHOT_PATH")
    skip_tts = os.getenv("CAMERA_SKIP_TTS", "").strip().lower() in {"1", "true", "yes", "on"}

    jpeg_bytes = capture_photo(camera_index=camera_index, save_path=save_path)
    print(f"Captured photo ({len(jpeg_bytes)} bytes). Sending to Gemma 4...")

    reply = call_gemma4_with_image(jpeg_bytes)
    if reply.strip():
        print("\nGemma 4 reply:\n")
        print(reply)
        send_whatsapp_alert(reply)
        if skip_tts:
            print("\nSkipping TTS and returning control to the launcher...\n")
            return 0
        print("\nSpeaking reply with Qwen 3 TTS...\n")
        speak_with_qwen3_tts(reply)
    else:
        print("\nGemma 4 returned an empty response.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
