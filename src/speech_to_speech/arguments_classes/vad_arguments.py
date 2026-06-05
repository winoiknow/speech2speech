import os
from dataclasses import dataclass, field


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.6,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        },
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        },
    )
    min_silence_ms: int = field(
        default=300,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 250 ms."
        },
    )
    min_speech_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        },
    )
    max_speech_ms: float = field(
        default=float("inf"),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        },
    )
    speech_pad_ms: int = field(
        default=500,
        metadata={
            "help": "Amount of audio retained before VAD triggers and prepended to detected speech segments. Once speech is detected, audio continues to be kept until VAD declares the segment done. Measured in milliseconds. Default is 500 ms."
        },
    )
    audio_enhancement: bool = field(
        default=False,
        metadata={
            "help": "improves sound quality by applying techniques like noise reduction, equalization, and echo cancellation. Default is False."
        },
    )
    enable_realtime_transcription: bool = field(
        default=False,
        metadata={"help": "Enable progressive audio release for live transcription during speech. Default is False."},
    )
    realtime_processing_pause: float = field(
        default=0.2,
        metadata={
            "help": "Interval (in seconds) for releasing progressive audio chunks during speech. Default is 0.2s."
        },
    )
    input_rms_gate: float = field(
        default_factory=lambda: float(os.environ.get("VAD_INPUT_RMS_GATE", "40")),
        metadata={
            "help": "Reject mic chunks whose int16 RMS is below this value before the VAD sees them (presented as silence). "
            "The VAD runs on the AEC-cleaned signal, where echo residual sits at very low RMS (~2-18) while real speech is "
            "far louder (hundreds+), so this gate kills phantom barge-ins on echo residual without affecting genuine speech. "
            "Set 0 to disable. Env: VAD_INPUT_RMS_GATE (default 40)."
        },
    )
