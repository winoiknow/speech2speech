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
        default_factory=lambda: float(os.environ.get("VAD_INPUT_RMS_GATE", "100")),
        metadata={
            "help": "Reject mic chunks whose RAW int16 RMS is below this value before the VAD decides (the chunk is "
            "presented to silero as silence). Keyed on raw mic energy, not the AEC-cleaned signal: AEC's nonlinear "
            "suppressor collapses a real double-talk barge-in to the same low level as echo residual, so only the raw "
            "mic keeps the user's voice (~140-2000+) above speaker echo (~6-40). This kills phantom barge-ins on echo "
            "leaks while genuine speech sails over. Tune with the DEBUG_MODE 'maxpass' heartbeat (loudest raw chunk that "
            "cleared the gate): set it above the echo maxpass, below your speech level. Set 0 to disable. "
            "Env: VAD_INPUT_RMS_GATE (default 100)."
        },
    )
