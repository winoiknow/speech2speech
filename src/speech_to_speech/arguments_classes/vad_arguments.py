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
    input_rms_gate_far: float = field(
        default_factory=lambda: float(os.environ.get("VAD_INPUT_RMS_GATE_FAR", "400")),
        metadata={
            "help": "Gate threshold applied to the AEC RESIDUAL while the agent is speaking (far-end active), instead "
            "of the raw gate. At higher speaker volume the raw echo can exceed the raw gate, but AEC cancels echo to a "
            "low residual while the user's voice survives (it isn't in the far reference) — so during playback we gate "
            "on the residual: echo (low residual) is rejected, a real barge-in (high residual) passes. Tune with the "
            "DEBUG_MODE 'maxpass' heartbeat during an agent turn (it shows residual levels then): set above echo "
            "residual, below your barge-in residual. Set 0 to disable far gating. Env: VAD_INPUT_RMS_GATE_FAR (default 400)."
        },
    )
    far_sustain_min: int = field(
        default_factory=lambda: int(os.environ.get("VAD_FAR_SUSTAIN_MIN", "6")),
        metadata={
            "help": "While the agent speaks, a chunk only passes the far gate if at least this many of the recent "
            "far_sustain_window chunks cleared the residual gate. Echo-residual leaks are sparse transient spikes "
            "(a level gate alone can't tell a 533 echo spike from a 533 speech onset); real speech is dense. This "
            "rejects the transient and passes sustained speech. Higher = stricter (fewer phantoms, slower/possibly "
            "missed brief barge-ins). Env: VAD_FAR_SUSTAIN_MIN (default 6 of 12)."
        },
    )
    far_sustain_window: int = field(
        default_factory=lambda: int(os.environ.get("VAD_FAR_SUSTAIN_WINDOW", "12")),
        metadata={
            "help": "Rolling window (chunks, ~32ms each) over which far_sustain_min is counted. Env: "
            "VAD_FAR_SUSTAIN_WINDOW (default 12 ≈ 384ms)."
        },
    )
    turn_detection: str = field(
        default_factory=lambda: os.environ.get("TURN_DETECTION", "vad"),
        metadata={
            "help": "End-of-turn strategy: 'vad' (fixed silence_duration_ms timer, default) or 'smart_turn' "
            "(Pipecat Smart Turn v3 semantic detector — silero finds the pause, the model decides if it's really "
            "end-of-turn vs a mid-thought breath). Env: TURN_DETECTION."
        },
    )
    turn_min_silence_ms: int = field(
        default_factory=lambda: int(os.environ.get("TURN_MIN_SILENCE_MS", "300")),
        metadata={
            "help": "smart_turn only: how much silence (ms) silero waits before asking Smart Turn whether the turn is "
            "complete. Short (≈250-400) keeps turn-taking snappy since the model — not this timer — makes the real "
            "decision. Env: TURN_MIN_SILENCE_MS."
        },
    )
    turn_max_s: float = field(
        default_factory=lambda: float(os.environ.get("TURN_MAX_S", "30")),
        metadata={
            "help": "smart_turn only: ceiling (s) on a single accumulated turn. If the detector keeps saying "
            "'incomplete', the turn is force-ended at this length so it can never hang. Env: TURN_MAX_S."
        },
    )
    turn_threshold: float = field(
        default_factory=lambda: float(os.environ.get("SMART_TURN_THRESHOLD", "0.5")),
        metadata={
            "help": "smart_turn only: probability threshold above which a turn is considered complete (0-1, default "
            "0.5). Higher = wait for stronger end-of-turn evidence (fewer early cutoffs, more latency). "
            "Env: SMART_TURN_THRESHOLD."
        },
    )
    turn_hold_grace_ms: int = field(
        default_factory=lambda: int(os.environ.get("TURN_HOLD_GRACE_MS", "3000")),
        metadata={
            "help": "smart_turn only: after an 'incomplete' verdict, if the user stays silent this long (ms) the turn "
            "is finalized anyway. This is ONLY a don't-hang-forever net, NOT a turn timer — we trust Smart Turn's "
            "'incomplete' for this whole window, so it MUST sit comfortably above a natural mid-sentence thinking pause "
            "(~1.5-2.5s) or it will cut the user off (which defeats semantic endpointing). Default 3000. The downside of "
            "a larger value is only dead air on the rare occasion Smart Turn wrongly says incomplete. Env: TURN_HOLD_GRACE_MS."
        },
    )
    smart_turn_model_path: str = field(
        default_factory=lambda: os.environ.get("SMART_TURN_MODEL_PATH", "/app/models/smart-turn-v3.2-cpu.onnx"),
        metadata={
            "help": "smart_turn only: path to the Smart Turn v3 ONNX model (baked into the image at build time). "
            "Env: SMART_TURN_MODEL_PATH."
        },
    )
