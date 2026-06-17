# Copyright 2024 The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Modified: added RemoteOpenAISTTHandler and RemoteOpenAITTSHandler dispatch,
#           CLI registration, and cancel_scope injection for remote TTS.

import argparse
import json
import logging
import os
import signal
import sys
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from sys import platform
from threading import Event
from types import FrameType
from typing import Any, Optional

import nltk
import torch
from rich.console import Console
from transformers import HfArgumentParser

from speech_to_speech.arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from speech_to_speech.arguments_classes.elevenlabs_tts_arguments import ElevenLabsTTSHandlerArguments
from speech_to_speech.arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import (
    FasterWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
from speech_to_speech.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from speech_to_speech.arguments_classes.minimax_tts_arguments import MiniMaxTTSHandlerArguments
from speech_to_speech.arguments_classes.mlx_audio_whisper_arguments import (
    MLXAudioWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import (
    ParakeetTDTSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.remote_openai_stt_arguments import RemoteOpenAISTTHandlerArguments
from speech_to_speech.arguments_classes.remote_openai_tts_arguments import RemoteOpenAITTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from speech_to_speech.arguments_classes.socket_sender_arguments import SocketSenderArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.arguments_classes.websocket_streamer_arguments import WebSocketStreamerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut, STTIn, STTOut, TTSIn, TTSOut
from speech_to_speech.pipeline.queue_types import (
    AudioInItem,
    AudioOutItem,
    LMOutItem,
    STTOutItem,
    TextEventItem,
    TextPromptItem,
    TTSInItem,
    VADOutItem,
)

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)  # quiet down numba logs


@dataclass
class ParsedArguments:
    module_kwargs: ModuleArguments
    socket_receiver_kwargs: SocketReceiverArguments
    socket_sender_kwargs: SocketSenderArguments
    websocket_streamer_kwargs: WebSocketStreamerArguments
    vad_handler_kwargs: VADHandlerArguments
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments
    language_model_handler_kwargs: LanguageModelHandlerArguments
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments
    chat_tts_handler_kwargs: ChatTTSHandlerArguments
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments
    remote_openai_stt_handler_kwargs: RemoteOpenAISTTHandlerArguments
    remote_openai_tts_handler_kwargs: RemoteOpenAITTSHandlerArguments
    elevenlabs_tts_handler_kwargs: ElevenLabsTTSHandlerArguments
    minimax_tts_handler_kwargs: MiniMaxTTSHandlerArguments


def rename_args(args: Any, prefix: str) -> None:
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def parse_arguments() -> ParsedArguments:
    # Pre-parse to determine which LM backend is selected, so only one of the two
    # mutually exclusive LM argument classes is registered with HfArgumentParser
    # (avoids duplicate field names from the shared LanguageModelBaseArguments base).
    _is_json = len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    if _is_json:
        with open(sys.argv[1]) as _f:
            _use_responses_api = json.load(_f).get("llm_backend") == "responses-api"
    else:
        _pre = argparse.ArgumentParser(add_help=False)
        _pre.add_argument("--llm_backend", default="responses-api")
        _use_responses_api = _pre.parse_known_args()[0].llm_backend == "responses-api"

    _lm_class = ResponsesApiLanguageModelHandlerArguments if _use_responses_api else LanguageModelHandlerArguments
    logger.debug("LLM backend pre-parse: use_responses_api=%s, registering %s", _use_responses_api, _lm_class.__name__)

    parser = HfArgumentParser(
        (  # type: ignore[arg-type]
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            WebSocketStreamerArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParaformerSTTHandlerArguments,
            FasterWhisperSTTHandlerArguments,
            MLXAudioWhisperSTTHandlerArguments,
            ParakeetTDTSTTHandlerArguments,
            _lm_class,
            ChatTTSHandlerArguments,
            FacebookMMSTTSHandlerArguments,
            PocketTTSHandlerArguments,
            KokoroTTSHandlerArguments,
            Qwen3TTSHandlerArguments,
            RemoteOpenAISTTHandlerArguments,
            RemoteOpenAITTSHandlerArguments,
            ElevenLabsTTSHandlerArguments,
            MiniMaxTTSHandlerArguments,
        )
    )

    if _is_json:
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        parsed = parser.parse_args_into_dataclasses()

    # Build a {type: instance} lookup so field assignment is order-independent.
    by_type: dict[type, Any] = {type(obj): obj for obj in parsed}
    logger.debug("Parsed %d argument classes: %s", len(by_type), [t.__name__ for t in by_type])

    return ParsedArguments(
        module_kwargs=by_type[ModuleArguments],
        socket_receiver_kwargs=by_type[SocketReceiverArguments],
        socket_sender_kwargs=by_type[SocketSenderArguments],
        websocket_streamer_kwargs=by_type[WebSocketStreamerArguments],
        vad_handler_kwargs=by_type[VADHandlerArguments],
        whisper_stt_handler_kwargs=by_type[WhisperSTTHandlerArguments],
        paraformer_stt_handler_kwargs=by_type[ParaformerSTTHandlerArguments],
        faster_whisper_stt_handler_kwargs=by_type[FasterWhisperSTTHandlerArguments],
        mlx_audio_whisper_stt_handler_kwargs=by_type[MLXAudioWhisperSTTHandlerArguments],
        parakeet_tdt_stt_handler_kwargs=by_type[ParakeetTDTSTTHandlerArguments],
        language_model_handler_kwargs=by_type.get(LanguageModelHandlerArguments, LanguageModelHandlerArguments()),
        responses_api_language_model_handler_kwargs=by_type.get(
            ResponsesApiLanguageModelHandlerArguments, ResponsesApiLanguageModelHandlerArguments()
        ),
        chat_tts_handler_kwargs=by_type[ChatTTSHandlerArguments],
        facebook_mms_tts_handler_kwargs=by_type[FacebookMMSTTSHandlerArguments],
        pocket_tts_handler_kwargs=by_type[PocketTTSHandlerArguments],
        kokoro_tts_handler_kwargs=by_type[KokoroTTSHandlerArguments],
        qwen3_tts_handler_kwargs=by_type[Qwen3TTSHandlerArguments],
        remote_openai_stt_handler_kwargs=by_type[RemoteOpenAISTTHandlerArguments],
        remote_openai_tts_handler_kwargs=by_type[RemoteOpenAITTSHandlerArguments],
        elevenlabs_tts_handler_kwargs=by_type[ElevenLabsTTSHandlerArguments],
        minimax_tts_handler_kwargs=by_type[MiniMaxTTSHandlerArguments],
    )


def setup_logger(log_level: str) -> None:
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)


MLX_DEFAULT_LM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-bf16"
TRANSFORMERS_DEFAULT_LM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def optimal_mac_settings(mac_optimal_settings: bool, *handler_kwargs: Any) -> None:
    if mac_optimal_settings:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "device"):
                kwargs.device = "mps"
            if hasattr(kwargs, "mode"):
                kwargs.mode = "local"
            if hasattr(kwargs, "stt"):
                kwargs.stt = "parakeet-tdt"
            if hasattr(kwargs, "llm_backend"):
                kwargs.llm_backend = "mlx-lm"
            if hasattr(kwargs, "tts"):
                kwargs.tts = "qwen3"
            if hasattr(kwargs, "model_name"):
                if kwargs.model_name == TRANSFORMERS_DEFAULT_LM_MODEL:
                    kwargs.model_name = MLX_DEFAULT_LM_MODEL


def check_mac_settings(module_kwargs: ModuleArguments) -> None:
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError("Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'.")
        if module_kwargs.llm_backend != "mlx-lm":
            logger.warning(
                "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm_backend mlx-lm."
            )
        if module_kwargs.tts not in ("pocket", "kokoro", "qwen3"):
            logger.warning(
                "For macOS users, it is recommended to use qwen3 for TTS (pocket and kokoro are also valid options)."
            )


def overwrite_device_argument(common_device: Optional[str], *handler_kwargs: Any) -> None:
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "llm_device"):
                kwargs.llm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device
            if hasattr(kwargs, "paraformer_stt_device"):
                kwargs.paraformer_stt_device = common_device
            if hasattr(kwargs, "facebook_mms_device"):
                kwargs.facebook_mms_device = common_device
            if hasattr(kwargs, "qwen3_tts_device"):
                kwargs.qwen3_tts_device = common_device


def prepare_module_args(module_kwargs: ModuleArguments, *handler_kwargs: Any) -> None:
    optimal_mac_settings(module_kwargs.local_mac_optimal_settings, module_kwargs, *handler_kwargs)
    if module_kwargs.tts is None:
        module_kwargs.tts = "qwen3"
    if module_kwargs.stt is None:
        module_kwargs.stt = "parakeet-tdt"
    if platform == "darwin":
        check_mac_settings(module_kwargs)
    overwrite_device_argument(module_kwargs.device, *handler_kwargs)


def prepare_all_args(
    module_kwargs: ModuleArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
    remote_openai_stt_handler_kwargs: RemoteOpenAISTTHandlerArguments,
    remote_openai_tts_handler_kwargs: RemoteOpenAITTSHandlerArguments,
    elevenlabs_tts_handler_kwargs: ElevenLabsTTSHandlerArguments,
    minimax_tts_handler_kwargs: MiniMaxTTSHandlerArguments,
) -> None:
    prepare_module_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        mlx_audio_whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs,
        language_model_handler_kwargs,
        responses_api_language_model_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
        pocket_tts_handler_kwargs,
        kokoro_tts_handler_kwargs,
        qwen3_tts_handler_kwargs,
        remote_openai_stt_handler_kwargs,
        remote_openai_tts_handler_kwargs,
        elevenlabs_tts_handler_kwargs,
        minimax_tts_handler_kwargs,
    )

    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(faster_whisper_stt_handler_kwargs, "faster_whisper_stt")
    rename_args(paraformer_stt_handler_kwargs, "paraformer_stt")
    rename_args(mlx_audio_whisper_stt_handler_kwargs, "mlx_audio_whisper")
    rename_args(parakeet_tdt_stt_handler_kwargs, "parakeet_tdt")
    rename_args(language_model_handler_kwargs, "llm")
    rename_args(responses_api_language_model_handler_kwargs, "responses_api")
    rename_args(chat_tts_handler_kwargs, "chat_tts")
    rename_args(facebook_mms_tts_handler_kwargs, "facebook_mms")
    rename_args(pocket_tts_handler_kwargs, "pocket_tts")
    rename_args(kokoro_tts_handler_kwargs, "kokoro")
    rename_args(qwen3_tts_handler_kwargs, "qwen3_tts")
    rename_args(remote_openai_stt_handler_kwargs, "stt_openai")
    rename_args(remote_openai_tts_handler_kwargs, "tts_openai")
    rename_args(elevenlabs_tts_handler_kwargs, "tts_elevenlabs")
    rename_args(minimax_tts_handler_kwargs, "tts_minimax")


def initialize_queues_and_events() -> dict[str, Any]:
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "response_playing": Event(),
        "cancel_scope": CancelScope(),
        "recv_audio_chunks_queue": Queue[AudioInItem](),
        "send_audio_chunks_queue": Queue[AudioOutItem](),
        "spoken_prompt_queue": Queue[VADOutItem](),
        "stt_output_queue": Queue[STTOutItem](),
        "text_prompt_queue": Queue[TextPromptItem](),
        "lm_response_queue": Queue[LMOutItem](),
        "lm_processed_queue": Queue[TTSInItem](),  # NEW: LLM -> LM processor -> TTS
        "text_output_queue": Queue[TextEventItem](),  # NEW: for text messages to WebSocket
    }


def get_stt_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    spoken_prompt_queue: Queue[VADOutItem],
    text_prompt_queue: Queue[STTOutItem],
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    remote_openai_stt_handler_kwargs: RemoteOpenAISTTHandlerArguments | None = None,
    speaker_client: Any | None = None,
    speaker_timeout: float = 0.8,
    diarize_enabled: bool = False,
) -> BaseHandler[STTIn, STTOut]:
    if module_kwargs.stt == "whisper":
        from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler

        return WhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "whisper-mlx":
        from speech_to_speech.STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler

        return LightningWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "mlx-audio-whisper":
        from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

        # Merge MLX Audio Whisper kwargs with shared language parameter from Whisper kwargs
        setup_kwargs = {**vars(mlx_audio_whisper_stt_handler_kwargs), "language": whisper_stt_handler_kwargs.language}
        return MLXAudioWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=setup_kwargs,
        )
    elif module_kwargs.stt == "paraformer":
        from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

        return ParaformerSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(paraformer_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "faster-whisper":
        from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

        return FasterWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(faster_whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "parakeet-tdt":
        from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

        # Add live transcription parameters to setup_kwargs
        setup_kwargs = {
            **vars(parakeet_tdt_stt_handler_kwargs),
            "enable_live_transcription": module_kwargs.enable_live_transcription,
            "live_transcription_update_interval": module_kwargs.live_transcription_update_interval,
        }

        return ParakeetTDTSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=setup_kwargs,
        )
    elif module_kwargs.stt == "openai-remote":
        from speech_to_speech.STT.remote_openai_stt_handler import RemoteOpenAISTTHandler

        setup_kwargs = vars(remote_openai_stt_handler_kwargs)
        if speaker_client is not None:  # concurrent speaker identify (Phase 3)
            setup_kwargs = {**setup_kwargs, "speaker_client": speaker_client, "speaker_timeout": speaker_timeout}
        if diarize_enabled:  # carry the turn audio forward for off-hot-path diarize (Phase 4)
            setup_kwargs = {**setup_kwargs, "diarize_enabled": True}
        return RemoteOpenAISTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=setup_kwargs,
        )
    else:
        raise ValueError(
            "The STT should be either whisper, whisper-mlx, mlx-audio-whisper, faster-whisper, "
            "parakeet-tdt, paraformer, or openai-remote."
        )


def get_llm_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
) -> BaseHandler[LLMIn, LLMOut]:
    if module_kwargs.llm_backend == "responses-api":
        from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

        return ResponsesApiModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(responses_api_language_model_handler_kwargs),
        )

    if module_kwargs.llm_backend in ("transformers", "mlx-lm"):
        lm_kwargs = vars(language_model_handler_kwargs)
        is_vlm = lm_kwargs.pop("is_vlm", False)
        if module_kwargs.llm_backend == "mlx-lm":
            lm_kwargs["backend"] = "mlx"

        if is_vlm:
            from speech_to_speech.LLM.language_model import VisionLanguageModelHandler

            return VisionLanguageModelHandler(
                stop_event,
                queue_in=text_prompt_queue,
                queue_out=lm_response_queue,
                setup_kwargs=lm_kwargs,
            )
        from speech_to_speech.LLM.language_model import LanguageModelHandler

        return LanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=lm_kwargs,
        )

    raise ValueError("The LLM should be either transformers, mlx-lm or responses-api")


def get_tts_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    lm_response_queue: Queue[TTSInItem],
    send_audio_chunks_queue: Queue[AudioOutItem],
    should_listen: Event,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
    remote_openai_tts_handler_kwargs: RemoteOpenAITTSHandlerArguments | None = None,
    elevenlabs_tts_handler_kwargs: ElevenLabsTTSHandlerArguments | None = None,
    minimax_tts_handler_kwargs: MiniMaxTTSHandlerArguments | None = None,
) -> BaseHandler[TTSIn, TTSOut]:
    if module_kwargs.tts == "chatTTS":
        try:
            from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler
        except (ImportError, RuntimeError) as e:
            logger.error('Error importing ChatTTSHandler. Install it with `pip install "speech-to-speech[chattts]"`.')
            raise e
        return ChatTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(chat_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "facebookMMS":
        from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

        return FacebookMMSTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(facebook_mms_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "pocket":
        try:
            from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler
        except ImportError as e:
            raise ImportError(
                'Pocket TTS is optional. Install it with `pip install "speech-to-speech[pocket]"`.'
            ) from e

        return PocketTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(pocket_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "kokoro":
        try:
            from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
        except ImportError as e:
            raise ImportError('Kokoro is optional. Install it with `pip install "speech-to-speech[kokoro]"`.') from e

        return KokoroTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(kokoro_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "qwen3":
        from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

        return Qwen3TTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(qwen3_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "openai-remote":
        from speech_to_speech.TTS.remote_openai_tts_handler import RemoteOpenAITTSHandler

        return RemoteOpenAITTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(remote_openai_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "elevenlabs":
        from speech_to_speech.TTS.elevenlabs_tts_handler import ElevenLabsTTSHandler

        return ElevenLabsTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(elevenlabs_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "minimax":
        from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler

        return MiniMaxTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(minimax_tts_handler_kwargs),
        )
    else:
        raise ValueError(
            "The TTS should be either chatTTS, facebookMMS, pocket, kokoro, qwen3, "
            "openai-remote, elevenlabs, or minimax"
        )


def main() -> None:
    args = parse_arguments()

    setup_logger(args.module_kwargs.log_level)

    prepare_all_args(
        args.module_kwargs,
        args.whisper_stt_handler_kwargs,
        args.paraformer_stt_handler_kwargs,
        args.faster_whisper_stt_handler_kwargs,
        args.mlx_audio_whisper_stt_handler_kwargs,
        args.parakeet_tdt_stt_handler_kwargs,
        args.language_model_handler_kwargs,
        args.responses_api_language_model_handler_kwargs,
        args.chat_tts_handler_kwargs,
        args.facebook_mms_tts_handler_kwargs,
        args.pocket_tts_handler_kwargs,
        args.kokoro_tts_handler_kwargs,
        args.qwen3_tts_handler_kwargs,
        args.remote_openai_stt_handler_kwargs,
        args.remote_openai_tts_handler_kwargs,
        args.elevenlabs_tts_handler_kwargs,
        args.minimax_tts_handler_kwargs,
    )

    # One HandlerFactory captures the prepared args. Imported here (not at module
    # top) to avoid a circular import — session_pipeline reuses get_*_handler /
    # initialize_queues_and_events from this module.
    from speech_to_speech.pipeline.session_pipeline import HandlerFactory
    from speech_to_speech.utils.thread_manager import ThreadManager

    factory = HandlerFactory(args)

    if args.module_kwargs.mode == "realtime":
        # Phase B: realtime builds only the shared server at startup; each
        # WebSocket connection builds + tears down its own SessionPipeline. Run
        # the server under a ThreadManager so the existing start/stop/wait +
        # signal-handling path is unchanged.
        pipeline_manager: Any = ThreadManager([factory.build_realtime_server()])
    else:
        # local / websocket / socket modes keep one startup-built pipeline.
        pipeline_manager = factory.build()

    # Set up graceful shutdown handler
    shutdown_requested = [False]  # Use list for nonlocal mutation

    def signal_handler(_sig: int, _frame: Optional[FrameType]) -> None:
        if not shutdown_requested[0]:
            shutdown_requested[0] = True
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        pipeline_manager.start()
        pipeline_manager.wait()
    except KeyboardInterrupt:
        if not shutdown_requested[0]:
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")


if __name__ == "__main__":
    main()
