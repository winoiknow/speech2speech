# Copyright 2024 The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Streamlined to a remote-only realtime build: STT via openai-remote, TTS via
# openai-remote/elevenlabs/minimax, LLM via responses-api. The local in-process
# model handlers and the non-realtime (local/socket/websocket) modes were removed.

import logging
import os
import signal
import sys
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event
from types import FrameType
from typing import Any

import nltk
import torch
from rich.console import Console
from transformers import HfArgumentParser

from speech_to_speech.arguments_classes.elevenlabs_tts_arguments import ElevenLabsTTSHandlerArguments
from speech_to_speech.arguments_classes.minimax_tts_arguments import MiniMaxTTSHandlerArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.remote_openai_stt_arguments import RemoteOpenAISTTHandlerArguments
from speech_to_speech.arguments_classes.remote_openai_tts_arguments import RemoteOpenAITTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
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
    vad_handler_kwargs: VADHandlerArguments
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments
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
    parser = HfArgumentParser(
        (  # type: ignore[arg-type]
            ModuleArguments,
            VADHandlerArguments,
            ResponsesApiLanguageModelHandlerArguments,
            RemoteOpenAISTTHandlerArguments,
            RemoteOpenAITTSHandlerArguments,
            ElevenLabsTTSHandlerArguments,
            MiniMaxTTSHandlerArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        parsed = parser.parse_args_into_dataclasses()

    # Build a {type: instance} lookup so field assignment is order-independent.
    by_type: dict[type, Any] = {type(obj): obj for obj in parsed}

    return ParsedArguments(
        module_kwargs=by_type[ModuleArguments],
        vad_handler_kwargs=by_type[VADHandlerArguments],
        responses_api_language_model_handler_kwargs=by_type[ResponsesApiLanguageModelHandlerArguments],
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


def prepare_all_args(
    module_kwargs: ModuleArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    remote_openai_stt_handler_kwargs: RemoteOpenAISTTHandlerArguments,
    remote_openai_tts_handler_kwargs: RemoteOpenAITTSHandlerArguments,
    elevenlabs_tts_handler_kwargs: ElevenLabsTTSHandlerArguments,
    minimax_tts_handler_kwargs: MiniMaxTTSHandlerArguments,
) -> None:
    # Remote-only defaults (the only valid STT/TTS are the remote handlers).
    if module_kwargs.stt is None:
        module_kwargs.stt = "openai-remote"
    if module_kwargs.tts is None:
        module_kwargs.tts = "openai-remote"

    rename_args(responses_api_language_model_handler_kwargs, "responses_api")
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
    remote_openai_stt_handler_kwargs: RemoteOpenAISTTHandlerArguments,
    speaker_client: Any | None = None,
    speaker_timeout: float = 0.8,
    diarize_enabled: bool = False,
) -> BaseHandler[STTIn, STTOut]:
    if module_kwargs.stt != "openai-remote":
        raise ValueError(f"Unsupported STT {module_kwargs.stt!r}; this build only supports 'openai-remote'.")

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


def get_llm_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
) -> BaseHandler[LLMIn, LLMOut]:
    if module_kwargs.llm_backend != "responses-api":
        raise ValueError(
            f"Unsupported LLM backend {module_kwargs.llm_backend!r}; this build only supports 'responses-api'."
        )

    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

    return ResponsesApiModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=vars(responses_api_language_model_handler_kwargs),
    )


def get_tts_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    lm_response_queue: Queue[TTSInItem],
    send_audio_chunks_queue: Queue[AudioOutItem],
    should_listen: Event,
    remote_openai_tts_handler_kwargs: RemoteOpenAITTSHandlerArguments,
    elevenlabs_tts_handler_kwargs: ElevenLabsTTSHandlerArguments,
    minimax_tts_handler_kwargs: MiniMaxTTSHandlerArguments,
) -> BaseHandler[TTSIn, TTSOut]:
    if module_kwargs.tts == "openai-remote":
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
            f"Unsupported TTS {module_kwargs.tts!r}; this build supports 'openai-remote', 'elevenlabs', or 'minimax'."
        )


def main() -> None:
    args = parse_arguments()

    setup_logger(args.module_kwargs.log_level)

    prepare_all_args(
        args.module_kwargs,
        args.responses_api_language_model_handler_kwargs,
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

    # Remote-only realtime build: the shared server is built once at startup; each
    # WebSocket connection builds + tears down its own SessionPipeline. Run the
    # server under a ThreadManager for the start/stop/wait + signal-handling path.
    pipeline_manager: Any = ThreadManager([factory.build_realtime_server()])

    # Set up graceful shutdown handler
    shutdown_requested = [False]  # Use list for nonlocal mutation

    def signal_handler(_sig: int, _frame: FrameType | None) -> None:
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
