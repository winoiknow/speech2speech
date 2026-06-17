# Copyright 2024 The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Streamlined to the remote-only realtime ParsedArguments surface.

import sys
from dataclasses import fields

from speech_to_speech.arguments_classes.elevenlabs_tts_arguments import ElevenLabsTTSHandlerArguments
from speech_to_speech.arguments_classes.minimax_tts_arguments import MiniMaxTTSHandlerArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.remote_openai_stt_arguments import RemoteOpenAISTTHandlerArguments
from speech_to_speech.arguments_classes.remote_openai_tts_arguments import RemoteOpenAITTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.s2s_pipeline import ParsedArguments, parse_arguments


def test_release_defaults_match_remote_realtime_profile():
    module_args = ModuleArguments()
    vad_args = VADHandlerArguments()
    responses_api_args = ResponsesApiLanguageModelHandlerArguments()

    assert module_args.mode == "realtime"
    assert module_args.stt == "openai-remote"
    assert module_args.llm_backend == "responses-api"
    assert module_args.tts == "openai-remote"
    assert module_args.host == "0.0.0.0"
    assert module_args.port == 8765
    assert module_args.log_level == "info"

    assert vad_args.thresh == 0.6
    assert responses_api_args.model_name == "gpt-5.4-mini"
    assert responses_api_args.chat_size == 30
    assert responses_api_args.responses_api_stream is True


# -- ParsedArguments dataclass tests ------------------------------------------

EXPECTED_FIELD_TYPES = {
    "module_kwargs": ModuleArguments,
    "vad_handler_kwargs": VADHandlerArguments,
    "responses_api_language_model_handler_kwargs": ResponsesApiLanguageModelHandlerArguments,
    "remote_openai_stt_handler_kwargs": RemoteOpenAISTTHandlerArguments,
    "remote_openai_tts_handler_kwargs": RemoteOpenAITTSHandlerArguments,
    "elevenlabs_tts_handler_kwargs": ElevenLabsTTSHandlerArguments,
    "minimax_tts_handler_kwargs": MiniMaxTTSHandlerArguments,
}


def test_parsed_arguments_has_all_expected_fields():
    actual_fields = {f.name: f.type for f in fields(ParsedArguments)}
    assert set(actual_fields) == set(EXPECTED_FIELD_TYPES)


def test_parsed_arguments_field_types_match():
    for f in fields(ParsedArguments):
        assert f.type is EXPECTED_FIELD_TYPES[f.name], (
            f"Field {f.name!r}: expected {EXPECTED_FIELD_TYPES[f.name].__name__}, got {f.type}"
        )


def test_parse_arguments_defaults_to_responses_api():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert isinstance(args, ParsedArguments)
    assert isinstance(args.module_kwargs, ModuleArguments)
    assert isinstance(args.responses_api_language_model_handler_kwargs, ResponsesApiLanguageModelHandlerArguments)
    assert args.responses_api_language_model_handler_kwargs.model_name == "gpt-5.4-mini"
    assert args.module_kwargs.llm_backend == "responses-api"


def test_parse_arguments_all_fields_populated():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    for f in fields(ParsedArguments):
        value = getattr(args, f.name)
        assert value is not None, f"Field {f.name!r} is None"
        assert isinstance(value, EXPECTED_FIELD_TYPES[f.name]), (
            f"Field {f.name!r}: expected {EXPECTED_FIELD_TYPES[f.name].__name__}, got {type(value).__name__}"
        )
