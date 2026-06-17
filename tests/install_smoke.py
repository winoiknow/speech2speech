from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess


def _require_modules(modules: list[str]) -> None:
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(f"Missing expected install-time modules: {', '.join(missing)}")


def _run_installed_cli_help() -> None:
    env = {**os.environ, "OPENAI_API_KEY": ""}
    result = subprocess.run(
        ["speech-to-speech", "--help"],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected_flags = ("--mode", "--stt", "--llm_backend", "--tts")
    missing_flags = [flag for flag in expected_flags if flag not in result.stdout]
    if missing_flags:
        raise RuntimeError(f"Installed CLI --help is missing flags: {', '.join(missing_flags)}")


def _validate_package_defaults() -> None:
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments

    module_args = ModuleArguments()
    assert module_args.mode == "realtime", module_args.mode
    assert module_args.stt == "openai-remote", module_args.stt
    assert module_args.llm_backend == "responses-api", module_args.llm_backend
    assert module_args.tts == "openai-remote", module_args.tts
    assert module_args.host == "0.0.0.0", module_args.host
    assert module_args.port == 8765, module_args.port


def _validate_pipeline_startup_primitives() -> None:
    from speech_to_speech.s2s_pipeline import initialize_queues_and_events

    queues_and_events = initialize_queues_and_events()
    expected_keys = {
        "recv_audio_chunks_queue",
        "send_audio_chunks_queue",
        "spoken_prompt_queue",
        "stt_output_queue",
        "text_prompt_queue",
        "lm_response_queue",
        "lm_processed_queue",
        "text_output_queue",
    }
    missing_keys = expected_keys.difference(queues_and_events)
    if missing_keys:
        raise RuntimeError(f"Pipeline startup primitives are missing: {', '.join(sorted(missing_keys))}")


def _validate_default_handler_imports() -> None:
    # The remote-only handler set this build ships with.
    default_handler_modules = [
        "speech_to_speech.LLM.responses_api_language_model",
        "speech_to_speech.STT.remote_openai_stt_handler",
        "speech_to_speech.TTS.remote_openai_tts_handler",
        "speech_to_speech.TTS.elevenlabs_tts_handler",
        "speech_to_speech.TTS.minimax_tts_handler",
        "speech_to_speech.VAD.vad_handler",
    ]
    for module_name in default_handler_modules:
        importlib.import_module(module_name)


def _validate_realtime_websocket_support() -> None:
    importlib.import_module("uvicorn.protocols.websockets.websockets_impl")


def main() -> None:
    # Core runtime modules for the remote-only realtime build. No local STT/TTS/LLM
    # model packages — all inference is delegated to external services.
    required_modules = [
        "fastapi",
        "httpx",
        "numpy",
        "openai",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "transformers",
        "uvicorn",
        "websockets",
    ]

    _require_modules(required_modules)
    _run_installed_cli_help()
    _validate_package_defaults()
    _validate_pipeline_startup_primitives()
    _validate_default_handler_imports()
    _validate_realtime_websocket_support()
    print("speech-to-speech installed package smoke test passed")


if __name__ == "__main__":
    main()
