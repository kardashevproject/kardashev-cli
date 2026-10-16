"""Declarative builder for Kardashev CLI local agent configurations."""

from __future__ import annotations

import enum
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.antigravity import (
    AgentConfig,
    CompactionConfig,
    LiteRTAgentConfig,
    LiteRTBackend,
    LocalOpenAIAgentConfig,
    types,
)
from google.antigravity.hooks import policy

from kardashev_cli.harness_hooks import (
    SUPER_INTELLIGENCE_COT_PROMPT,
    ActionableToolErrorHook,
    SelfHealingEditHook,
    SyntaxGatePostToolHook,
)
from kardashev_cli.presets import PRESETS, BackendKind, get_preset


class PolicyMode(str, enum.Enum):
    """Tool execution permission policy modes."""

    ASK = "ask"
    ALLOW_ALL = "allow-all"
    SANDBOX = "sandbox"


@dataclass
class HarnessOptions:
    """Configuration options for building a Kardashev CLI agent runtime."""

    preset: str | None = None
    backend: BackendKind | str | None = None
    model: str | None = None
    model_path: str | None = None
    base_url: str | None = None
    litert_backend: str = "gpu"
    speculative_decoding: bool = False
    cache_dir: str | None = None
    download_if_missing: bool = False
    workspaces: list[str] = field(default_factory=lambda: [os.getcwd()])
    policy_mode: PolicyMode | str = PolicyMode.ALLOW_ALL
    lightweight: bool = True
    token_threshold: int | None = None
    system_instructions: str | None = None
    enable_cot_harness: bool = True
    enable_self_healing_edits: bool = True
    enable_ast_repo_map: bool = True
    enable_opencommons: bool = True
    skills_paths: list[str] | None = None
    with_chronosml_mcp: bool = False
    with_opencommons_mcp: bool = False
    extra_mcp_stdio: list[tuple[str, str, list[str]]] = field(default_factory=list)
    custom_tools: list[Callable[..., Any]] = field(default_factory=list)


def _resolve_policies_and_sandbox(
    mode: PolicyMode,
) -> tuple[list[policy.Policy], bool]:
    if mode == PolicyMode.ALLOW_ALL:
        return [policy.allow_all()], False
    if mode == PolicyMode.SANDBOX:
        return [policy.allow_all()], True
    # ASK mode: default interactive confirmation for run_command
    return [policy.confirm_run_command()], False


def commons_find(
    query: str,
    modality: str = "dual_card_with_pitfalls",
    top_k: int = 3,
    io_examples: str = "",
    param_types: list[str] | None = None,
    return_type: str = "",
) -> str:
    """Search the OpenCommons Concept Commons for verified scientific/algorithmic building blocks and refuted pitfalls."""
    import json as _json

    from kardashev_cli.opencommons import execute_commons_find

    payload: dict[str, Any] = {"query": query, "modality": modality, "top_k": top_k}
    if io_examples:
        payload["io_examples"] = io_examples
    if param_types:
        payload["param_types"] = param_types
    if return_type:
        payload["return_type"] = return_type
    res = execute_commons_find(payload)
    return _json.dumps(res, indent=2)


def commons_record(
    source_code: str = "",
    domain: str = "general",
    summary: str = "",
    invariants: list[str] | None = None,
    contract: str = "",
    pseudocode: str = "",
    teleological_intent: str = "",
    dimensional_units: str = "",
    require_human_review: bool = False,
    counterexample_block_id: str = "",
    counterexample_args: str = "",
    counterexample_expected: str = "",
    sync_to_hub: bool = True,
) -> str:
    """Contribute a verified Python function to OpenCommons (requires source_code, summary, 3+ contract examples, and 1+ executable invariants; runs 16-probe behavioral fingerprinting & 5-gate peer review) or report a counterexample."""
    import json as _json

    from kardashev_cli.opencommons import execute_commons_record

    args: dict[str, Any] = {
        "source_code": source_code,
        "domain": domain,
        "summary": summary,
        "pseudocode": pseudocode,
        "teleological_intent": teleological_intent,
        "dimensional_units": dimensional_units,
        "require_human_review": require_human_review,
        "sync_to_hub": sync_to_hub,
    }
    if invariants:
        args["invariants"] = invariants
    if contract:
        args["contract"] = contract
    if counterexample_block_id and counterexample_args:
        args["counterexample_block_id"] = counterexample_block_id
        args["counterexample_args"] = counterexample_args
    if counterexample_expected:
        args["counterexample_expected"] = counterexample_expected
    res = execute_commons_record(args)
    return _json.dumps(res, indent=2)


def _build_mcp_servers(options: HarnessOptions) -> list[types.McpServerConfig]:
    import sys as _sys

    servers: list[types.McpServerConfig] = []
    if options.with_chronosml_mcp:
        chronos_bin = shutil.which("chronosml-mcp") or "chronosml-mcp"
        servers.append(
            types.McpStdioServer(
                name="chronosml",
                command=chronos_bin,
                args=[],
            )
        )
    if options.with_opencommons_mcp:
        opencommons_bin = shutil.which("opencommons-mcp")
        if opencommons_bin:
            servers.append(
                types.McpStdioServer(
                    name="opencommons",
                    command=opencommons_bin,
                    args=[],
                )
            )
        else:
            servers.append(
                types.McpStdioServer(
                    name="opencommons",
                    command=_sys.executable,
                    args=["-m", "kardashev_cli.opencommons"],
                )
            )
    for name, command, args in options.extra_mcp_stdio:
        servers.append(
            types.McpStdioServer(
                name=name,
                command=command,
                args=args,
            )
        )
    return servers


def build_agent_config(options: HarnessOptions) -> AgentConfig:
    """Builds a ready-to-run `LiteRTAgentConfig` or `LocalOpenAIAgentConfig`."""
    resolved_backend: BackendKind | None = (
        BackendKind(options.backend) if options.backend else None
    )
    resolved_model = options.model
    resolved_model_path = options.model_path
    resolved_base_url = options.base_url

    matched_preset = None
    if options.preset:
        preset_obj = get_preset(options.preset)
        matched_preset = preset_obj
        if resolved_backend is None:
            resolved_backend = preset_obj.backend
        if resolved_model is None:
            resolved_model = preset_obj.model_id
        if resolved_model_path is None and preset_obj.default_model_path:
            resolved_model_path = preset_obj.resolved_model_path()
        if resolved_base_url is None and preset_obj.default_base_url:
            resolved_base_url = preset_obj.default_base_url

    if resolved_backend is None:
        if resolved_model_path or (resolved_model and resolved_model.endswith(".litertlm")):
            resolved_backend = BackendKind.LITERT
        elif resolved_base_url or resolved_model:
            resolved_backend = BackendKind.OPENAI
        else:
            # Default to gemma4-26b LiteRT preset if nothing specified
            default_preset = get_preset("gemma4-26b")
            matched_preset = default_preset
            resolved_backend = default_preset.backend
            resolved_model = default_preset.model_id
            resolved_model_path = default_preset.resolved_model_path()

    if matched_preset is None and resolved_model:
        for candidate_preset in PRESETS.values():
            if candidate_preset.model_id == resolved_model:
                matched_preset = candidate_preset
                break

    policy_mode = (
        options.policy_mode
        if isinstance(options.policy_mode, PolicyMode)
        else PolicyMode(options.policy_mode)
    )
    policies, enable_sandbox = _resolve_policies_and_sandbox(policy_mode)
    workspaces = [os.path.abspath(os.path.expanduser(w)) for w in options.workspaces]
    mcp_servers = _build_mcp_servers(options)
    effective_token_threshold = options.token_threshold
    if (
        effective_token_threshold is None
        and matched_preset is not None
        and matched_preset.context_window > 65536
    ):
        effective_token_threshold = min(160000, int(matched_preset.context_window * 0.75))
    compaction_cfg = (
        CompactionConfig(token_threshold=effective_token_threshold)
        if effective_token_threshold is not None
        else None
    )

    hooks_list: list[Any] = []
    if options.enable_self_healing_edits:
        primary_ws = workspaces[0] if workspaces else None
        hooks_list.extend(
            [
                SelfHealingEditHook(workspace=primary_ws),
                SyntaxGatePostToolHook(auto_harvest_commons=options.enable_opencommons),
                ActionableToolErrorHook(),
            ]
        )

    combined_instructions = ""
    if options.enable_cot_harness:
        combined_instructions = SUPER_INTELLIGENCE_COT_PROMPT
    if options.system_instructions:
        combined_instructions = (
            f"{combined_instructions}\n\n{options.system_instructions}".strip()
            if combined_instructions
            else options.system_instructions
        )

    common_kwargs: dict[str, Any] = {
        "workspaces": workspaces,
        "policies": policies,
    }
    if hooks_list:
        common_kwargs["hooks"] = hooks_list
    if combined_instructions:
        common_kwargs["system_instructions"] = combined_instructions
    if options.skills_paths:
        common_kwargs["skills_paths"] = [
            os.path.abspath(os.path.expanduser(p)) for p in options.skills_paths
        ]
    if mcp_servers:
        common_kwargs["mcp_servers"] = mcp_servers

    tools_list: list[Callable[..., Any]] = list(options.custom_tools)
    if options.enable_opencommons and not options.with_opencommons_mcp:
        for oc_fn in (commons_find, commons_record):
            if oc_fn not in tools_list:
                tools_list.append(oc_fn)
    if tools_list:
        common_kwargs["tools"] = tools_list
    if compaction_cfg is not None:
        common_kwargs["compaction_config"] = compaction_cfg

    if resolved_backend == BackendKind.LITERT:
        target_path = resolved_model_path or resolved_model
        if not target_path:
            target_path = get_preset("gemma4-26b").resolved_model_path()
        expanded_path = os.path.abspath(os.path.expanduser(str(target_path)))

        cache_dir = options.cache_dir
        if cache_dir:
            cache_dir = os.path.abspath(os.path.expanduser(cache_dir))

        cfg = LiteRTAgentConfig(
            model_path=expanded_path,
            backend=LiteRTBackend(options.litert_backend.lower()),
            enable_speculative_decoding=options.speculative_decoding,
            cache_dir=cache_dir,
            download_if_missing=options.download_if_missing,
            **common_kwargs,
        )
        if enable_sandbox:
            caps = cfg.capabilities.model_copy(
                update={
                    "run_command_config": types.RunCommandConfig(enable_sandbox=True),
                }
            )
            cfg = cfg.model_copy(update={"capabilities": caps})
        return cfg

    # BackendKind.OPENAI
    if resolved_base_url:
        base_url = resolved_base_url
    elif resolved_model and "/" in resolved_model and resolve_api_key("https://openrouter.ai/api/v1"):
        base_url = "https://openrouter.ai/api/v1"
    else:
        base_url = "http://127.0.0.1:8000/v1"
    model_name = resolved_model or "qwen3-vl-8b"
    api_key = resolve_api_key(base_url)

    openai_cfg: LocalOpenAIAgentConfig
    if api_key or "openrouter.ai" in base_url.lower():
        openai_cfg = AuthenticatedOpenAIAgentConfig(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            **common_kwargs,
        )
    else:
        openai_cfg = LocalOpenAIAgentConfig(
            model=model_name,
            base_url=base_url,
            **common_kwargs,
        )
    if options.lightweight:
        openai_cfg = openai_cfg.lightweight()

    if enable_sandbox:
        caps = openai_cfg.capabilities.model_copy(
            update={
                "run_command_config": types.RunCommandConfig(enable_sandbox=True),
            }
        )
        openai_cfg = openai_cfg.model_copy(update={"capabilities": caps})

    return openai_cfg


def _read_dotenv_var(var_name: str, dotenv_path: str | os.PathLike[str] = "~/.env") -> str | None:
    """Safely reads a single variable from `~/.env` without mutating global environment."""
    from pathlib import Path

    path = Path(os.path.expanduser(str(dotenv_path)))
    if not path.is_file():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == var_name:
                val = v.strip().strip("\"'")
                if val:
                    return val
    except OSError:
        return None
    return None


def resolve_api_key(base_url: str, explicit_key: str | None = None) -> str | None:
    """Resolves the bearer API key for OpenRouter or other authenticated OpenAI endpoints."""
    if explicit_key:
        return explicit_key
    lower_url = (base_url or "").lower()
    if "openrouter.ai" in lower_url:
        candidates = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    elif "bigmodel.cn" in lower_url:
        candidates = ("ZHIPU_API_KEY", "OPENAI_API_KEY")
    elif lower_url.startswith(("http://127.0.0.1", "http://localhost")):
        return os.environ.get("OPENAI_API_KEY")
    else:
        candidates = ("OPENAI_API_KEY", "OPENROUTER_API_KEY")

    for var in candidates:
        env_val = os.environ.get(var, "").strip()
        if env_val:
            return env_val
        file_val = _read_dotenv_var(var)
        if file_val:
            return file_val
    return None


def _install_harness_process_isolation() -> None:
    """Isolates `localharness` in its own POSIX session so terminal Ctrl+C (SIGINT) never kills the Go daemon."""
    import logging as _logging

    from google.antigravity.connections.local import local_connection as _lc

    if getattr(_lc, "_kardashev_sigint_isolated", False):
        return

    orig_popen = _lc.subprocess.Popen

    def _session_isolated_popen(*args: Any, **kwargs: Any) -> Any:
        if os.name == "posix":
            kwargs.setdefault("start_new_session", True)
        return orig_popen(*args, **kwargs)

    _lc.subprocess.Popen = _session_isolated_popen  # type: ignore[assignment]

    orig_init = _lc.LocalConnection.__init__

    class _StderrFilterProxy:
        def __init__(self, raw_deque: Any) -> None:
            self._raw = raw_deque

        def append(self, line: str) -> None:
            if "logging before google.Init" in line or "auth_store.go:" in line:
                return
            self._raw.append(line)

        def __iter__(self) -> Any:
            return (
                ln
                for ln in self._raw
                if "logging before google.Init" not in ln and "auth_store.go:" not in ln
            )

        def __len__(self) -> int:
            return sum(1 for _ in self.__iter__())

        def __bool__(self) -> bool:
            return any(True for _ in self.__iter__())

    def _filtered_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._stderr_lines = _StderrFilterProxy(self._stderr_lines)

    _lc.LocalConnection.__init__ = _filtered_init  # type: ignore[method-assign]

    class _HarnessLogFilter(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            if os.environ.get("KARDASHEV_DEBUG") == "1":
                return True
            msg = record.getMessage()
            for noisy in (
                "Harness process exited unexpectedly",
                "Error in reader loop:",
                "auth_store.go:",
                "logging before google.Init",
                "enable_sandbox=True but the OS sandbox is unavailable",
            ):
                if noisy in msg:
                    return False
            return True

    _filter = _HarnessLogFilter()
    _logging.getLogger().addFilter(_filter)
    _logging.getLogger("google.antigravity").addFilter(_filter)
    _lc._kardashev_sigint_isolated = True  # type: ignore[attr-defined]


_install_harness_process_isolation()


def normalize_sse_reasoning_line(
    raw_line: bytes,
    stream_state: dict[str, Any] | None = None,
) -> bytes:
    """Normalizes OpenRouter/OpenAI SSE `delta.reasoning` / `delta.reasoning_content` chunks for `localharness`.

    1. Maps `delta.reasoning` / `delta.reasoning_content` (when `delta.content` is empty) to
       `delta.content` + `delta.extra_content = {"google": {"thought": True}}` so `localharness`
       streams `thinking_delta` live to the TUI instead of dropping 60-100s of reasoning tokens.
    2. If a completion reaches `finish_reason` having emitted ONLY reasoning tokens and neither
       visible `content` nor `tool_calls`, promotes the accumulated reasoning text to visible
       `delta.content` on the terminal chunk so `localharness`'s `empty_output_continuation_check`
       does not fail with `model output must contain either output text or tool calls`.
    """
    stripped = raw_line.lstrip()
    if not stripped.startswith(b"data:"):
        return raw_line
    payload_bytes = stripped[5:].strip()
    if not payload_bytes or payload_bytes == b"[DONE]":
        return raw_line

    import json as _json

    try:
        obj = _json.loads(payload_bytes.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return raw_line
    if not isinstance(obj, dict):
        return raw_line
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return raw_line

    state = stream_state if stream_state is not None else {}
    reasoning_parts: list[str] = state.setdefault("reasoning_parts", [])
    choice0 = choices[0]
    delta = choice0.get("delta")
    modified = False

    if isinstance(delta, dict):
        if delta.get("tool_calls"):
            state["saw_tool_calls"] = True
        existing_content = delta.get("content")
        extra = delta.get("extra_content")
        is_thought = bool(
            isinstance(extra, dict)
            and isinstance(extra.get("google"), dict)
            and extra["google"].get("thought")
        )
        if isinstance(existing_content, str) and existing_content and not is_thought:
            state["saw_visible_content"] = True

        reasoning_text = delta.get("reasoning") or delta.get("reasoning_content")
        if isinstance(reasoning_text, str) and reasoning_text:
            reasoning_parts.append(reasoning_text)
            if not existing_content:
                delta["content"] = reasoning_text
                extra_dict = delta.setdefault("extra_content", {})
                if isinstance(extra_dict, dict):
                    google_dict = extra_dict.setdefault("google", {})
                    if isinstance(google_dict, dict):
                        google_dict["thought"] = True
                modified = True

    finish_reason = choice0.get("finish_reason")
    if (
        finish_reason
        and reasoning_parts
        and not state.get("saw_visible_content")
        and not state.get("saw_tool_calls")
    ):
        if not isinstance(delta, dict):
            delta = {}
            choice0["delta"] = delta
        delta["content"] = "".join(reasoning_parts)
        if isinstance(delta.get("extra_content"), dict) and isinstance(
            delta["extra_content"].get("google"), dict
        ):
            delta["extra_content"]["google"].pop("thought", None)
        state["saw_visible_content"] = True
        modified = True

    if not modified:
        return raw_line

    line_ending = b"\r\n" if raw_line.endswith(b"\r\n") else (b"\n" if raw_line.endswith(b"\n") else b"")
    encoded = _json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"data: " + encoded + line_ending


class AuthenticatedOpenAIConnectionStrategy(
    types.Any if False else __import__(
        "google.antigravity.connections.local.local_openai_connection",
        fromlist=["LocalOpenAIConnectionStrategy"],
    ).LocalOpenAIConnectionStrategy
):
    """Connection strategy that bridges `localharness` to authenticated OpenAI/OpenRouter endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._upstream_base_url = base_url.rstrip("/")
        self._api_key = api_key or resolve_api_key(base_url)
        self._proxy_server: Any = None
        self._proxy_thread: Any = None
        super().__init__(base_url=base_url, model_name=model_name, **kwargs)

    async def __aenter__(self) -> None:
        if self._api_key:
            import contextlib
            import http.server
            import socketserver
            import threading
            import urllib.error
            import urllib.request

            upstream = self._upstream_base_url
            if not upstream.endswith("/v1"):
                upstream_v1 = f"{upstream}/v1"
            else:
                upstream_v1 = upstream
            api_key = self._api_key

            class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
                daemon_threads = True
                allow_reuse_address = True

                def handle_error(self, request: Any, client_address: Any) -> None:
                    # Suppress socketserver tracebacks on client disconnect / Ctrl+C
                    return

            class _BridgeHandler(http.server.BaseHTTPRequestHandler):
                def _forward(self, method: str) -> None:
                    headers_sent = False
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(length) if length > 0 else None
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        return

                    rel_path = self.path
                    if rel_path.startswith("/v1/"):
                        target_url = f"{upstream_v1}/{rel_path[4:]}"
                    elif rel_path == "/v1":
                        target_url = upstream_v1
                    else:
                        target_url = f"{upstream_v1}/{rel_path.lstrip('/')}"

                    req_headers = {
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com/kardashevproject/kardashev-cli",
                        "X-Title": "Kardashev Studio",
                        "Accept": self.headers.get("Accept", "application/json"),
                    }
                    if body is not None:
                        req_headers["Content-Type"] = self.headers.get(
                            "Content-Type", "application/json"
                        )

                    req = urllib.request.Request(
                        target_url,
                        data=body,
                        headers=req_headers,
                        method=method,
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=180) as resp:
                            self.send_response(resp.status)
                            ct = resp.headers.get("Content-Type", "application/json")
                            self.send_header("Content-Type", ct)
                            self.end_headers()
                            headers_sent = True
                            if "text/event-stream" in ct.lower():
                                stream_state: dict[str, Any] = {
                                    "saw_visible_content": False,
                                    "saw_tool_calls": False,
                                    "reasoning_parts": [],
                                }
                                while True:
                                    raw_line = resp.readline()
                                    if not raw_line:
                                        break
                                    out_line = normalize_sse_reasoning_line(raw_line, stream_state)
                                    self.wfile.write(out_line)
                                    self.wfile.flush()
                            else:
                                while True:
                                    chunk = resp.read(1024)
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        # Client (localharness) cancelled or closed the stream mid-flight
                        return
                    except urllib.error.HTTPError as exc:
                        with contextlib.suppress(
                            BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError
                        ):
                            err_bytes = exc.read()
                            if not headers_sent:
                                self.send_response(exc.code)
                                self.send_header("Content-Type", "application/json")
                                self.end_headers()
                                headers_sent = True
                            self.wfile.write(err_bytes)
                    except Exception as exc:  # noqa: BLE001
                        if headers_sent:
                            return
                        import json as _json

                        err_payload = _json.dumps(
                            {"error": {"message": f"Upstream connection error: {exc}"}}
                        ).encode("utf-8")
                        with contextlib.suppress(
                            BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError
                        ):
                            self.send_response(502)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(err_payload)

                def do_POST(self) -> None:
                    self._forward("POST")

                def do_GET(self) -> None:
                    self._forward("GET")

                def log_message(self, format: str, *args: Any) -> None:
                    return

            self._proxy_server = _ThreadedServer(("127.0.0.1", 0), _BridgeHandler)
            host, port = self._proxy_server.server_address[:2]
            self._base_url = f"http://{host}:{port}/v1"
            self._proxy_thread = threading.Thread(
                target=self._proxy_server.serve_forever, daemon=True
            )
            self._proxy_thread.start()

        try:
            await super().__aenter__()
        except Exception:
            self._shutdown_proxy()
            raise

    def _shutdown_proxy(self) -> None:
        if self._proxy_server is not None:
            import contextlib

            try:
                with contextlib.suppress(Exception):
                    self._proxy_server.shutdown()
                    self._proxy_server.server_close()
                    if self._proxy_thread is not None:
                        self._proxy_thread.join(timeout=2.0)
            finally:
                self._proxy_server = None
                self._proxy_thread = None
                self._base_url = self._upstream_base_url

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        try:
            await super().__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._shutdown_proxy()


class AuthenticatedOpenAIAgentConfig(LocalOpenAIAgentConfig):
    """OpenAI/OpenRouter agent configuration with automatic Bearer authentication."""

    api_key: str | None = None

    def create_strategy(
        self,
        *,
        tool_runner: Any,
        hook_runner: Any,
    ) -> Any:
        model_name = ""
        resolved_base_url = self.base_url
        if isinstance(self.model, types.ModelTarget):
            model_name = self.model.name or ""
            if not resolved_base_url and self.model.endpoint:
                resolved_base_url = self.model.endpoint.base_url
        elif isinstance(self.model, str):
            model_name = self.model

        return AuthenticatedOpenAIConnectionStrategy(
            base_url=resolved_base_url or "",
            model_name=model_name,
            api_key=self.api_key,
            tool_runner=tool_runner,
            hook_runner=hook_runner,
            system_instructions=self._get_system_instructions(),
            capabilities_config=self.capabilities,
            compaction_config=self._get_effective_compaction_config(),
            conversation_id=self.conversation_id,
            session_continuation_mode=self.session_continuation_mode,
            save_dir=self._get_or_create_save_dir(),
            workspaces=self.workspaces,
            app_data_dir=self.app_data_dir,
            skills_paths=self.skills_paths,
            inline_skills=self.inline_skills,
            mcp_servers=self.mcp_servers,
            subagents=self.subagents,
            env=self.env,
            debug_config=self.debug_config,
            retry_config=self.retry_config,
            budget_config=self.budget_config,
            policies=list(self.policies) if self.policies is not None else None,
            tools=self.tools,
        )

