"""The ``vrambiter`` command.

    vrambiter daemon [--config vrambiter.toml]
    vrambiter status [--json]
    vrambiter run --name NAME -- command ...     # launch a satellite with VRAMBITER_SOCKET set
    vrambiter pin|unpin MODEL
    vrambiter evict MODEL
    vrambiter warm MODEL                         # acquire and release: make it resident now
    vrambiter profile NAME | --clear

Exit codes: 0 success, 1 the arbiter refused, 2 usage error, 3 no arbiter reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .client import ArbiterClient, default_socket_path
from .errors import ArbiterUnavailable, VrambiterError
from .units import format_bytes, parse_bytes

__all__ = ["main", "render_status"]

EXIT_OK, EXIT_REFUSED, EXIT_USAGE, EXIT_UNAVAILABLE = 0, 1, 2, 3


def _default_config() -> str | None:
    explicit = os.environ.get("VRAMBITER_CONFIG")
    if explicit:
        return explicit
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    candidate = Path(base) / "vrambiter" / "vrambiter.toml"
    return str(candidate) if candidate.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrambiter",
        description="Cooperative VRAM arbitration for processes sharing a GPU.",
    )
    parser.add_argument("--version", action="version", version=f"vrambiter {__version__}")
    parser.add_argument(
        "--socket",
        help="arbiter socket (default: $VRAMBITER_SOCKET, $XDG_RUNTIME_DIR/vrambiter.sock)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    # --socket is accepted after the subcommand too (`vrambiter status --socket X`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--socket", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    d = sub.add_parser("daemon", help="run the arbiter", parents=[common])
    d.add_argument(
        "--config",
        default=None,
        help="vrambiter.toml (default: $VRAMBITER_CONFIG, ~/.config/vrambiter/vrambiter.toml)",
    )
    d.add_argument(
        "--fake-gpu",
        metavar="SIZE",
        help="use an in-memory fake GPU of SIZE (try vrambiter without NVIDIA hardware)",
    )
    d.add_argument("--log-level", default=None, help="debug, info, warning (default: config)")

    s = sub.add_parser(
        "status", help="show devices, models, satellites and the queue", parents=[common]
    )
    s.add_argument("--json", action="store_true", help="machine-readable output")

    r = sub.add_parser(
        "run", help="run a command as a satellite (sets VRAMBITER_SOCKET/_NAME)", parents=[common]
    )
    r.add_argument("--name", required=True, help="satellite name")
    r.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command [args...]")

    for name, text in (
        ("pin", "never evict MODEL"),
        ("unpin", "allow MODEL to be evicted again"),
        ("evict", "evict MODEL now if it is idle"),
    ):
        c = sub.add_parser(name, help=text, parents=[common])
        c.add_argument("model", help="full model id, e.g. llama/gemma-4-26b")

    w = sub.add_parser(
        "warm", help="make MODEL resident now (acquire, then release)", parents=[common]
    )
    w.add_argument("model")
    w.add_argument("--timeout", type=float, default=None, help="seconds to wait for room")

    pr = sub.add_parser("profile", help="activate a profile from the config", parents=[common])
    group = pr.add_mutually_exclusive_group(required=True)
    group.add_argument("name", nargs="?")
    group.add_argument("--clear", action="store_true", help="deactivate the current profile")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "daemon":
            return _daemon(args)
        if args.command == "run":
            return _run(args)
        return _control(args)
    except ArbiterUnavailable as exc:
        print(f"vrambiter: {exc.message or exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except VrambiterError as exc:
        print(f"vrambiter: {exc.message or exc}", file=sys.stderr)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        return 130


def _daemon(args: argparse.Namespace) -> int:
    from .config import Config, ConfigError, load_config
    from .daemon import run_daemon

    path = args.config or _default_config()
    try:
        config = load_config(path) if path else Config()
    except ConfigError as exc:
        print(f"vrambiter: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.socket:
        config.socket = args.socket
    if args.fake_gpu:
        config.gpu = "fake"
        config.fake_vram = parse_bytes(args.fake_gpu)
    level = (args.log_level or config.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    from .gpu import GpuBackendError

    try:
        asyncio.run(run_daemon(config))
    except GpuBackendError as exc:
        print(f"vrambiter: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:  # e.g. another daemon already listening
        print(f"vrambiter: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


def _run(args: argparse.Namespace) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("vrambiter run: give a command after --", file=sys.stderr)
        return EXIT_USAGE
    env = dict(os.environ)
    env["VRAMBITER_NAME"] = args.name
    env["VRAMBITER_SOCKET"] = args.socket or default_socket_path()
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:
        print(f"vrambiter run: {cmd[0]}: {exc.strerror}", file=sys.stderr)
        return 127
    return EXIT_OK  # pragma: no cover - exec does not return


def _control(args: argparse.Namespace) -> int:
    client = ArbiterClient(
        f"vrambiter-cli-{os.getpid()}",
        socket=args.socket or default_socket_path(),
        role="control",
        reconnect=False,
    )
    try:
        if args.command == "status":
            status = client.status()
            if args.json:
                print(json.dumps(status, indent=2, sort_keys=True))
            else:
                print(render_status(status))
        elif args.command == "pin":
            client.pin(args.model)
            print(f"pinned {args.model}")
        elif args.command == "unpin":
            client.unpin(args.model)
            print(f"unpinned {args.model}")
        elif args.command == "evict":
            client.evict(args.model)
            print(f"evicting {args.model}")
        elif args.command == "warm":
            with client.lease(args.model, timeout=args.timeout) as lease:
                print(f"{args.model} is resident ({lease.action})")
        elif args.command == "profile":
            client.profile("" if args.clear else args.name)
            print("profile cleared" if args.clear else f"profile {args.name} active")
    finally:
        client.close()
    return EXIT_OK


# --------------------------------------------------------------------------- rendering


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    return lines


def render_status(status: dict[str, Any]) -> str:
    """Human-readable ``vrambiter status``."""
    out: list[str] = []
    arb = status.get("arbiter", {})
    profile = arb.get("profile") or "none"
    out.append(
        f"vrambiter {arb.get('version', '?')} (pid {arb.get('pid', '?')}, "
        f"up {_duration(arb.get('uptime_s'))}), profile: {profile}"
    )
    for d in status.get("devices", []):
        out.append(
            f"GPU {d['index']}: {format_bytes(d['total'])} total, {format_bytes(d['free'])} free, "
            f"{format_bytes(d['reserved'])} reserved, {format_bytes(d['headroom'])} headroom "
            f"-> {format_bytes(max(0, d['effective_free']))} available"
            + (f" ({format_bytes(d['pending'])} returning)" if d.get("pending") else "")
        )
    host = status.get("host", {})
    if host.get("available") is not None:
        out.append(
            f"host RAM: {format_bytes(host['available'])} available of "
            f"{format_bytes(host.get('total'))} ({format_bytes(host.get('headroom'))} headroom), "
            f"loads in flight {host.get('loads_in_flight', 0)}/{arb.get('max_concurrent_loads', 1)}"
        )
    models = status.get("models", [])
    if models:
        out.append("")
        rows = []
        for m in models:
            holds = m.get("measured") if m.get("measured") is not None else m.get("reported")
            state = m["state"] + (" (unresponsive)" if m.get("unresponsive") else "")
            rows.append(
                [
                    m["id"],
                    state,
                    str(m.get("leases", 0)),
                    format_bytes(holds) if holds is not None and m["state"] != "unloaded" else "-",
                    format_bytes(m.get("vram_peak")),
                    str(m.get("priority", 0)),
                    "yes" if m.get("pinned") else "",
                    _duration(m.get("idle_s")) if m["state"] in ("resident",) else "",
                ]
            )
        out += _table(["MODEL", "STATE", "LEASES", "HOLDS", "PEAK", "PRIO", "PINNED", "IDLE"], rows)
    sats = status.get("satellites", [])
    if sats:
        out.append("")
        rows = []
        for s in sats:
            kind = "adapter" if s.get("adapter") else ("managed" if s.get("managed") else "")
            usage = sum(s.get("usage", {}).values())
            residue = sum(s.get("residue", {}).values())
            rows.append(
                [
                    s["name"],
                    str(s.get("pid") or "-"),
                    "yes" if s.get("connected") else "no",
                    kind,
                    format_bytes(usage) if usage else "-",
                    format_bytes(residue) if residue else "",
                ]
            )
        out += _table(["SATELLITE", "PID", "CONNECTED", "KIND", "GPU", "RESIDUE"], rows)
    queue = status.get("queue", [])
    if queue:
        out.append("")
        out.append("QUEUE")
        for q in queue:
            timeout = f", times out in {_duration(q['timeout_s'])}" if q.get("timeout_s") else ""
            out.append(
                f"  {q['model']} for {q['requester']} (priority {q.get('priority', 0)}), "
                f"waiting {_duration(q.get('waiting_s'))}{timeout}: {q.get('reason', '')}"
            )
    foreign = status.get("foreign", [])
    if foreign:
        out.append("")
        out.append("FOREIGN (not managed by vrambiter)")
        for f in foreign:
            out.append(f"  pid {f['pid']} on GPU {f['device']}: {format_bytes(f['bytes'])}")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
