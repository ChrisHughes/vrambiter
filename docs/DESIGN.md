# vrambiter design

vrambiter lets several independent model-serving processes share one GPU (or a few) without
running each other out of memory. Each process keeps its models resident while they are useful
and gives the memory back, least-recently-used first, when another process needs room. Nothing is
preempted: busy work is never interrupted, only idle models are evicted.

It grew out of an in-process arbiter in a single-user ML server (one FastAPI app holding SDXL,
an image-edit model, a matting model and a 3D pipeline on one 96 GB card). Moving to separate
processes keeps that behaviour and fixes what in-process arbitration cannot:

- **A process exit returns all of its memory.** In-process, freed tensors stay in the PyTorch
  caching allocator and read as "used" to the only measurement an arbiter has.
- **The driver reports memory per process** (NVML), so attribution stops being guesswork.
- **Each server keeps its own dependencies.** A TTS model pinned to torch 2.9, a speech
  recogniser with its own CUDA libraries and a llama.cpp binary can share one card.

## Goals and non-goals

Goals:
1. Cooperative LRU eviction of idle models across processes, driven by real free memory (NVML).
2. Leases: a model with an active lease is busy and is never evicted.
3. Pinning and priority, so a latency-critical service can keep its models resident.
4. **Host-RAM admission.** Loading a model can need more host RAM than it will finally occupy
   (safetensors staging, fp32 materialisation). On a box with 27 GB RAM and 97 GB VRAM, two loads at
   once can OOM-kill the machine. The arbiter serialises loads and checks available RAM first.
5. **Standalone mode.** A satellite that imports the client library runs unchanged as a normal
   server when no arbiter is present. Coordination is an improvement, never a dependency.
6. Black-box processes (no client library) can still be managed: started, stopped and measured.
7. Zero dependencies in the client library, so importing it costs nothing.

Non-goals: scheduling compute or time-slicing kernels, multi-node clusters, preempting busy work,
replacing an inference server.

## Vocabulary

| term | meaning |
|---|---|
| **arbiter** | the daemon (`vrambiter daemon`). One per machine. Owns the policy and the truth. |
| **satellite** | a process that talks to the arbiter. Usually a model server. |
| **model** | the unit of eviction: something a satellite can load and unload. |
| **lease** | a claim that a model is in use right now. A model with leases is busy. |
| **admission** | permission to load: the arbiter has made (or found) room in VRAM and host RAM. |
| **pinned** | never evicted. |
| **foreign** | GPU memory held by a process the arbiter does not know. Counted, never evicted. |

## Satellite kinds

1. **Cooperative.** Imports `vrambiter`, registers its models with size declarations, takes leases
   around work, and unloads a model when the arbiter sends `evict`. Fine-grained: a satellite with
   three models can give back one.
2. **Managed.** The arbiter starts the process from its config. It may also be cooperative (the
   arbiter sets `VRAMBITER_SOCKET` in its environment), or a black box the arbiter can only start,
   stop (SIGTERM, then SIGKILL) and measure through NVML.
3. **Adapter-backed.** A built-in adapter speaks a server's own API on the satellite's behalf. The
   first is `llama-router`: llama.cpp's `llama-server` in router mode loads and unloads models
   through `POST /models/load` and `POST /models/unload`, each model in its own child process.

Consumers that call an adapter-backed model (for example an app sending chat requests to
llama-server) take a lease on it by its full id, `llama/gemma-4-26b`, so the arbiter knows it is
busy and loads it if needed.

## Model state machine

```
            acquire (admitted)            loaded
 UNLOADED ─────────────────────► LOADING ────────► RESIDENT ◄──┐
    ▲   ▲                           │                │  ▲       │ release (last lease)
    │   │ load_failed               │                │  │       │
    │   └───────────────────────────┘      acquire   ▼  │       │
    │                                     (lease)   BUSY ───────┘
    │          evicted / unloaded / owner gone       │
    └───────────────────── EVICTING ◄── evict ── RESIDENT (idle, not pinned)
```

- `RESIDENT` with zero leases is **idle** and evictable unless pinned.
- `BUSY` means one or more leases. Never evicted.
- `LOADING` holds a **reservation** of `vram_peak` against free memory until the satellite reports
  `loaded`, so two admissions cannot both count the same free gigabytes.
- When a satellite disconnects, all its leases end and its models go to `UNLOADED`. The arbiter
  confirms through NVML that the process's memory is gone before releasing the reservation.

## Sizes

Each model declares:

| field | meaning |
|---|---|
| `vram_peak` | VRAM needed while loading or working. Admission reserves this. |
| `vram_resident` | VRAM held while idle-resident. Optional; refined by measurement. |
| `host_peak` | host RAM needed during load. Optional; default 0. |
| `priority` | integer, default 0. Higher priority may evict lower; equal priority evicts by LRU. |
| `pinned` | never evict. Settable at runtime and by profile. |

**A declared size is a peak, not a residency.** A model that declares 34 GB and holds 29.7 GB while
idle is behaving correctly. The arbiter keeps both numbers and uses each where it belongs.

Measurement: after `loaded`, the satellite may report the bytes it holds (for PyTorch,
`torch.cuda.memory_reserved()`); the arbiter also reads NVML per-process usage. For a single-model
satellite the NVML number wins. For a multi-model satellite the process total is split among its
models using their reports or declarations. A mismatch above 20% is logged, never fatal.

## Admission policy

Pure function in `policy.py`, the most heavily tested code in the project:

```
plan(request, snapshot) -> Admit | Evict(victims) | Wait(reason) | Fail(VramUnavailable)
```

Inputs: the requested model and its priority, current NVML free memory per device, outstanding
reservations, every model's state, leases, pins, priority and `last_used`, host `MemAvailable`,
and loads in flight.

1. `effective_free = nvml_free - reservations - headroom`.
2. If `vram_peak <= effective_free`, and host RAM and load concurrency allow it: **Admit**.
3. Otherwise choose **victims** from idle, unpinned models on the same device whose priority is
   `<=` the requester's, ordered by `(priority asc, last_used asc)`. Take the shortest prefix whose
   resident sizes cover the shortfall. If that covers it: **Evict(victims)**, then re-plan after
   the evictions are confirmed.
4. If evicting every eligible victim still is not enough:
   - if any busy model (or any load in flight) could free memory later: **Wait** (queue).
   - else: **Fail** with `VramUnavailable(need, free, holders)`, a structured error that names
     what holds the memory, including foreign processes.
5. Host RAM: if `host_peak > MemAvailable - host_headroom`, or `loads_in_flight >= max_concurrent_loads`:
   **Wait** for the in-flight load to finish (or Fail if none is in flight and RAM is still short).

Waiting is **event-driven**: every release, `loaded`, `evicted`, disconnect and each NVML poll
(default every 1 s, to catch memory moved by foreign processes) re-plans the queue. The queue is
ordered by priority, then FIFO. A request may carry a timeout; expiry fails it with
`VramUnavailable`.

## Eviction

1. The arbiter marks the victim `EVICTING` and sends `evict` to its owner.
2. A cooperative owner unloads it (its `unload` callback; the library then runs `gc.collect()` and
   `torch.cuda.empty_cache()` if torch is loaded) and replies `evicted`, or replies
   `evict_refused` if a lease arrived first (the model goes back to `BUSY`).
3. If no reply arrives within `evict_timeout` (default 30 s):
   - managed process: SIGTERM, then SIGKILL after `kill_grace`, only when the model is that
     process's only model or policy `kill_on_evict_timeout` is set.
   - otherwise: the model is marked `unresponsive` and excluded from victim selection, and the
     request re-plans.
4. After `evicted`, the arbiter waits for NVML to show the memory returned (up to a few seconds)
   before treating the room as free. A satellite whose memory does not come back is reported
   (`status` shows the residue); it is not punished.

## Protocol

Newline-delimited JSON over a Unix domain socket (default `$XDG_RUNTIME_DIR/vrambiter.sock`,
override with `VRAMBITER_SOCKET`). Requests carry an `id`; responses echo it. The arbiter can send
unsolicited messages (`evict`, `notice`) at any time.

```
client -> arbiter                               arbiter -> client
hello    {name, pid, version, protocol}    ->   welcome {satellite, protocol, arbiter_version}
register {model, device, vram_peak, ...}   ->   ok | error
acquire  {model, wait, timeout_s}          ->   granted {lease, action: "load"|"ready"} | error
loaded   {model, vram_bytes?}              ->   ok
load_failed {model, error}                 ->   ok
release  {lease}                           ->   ok
unloaded {model}                           ->   ok           (voluntary unload)
evicted  {model, evict_id}                 ->   ok
evict_refused {model, evict_id, reason}    ->   ok
pin / unpin {model}                        ->   ok
status   {}                                ->   status {devices, models, satellites, queue, foreign}
                                            <-  evict  {evict_id, model, reason}
                                            <-  notice {kind, ...}
```

`acquire` on a model the caller does not own (a consumer lease) is allowed for adapter-backed and
managed models: the arbiter loads it through the adapter if needed and replies `ready` when it is
resident. Consumer leases on another cooperative satellite's models are a later extension.

Protocol version starts at 1. Unknown fields are ignored; unknown message types get an `error`
reply, never a disconnect.

## Client library

```python
import vrambiter

arb = vrambiter.connect("my-tts-server")       # NullArbiter when no socket: standalone mode

model = arb.register(
    "voxcpm2",
    vram_peak="9GiB", vram_resident="8GiB", host_peak="6GiB",
    load=load_weights,        # () -> None, called when the model must be made resident
    unload=free_weights,      # () -> None, called on eviction
)

with model.lease():           # blocks until admitted; calls load() if needed; busy until exit
    audio = synthesize(text)

async with model.alease():    # asyncio variant
    ...

with arb.lease("llama/gemma-4-26b"):   # consumer lease on an adapter-backed model
    reply = httpx.post(...)
```

- The client runs one background thread with its own event loop and a single socket. Sync and
  async APIs bridge into it. `load` and `unload` run under a per-model lock, so an eviction can
  never race a lease.
- Local lease counting: the client knows its own leases and answers `evict_refused` immediately
  when the model is in use locally.
- **Standalone mode** (`NullArbiter`): `lease()` loads on first use and never unloads. Behaviour
  without an arbiter is exactly what a plain server would do. `vrambiter.connect(required=True)`
  raises instead.
- If the arbiter goes away mid-run, the client falls back to standalone behaviour and reconnects
  in the background, re-registering its models with their current state.

## Managed processes and configuration

`vrambiter.toml`:

```toml
[arbiter]
socket = "/run/user/1000/vrambiter.sock"
headroom = "2GiB"               # never plan into the last 2 GiB
host_headroom = "3GiB"
max_concurrent_loads = 1
poll_interval_s = 1.0
evict_timeout_s = 30

[profiles.party]                # `vrambiter profile party` pins these
pin = ["cerberus-tts/*", "cerberus-stt/*", "llama/gemma-4-26b"]

[[satellite]]
name = "llama"
adapter = "llama-router"
command = ["/home/me/llama.cpp/llama-server", "--models-preset", "models.ini", "--port", "8000"]
url = "http://127.0.0.1:8000"
autostart = true

  [[satellite.model]]
  name = "gemma-4-26b"
  vram_peak = "30GiB"

[[satellite]]
name = "cerberus-tts"
command = ["/home/me/cerberus/services/tts/.venv/bin/cerberus-tts", "--port", "8101"]
autostart = true                # cooperative: registers its own models over the socket
```

CLI:

```
vrambiter daemon [--config vrambiter.toml]
vrambiter status [--json]
vrambiter run --name NAME -- command ...     # launch a satellite with VRAMBITER_SOCKET set
vrambiter pin|unpin MODEL
vrambiter evict MODEL
vrambiter warm MODEL                         # acquire and release: make it resident now
vrambiter profile NAME
```

## Backends (seams for testing)

- `gpu.GpuBackend`: `devices()`, `mem_info(device) -> (free, total)`,
  `process_usage(device) -> {pid: bytes}`. Implementations: `NvmlBackend` (`nvidia-ml-py`, optional
  extra) and `FakeGpu` (in-memory, programmable; tests allocate and free "memory" per pid).
- `host.HostBackend`: `mem_available()`. Implementations: `/proc/meminfo` (Linux), `psutil`
  fallback, and `FakeHost`.
- `clock`: injected for timeouts and LRU, so tests run with a fake clock.

## Package layout

```
src/vrambiter/
  __init__.py          connect(), public types
  units.py             "17GiB" / "9GB" / ints  <->  bytes
  protocol.py          message dataclasses, NDJSON framing, version
  errors.py            VramUnavailable, ProtocolError, ...
  gpu.py               GpuBackend, NvmlBackend, FakeGpu
  host.py              HostBackend, ProcMeminfo, FakeHost
  policy.py            plan(): pure admission and victim selection
  state.py             models, leases, reservations, queue (no I/O)
  arbiter.py           the daemon core: applies plans, drives evictions, timeouts, events
  server.py            Unix-socket server and connection handling
  client.py            Arbiter / NullArbiter, Model, leases (sync + async), reconnect
  managed.py           process supervisor for configured satellites
  adapters/
    llama_router.py    llama-server router mode
  config.py            TOML config (tomllib; tomli on 3.10)
  cli.py               the `vrambiter` command
  integrations/torch.py  unload helpers: drop refs, gc, empty_cache; reserved-bytes measurement
```

Python 3.10+. The client (`client.py`, `protocol.py`, `units.py`, `errors.py`) imports nothing outside
the standard library. The daemon optionally uses `nvidia-ml-py` and `psutil`.

## Testing

- `policy.py`: table tests for every rule above, plus property tests (Hypothesis): never evict a
  busy or pinned model; never admit beyond effective free; victims are chosen in
  (priority, LRU) order; equal inputs give equal plans.
- Arbiter + server + client end to end over a real Unix socket with `FakeGpu` and a fake clock:
  admission, eviction round trips, refusals, timeouts, disconnect cleanup, queue fairness,
  host-RAM serialisation, standalone fallback, reconnect.
- Managed processes: tiny Python scripts as satellites that "allocate" in `FakeGpu` (shared through
  a file or a socket), killed and restarted by the supervisor.
- `llama-router` adapter against a fake HTTP server that mimics `/models`, `/models/load` and
  `/models/unload`.
- On a real GPU (`-m gpu`): a torch satellite allocates tensors, gets evicted, and NVML shows the
  memory returned.
