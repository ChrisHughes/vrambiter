# vrambiter

**Cooperative VRAM arbitration for multiple model-serving processes sharing a GPU.**

You have one big GPU and several servers that each want to keep models resident: a TTS server, a
speech recogniser, `llama-server`, a diffusion app. Each one is happy alone; together they run the
card out of memory, or every one of them has to load cold on every request.

vrambiter is a small daemon plus a zero-dependency client library. Each server **registers** the
models it can load, **leases** a model while it is using it, and **unloads** it when the arbiter
asks. When a server needs room, the arbiter evicts idle models least-recently-used first, across
processes, using real free memory from NVML. Busy work is never interrupted.

```python
import vrambiter

arb = vrambiter.connect("my-tts-server")      # standalone (no-op) when no arbiter is running

tts = arb.register("voxcpm2", vram_peak="9GiB", host_peak="6GiB",
                   load=load_weights, unload=free_weights)

with tts.lease():          # waits for room, loads if needed, marks the model busy
    audio = synthesize("Hold it right there, fluffball.")
```

- **Leases, not guesses.** A model is busy exactly while a lease is held, and is never evicted then.
- **Real memory, not declarations.** Admission compares against what the driver says is free, and
  the arbiter measures what each model actually holds: a model declared at 34 GB that sits at
  29.7 GB while idle is behaving correctly, and is accounted at 29.7.
- **Fails fast, and says why.** When nothing could ever make room, `VramUnavailable` names who
  holds the memory, including processes vrambiter does not know about.
- **Pinning, priority and profiles** for latency-critical services.
- **Host-RAM admission.** Loads are serialised and checked against available RAM, because on many
  workstations loading two models at once kills the host before the GPU runs out.
- **Standalone mode.** Without an arbiter the same code loads on first use and keeps the model
  resident, just like a plain server.
- **Black boxes too.** Processes that don't use the library can be started, stopped and measured;
  `llama-server` in router mode is supported directly.

Status: alpha. Linux with NVIDIA GPUs is the target; the client and the test suite also run on
macOS. How the arbiter decides is in [docs/DESIGN.md](docs/DESIGN.md); adapting an existing
PyTorch server or wrapping `llama-server` is in [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Install

```sh
pip install vrambiter              # the client library: standard library only
pip install 'vrambiter[daemon]'    # plus nvidia-ml-py and psutil, where the daemon runs
```

Python 3.10 or newer (on 3.10 the package also pulls in `tomli`, for reading the daemon's config).

## Quick start

**1. Start the arbiter.** One per machine.

```sh
vrambiter daemon                    # NVML; socket at $XDG_RUNTIME_DIR/vrambiter.sock
vrambiter daemon --fake-gpu 24GiB   # no NVIDIA GPU? a simulated card, driven by declared sizes
```

**2. Make a server a satellite.** Wrap the code that loads and frees the model, and take a lease
around each use:

```python
import asyncio
import vrambiter
from vrambiter.integrations.torch import unload

state = {}

def load_weights():
    state["pipe"] = build_pipeline().to("cuda")

def free_weights():
    unload(state, "pipe")      # drop the references; vrambiter then runs gc + empty_cache

arb = vrambiter.connect("sdxl-server")
sdxl = arb.register("sdxl", vram_peak="17GiB", vram_resident="8GiB", host_peak="14GiB",
                    load=load_weights, unload=free_weights)

async def generate(prompt):
    async with sdxl.alease(timeout=300):    # asyncio servers: alease(); load() runs in a thread
        return await asyncio.to_thread(state["pipe"], prompt)
```

**3. Run it** as usual, or under `vrambiter run`, which sets the socket and the satellite name:

```sh
vrambiter run --name sdxl-server -- python server.py
```

**4. Watch.**

```console
$ vrambiter status
vrambiter 0.1.0 (pid 2439, up 3m12s), profile: none
GPU 0: 96.0 GiB total, 40.2 GiB free, 9.0 GiB reserved, 2.0 GiB headroom -> 29.2 GiB available
host RAM: 18.3 GiB available of 27.0 GiB (3.0 GiB headroom), loads in flight 0/1

MODEL              STATE     LEASES  HOLDS     PEAK      PRIO  PINNED  IDLE
llama/gemma-4-26b  resident  0       29.7 GiB  30.0 GiB  0     yes     12m03s
sdxl-server/sdxl   busy      1       8.1 GiB   17.0 GiB  0
tts/voxcpm2        resident  0       8.0 GiB   9.0 GiB   5             41s

SATELLITE    PID   CONNECTED  KIND     GPU       RESIDUE
llama        8812  yes        adapter  29.9 GiB
sdxl-server  9120  yes                 8.4 GiB
tts          9087  yes        managed  8.3 GiB
```

(`reserved` is the room promised to the busy SDXL pipeline: it holds 8.1 GiB now and may grow to
its 17 GiB peak while it works.)

## Concepts

| term | meaning |
|---|---|
| **arbiter** | the daemon (`vrambiter daemon`). One per machine. Owns the policy and the truth. |
| **satellite** | a process that talks to the arbiter, usually a model server. |
| **model** | the unit of eviction: something a satellite can load and unload. Its id is `satellite/model`. |
| **lease** | a claim that a model is in use right now. A model with leases is busy and never evicted. |
| **pinned** | never evicted. Set by the satellite, by `vrambiter pin`, or by a profile. |
| **foreign** | GPU memory held by a process vrambiter does not know. Counted, never evicted. |

**Sizes.** Each model declares `vram_peak`, the most it needs while loading *or working*
(admission reserves it), and optionally `vram_resident`, what it holds while idle, and
`host_peak`, the host RAM a load needs. A declared size is a peak, not a residency: the arbiter
measures what each model really holds (NVML per-process usage, refined by the satellite's own
`torch.cuda.memory_reserved()` report) and uses that to decide what an eviction frees. While a
model is leased, the gap between its peak and what it holds stays reserved, so a model that grows
during inference never meets a neighbour that was admitted into its room.

**Admission.** A request is admitted if `free - reserved - headroom` covers it. Otherwise the
arbiter evicts idle, unpinned models of equal or lower priority, lowest priority first and then
least recently used, just enough to cover the shortfall. If even that would not be enough it waits
while something busy, loading or evicting could change the picture, and fails with
`VramUnavailable` when nothing could. Loads also pass a host gate: at most `max_concurrent_loads`
at once, and only if `host_peak` fits in available host RAM. Waiting requests are served by
priority, then first come first served, and may carry a timeout.

**Satellite kinds.**

1. **Cooperative**: imports `vrambiter`, registers models, takes leases, unloads on request.
2. **Managed**: started by the daemon from its config. Either cooperative (it gets
   `VRAMBITER_SOCKET` and `VRAMBITER_NAME`) or a **black box**, whose one model is loaded by
   starting the process and evicted by stopping it (SIGTERM, then SIGKILL).
3. **Adapter-backed**: the daemon loads and unloads models through the server's own API. Built
   in: `llama-router`, for llama.cpp's `llama-server` in router mode.

Apps that call an adapter-backed or managed model take a **consumer lease** by full id, so the
arbiter knows it is busy and loads it if needed:

```python
with arb.lease("llama/gemma-4-26b"):
    reply = httpx.post("http://127.0.0.1:8000/v1/chat/completions", json=...)
```

**Standalone mode.** `vrambiter.connect()` returns a `NullArbiter` when no daemon is running:
`lease()` loads on first use and never unloads. If the daemon goes away mid-run, a connected client
falls back to the same behaviour and reconnects in the background, re-registering its models with
their current state. `connect(required=True)` raises instead; `connect(wait_s=10)` waits for a
daemon that is still starting.

## Client API

```python
arb = vrambiter.connect(name, socket=None, required=False, wait_s=0.0)
model = arb.register(name, vram_peak=..., load=..., unload=None, vram_resident=None,
                     host_peak=0, priority=0, pinned=False, device=0, measure=None, cleanup=True)

with model.lease(timeout=None, wait=True) as lease: ...     # sync
async with model.alease(timeout=None, wait=True): ...       # asyncio
lease = model.acquire(); ...; lease.release()                # manual
model.unload()                                               # give it back now, if idle

with arb.lease("llama/gemma-4-26b"): ...                     # consumer lease by full id
arb.status(); arb.pin(id); arb.unpin(id); arb.evict(id); arb.profile(name)
```

Errors derive from `vrambiter.VrambiterError`. `VramUnavailable` (and its subclass
`HostRamUnavailable`) carries `need`, `free`, `reason` and `holders`, a list of `Holder(kind, name,
bytes, state, pinned, priority, pid)`.

`load` and `unload` are plain callables run under a per-model lock (for `alease()`, in a worker
thread). After `unload`, the client runs `gc.collect()` and, if torch is imported,
`torch.cuda.empty_cache()`; `cleanup=False` turns that off. `$VRAMBITER_NAME` overrides the name
passed to `connect()`, so whoever launches a process decides its identity.

## Configuration

`vrambiter daemon --config vrambiter.toml` (default: `$VRAMBITER_CONFIG`, then
`~/.config/vrambiter/vrambiter.toml`). A complete, commented example with llama-server, a
cooperative satellite and a black box is in [examples/vrambiter.toml](examples/vrambiter.toml).

```toml
[arbiter]
headroom = "2GiB"               # never plan into the last 2 GiB of each GPU
host_headroom = "3GiB"
max_concurrent_loads = 1
evict_timeout_s = 30

[profiles.party]                # `vrambiter profile party` pins these
pin = ["tts/*", "llama/gemma-4-26b"]

[[satellite]]
name = "llama"
adapter = "llama-router"
url = "http://127.0.0.1:8000"
command = ["llama-server", "--models-preset", "models.ini", "--models-max", "0",
           "--no-models-autoload", "--port", "8000"]

  [[satellite.model]]
  name = "gemma-4-26b"
  vram_peak = "30GiB"

[[satellite]]
name = "tts"                    # cooperative: registers its own models over the socket
command = ["/srv/tts/.venv/bin/tts-server", "--port", "8101"]
```

Sizes accept `GiB`/`MiB` (powers of two) and `GB`/`MB` (powers of ten). Unknown keys are errors.

## CLI

| command | |
|---|---|
| `vrambiter daemon [--config F] [--fake-gpu SIZE]` | run the arbiter |
| `vrambiter status [--json]` | devices, models, satellites, queue, foreign processes |
| `vrambiter run --name NAME -- cmd ...` | run a command as a satellite |
| `vrambiter pin MODEL` / `unpin MODEL` | never evict / allow evicting again |
| `vrambiter evict MODEL` | evict now, if idle |
| `vrambiter warm MODEL [--timeout S]` | make an adapter-backed or managed model resident now |
| `vrambiter profile NAME` / `--clear` | activate or clear a profile |

`--socket PATH` works with every command. Exit codes: 0 success, 1 refused by the arbiter, 2 usage
or configuration error, 3 no arbiter reachable.

## Running it as a service

```ini
# ~/.config/systemd/user/vrambiter.service
[Unit]
Description=vrambiter VRAM arbiter

[Service]
ExecStart=%h/.local/bin/vrambiter daemon --config %h/.config/vrambiter/vrambiter.toml
Restart=on-failure

[Install]
WantedBy=default.target
```

The socket is created `0600`: whoever can connect can evict models. Set `socket_mode = "0660"` and
a shared directory to let a group in. Satellites in containers need the host PID namespace
(`--pid=host`) for NVML's per-process numbers to match their pids.

## Development

```sh
uv sync --extra dev --extra daemon
uv run pytest                 # unit, property, end-to-end and subprocess tests; no GPU needed
uv run pytest -m gpu          # on a CUDA machine with torch installed
uv run ruff check && uv run ruff format --check
uv run mypy
```

## License

Apache-2.0.
