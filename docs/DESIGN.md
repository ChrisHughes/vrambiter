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

This document describes the implementation as built. Where it departs from the first draft of the
design, the change and the reason are listed at the end ("Changes from the first draft").

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
| **model** | the unit of eviction: something a satellite can load and unload. Id: `satellite/model`. |
| **lease** | a claim that a model is in use right now. A model with leases is busy. |
| **admission** | permission to load (or to work): the arbiter has made (or found) room. |
| **pinned** | never evicted. |
| **foreign** | GPU memory held by a process the arbiter does not know. Counted, never evicted. |
| **settle** | the window after an eviction or exit in which freed memory is expected back. |

## Satellite kinds

1. **Cooperative.** Imports `vrambiter`, registers its models with size declarations, takes leases
   around work, and unloads a model when the arbiter sends `evict`. Fine-grained: a satellite with
   three models can give back one.
2. **Managed.** The arbiter starts the process from its config. It may also be cooperative (the
   arbiter sets `VRAMBITER_SOCKET` and `VRAMBITER_NAME` in its environment), or a black box the
   arbiter can only start, stop (SIGTERM, then SIGKILL) and measure through NVML. A black box is
   exactly one model: loading it is starting the process (and waiting for its health check),
   evicting it is stopping it.
3. **Adapter-backed.** A built-in adapter speaks a server's own API on the satellite's behalf. The
   first is `llama-router`: llama.cpp's `llama-server` in router mode loads and unloads models
   through `POST /models/load` and `POST /models/unload`, each model in its own child process.

Consumers that call an adapter-backed or managed model (for example an app sending chat requests
to llama-server) take a lease on it by its full id, `llama/gemma-4-26b`, so the arbiter knows it is
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
                              │
                              └── evict_refused ──► RESIDENT (protected for a moment)
                              └── evict timeout ──► RESIDENT, unresponsive
```

- `RESIDENT` with zero leases is **idle** and evictable unless pinned. `BUSY` is not a separate
  phase in the code: it is `RESIDENT` with leases. Never evicted.
- `LOADING` holds a **reservation** against free memory until the satellite reports `loaded`, so
  two admissions cannot both count the same free gigabytes (see "Reservations").
- After `evict_refused` (a lease arrived first), the model is *protected* for `refuse_cooldown_s`
  (2 s) so the next planning pass does not immediately evict it again.
- If an eviction is not answered within `evict_timeout_s`, the model goes back to `RESIDENT` marked
  **unresponsive**: excluded from victim selection until the satellite talks to the arbiter again
  (a late `evicted` is still accepted, as is the owner's next `acquire`).
- When a satellite disconnects, all its leases end and its models go to `UNLOADED`. The arbiter
  confirms through NVML that the process's memory is gone before releasing what it reserved
  (see "Settling").

## Sizes and measurement

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

What a model holds while idle (its *resident estimate*) is, in order of preference: the measured
figure, the satellite's own report, `vram_resident`, `vram_peak`.

Measurement: with `loaded`, the satellite may report the bytes it holds (for PyTorch the client
reports the growth of `torch.cuda.memory_reserved()` across `load()`); the arbiter also reads NVML
per-process usage of the satellite's process tree every poll, and while a satellite's models are
all idle it attributes that usage to them:

- one resident model takes the satellite's usage minus its **baseline** (the CUDA context and
  allocator floor). The baseline is learned when an unload settles with nothing left resident; it
  is never learned from a satellite that has just connected, since it may be re-registering models
  it still holds.
- several resident models split it in proportion to their reports or declarations.
- models that run in their own processes (llama-router instances) are measured exactly, by pid.
- a mismatch above 20% against the declaration is logged once, never fatal.
- usage of exactly zero for a process with resident models means NVML cannot see it (a container
  in its own PID namespace, typically): measurements are treated as unknown, declared sizes are
  used, and this is logged once.

Measurements are not taken while a satellite has a model loading, leased or being evicted, or
while its freed memory is still settling, since the process total then includes memory that is
nobody's residency.

Free memory and per-process usage cannot be read atomically. Each snapshot reads per-process usage
before and after the free reading and keeps the smaller figure per process, so an allocation (or a
free) landing between the reads can never count as both free and already held.

## Reservations

Reserved memory is subtracted from free memory before any admission. It covers loads in flight and
the working headroom of busy models, and it is computed per satellite process as an **envelope**:

```
envelope    = baseline + Σ peak of loading models + Σ max(peak, resident) of busy models
              + Σ resident of idle models
outstanding = Σ peak of loading models + Σ (peak - resident) of busy models
reserved    = clamp(envelope - usage_of_the_process, 0, outstanding)
```

(`baseline` is the process's learned floor, the CUDA context and allocator pool: it is in the
process's usage but in no model's estimate.)

So a load reserves its full peak at first and less as it allocates (no double counting of what the
driver already shows as used), and a leased model keeps the room it may still grow into while it
works. When per-process usage is unknown, `outstanding` is reserved in full. Requests for a lease
on an idle resident model ask for its working headroom (`peak - resident`); a second lease on a
busy model asks for nothing.

## Admission policy

Pure functions in `policy.py`, the most heavily tested code in the project:

```
plan(request, snapshot) -> Admit | Evict(victims) | Wait(kind, reason) | Fail(VramUnavailable)
plan_queue(requests, snapshot) -> [(request, decision)]
```

Inputs: the requested model, its priority and need, current NVML free memory per device,
reservations, memory expected back (`pending`: evictions in flight and settling), every model's
state, leases, pins, priority, `last_used` and resident estimate, host `MemAvailable`, loads in
flight and the host peaks they reserved.

0. If `need > total - headroom`, nothing could ever make room: **Fail** immediately.
1. Loads pass the **host gate** first: if `loads_in_flight >= max_concurrent_loads`, **Wait**; if
   `host_peak > MemAvailable - host_headroom - Σ host_peak of loads in flight`, **Wait** when a
   load is in flight, else **Fail** (`HostRamUnavailable`). The gate comes before any eviction, so
   the arbiter never evicts models for a load that could not start anyway.
2. `effective_free = nvml_free - reserved - headroom`. If `need <= effective_free`: **Admit**.
3. If `need <= effective_free + pending`: **Wait** for the memory already on its way, evicting
   nothing more. (Without this rule a driver slow to return pages turns one eviction into several.)
4. Otherwise choose **victims** from idle, unpinned, responsive, unprotected models on the same
   device whose priority is `<=` the requester's, ordered by `(priority asc, last_used asc, id)`.
   Take the shortest prefix whose resident estimates cover the shortfall: **Evict(victims)**, then
   re-plan once the evictions are confirmed.
5. If evicting every eligible victim still is not enough:
   - if something on the device is in flight (a busy, loading, evicting or momentarily protected
     model, or memory still returning) **and** the most that could ever be made available to
     this requester once it all finishes would cover the need: **Wait** (queue). That bound is
     free + pending + every eligible idle model + every in-flight model the requester will be
     allowed to evict once it is idle + every reservation (a busy model gives back its working
     headroom when its lease ends). A busy model of higher priority never becomes evictable by a
     lower-priority request, so waiting on it alone would only end in a timeout.
   - else: **Fail** with `VramUnavailable(need, free, holders)`, a structured error that names
     what holds the memory: models with their state, pins and priorities, foreign processes, and
     whatever the driver counts as used that nobody claims.

Waiting is **event-driven**: every acquire, release, `loaded`, `evicted`, refusal, disconnect,
settle check and each NVML poll (default every 1 s, to catch memory moved by foreign processes)
wakes a single planning task, which measures (in a worker thread) and re-plans the whole queue.
Only requests that arrived before the measurement started are planned in a pass, so no admission is
ever made on numbers older than the request.

**The queue** is planned by `plan_queue` in priority order, then FIFO, against a working copy of the
snapshot that each decision updates: an admission reserves its need and takes a load slot; an
eviction marks its victims; a request left waiting on VRAM *earmarks* the room it is counting on,
so later requests may **backfill** into what remains but never take an earlier request's room; a
load left waiting at the host gate holds back later loads, so loads start in queue order; a
request that would fail while requests ahead of it on the same device are still waiting waits
instead, since their admission changes the picture; one decision is made per model per pass (a
second request for a model being loaded, evicted for or waited on just queues behind the first);
and a model chosen as a victim earlier in the pass is not leased later in it. A request may carry a timeout; expiry fails it
with `VramUnavailable` naming the holders. `wait=False` (or a zero timeout) fails instead of
waiting (including waiting behind other requests), but still lets its own evictions run.

## Eviction

1. The arbiter marks the victim `EVICTING` and sends `evict` to its owner (or calls the adapter, or
   stops the black-box process).
2. A cooperative owner unloads it (its `unload` callback; the library then runs `gc.collect()` and
   `torch.cuda.empty_cache()` if torch is loaded) and replies `evicted`, or replies
   `evict_refused` if a lease arrived first (the model goes back to `RESIDENT`, protected briefly)
   or if `unload` raised.
3. If no reply arrives within `evict_timeout` (default 30 s):
   - managed process: SIGTERM to its process group, then SIGKILL after `kill_grace`, only when the
     model is that process's only resident model or policy `kill_on_evict_timeout` is set. A
     cooperative satellite or a llama-server router is then started again (its restart policy
     permitting); if the stop changed nothing, the model falls back to `unresponsive`.
   - otherwise: the model is marked `unresponsive` and excluded from victim selection, and the
     request re-plans.
4. After `evicted`, the model is `UNLOADED` but its memory is counted as **settling** (see below)
   until NVML shows it returned, so the room is not double-counted and no extra victims are taken
   meanwhile. A satellite whose memory does not come back is reported (`status` shows the residue);
   it is not punished.

### Settling

A settle record is created for every eviction, voluntary `unloaded`, disconnect and process exit.
Until it settles, its outstanding bytes count as *pending* (rule 3) and a load that was in flight
keeps its reservation. It settles when the processes involved hold nothing on the device, when 80%
of the expected bytes are back (by per-process usage, else by free memory), or after
`settle_timeout_s` (5 s), at which point any shortfall is recorded as residue. A disconnect or exit
creates one record for the whole satellite (its models share one process's usage). A satellite that
reconnects and re-registers a resident model drops its old records: that memory is in use, not
returning.

## Protocol

Newline-delimited JSON over a Unix domain socket (default `$XDG_RUNTIME_DIR/vrambiter.sock`, else
`/tmp/vrambiter-$UID.sock`; override with `VRAMBITER_SOCKET`). The socket is created mode 0600:
whoever can connect can evict models. Requests carry an integer `id`; responses echo it. The
arbiter can send unsolicited messages (`evict`, `notice`) at any time. Lines are limited to 1 MiB.

```
client -> arbiter                                 arbiter -> client
hello    {name, pid, version, protocol, role}  ->   welcome {satellite, protocol, arbiter_version}
register {model, device, vram_peak, vram_resident?, host_peak, priority, pinned,
          state, vram_bytes?, leases}          ->   ok {model, leases?} | error
acquire  {model, wait, timeout_s?}             ->   granted {lease, action: "load"|"ready", model}
                                                    | error
cancel   {request}                             ->   ok   (the acquire also gets error "cancelled")
loaded   {model, vram_bytes?}                  ->   ok
load_failed {model, error}                     ->   ok
release  {lease}                               ->   ok   (idempotent)
unloaded {model}                               ->   ok   (voluntary unload)
evicted  {model, evict_id}                     ->   ok
evict_refused {model, evict_id, reason}        ->   ok
pin / unpin {model}                            ->   ok {model}
evict_model {model}                            ->   ok {model} | error   (operator: evict if idle)
profile  {name}                                ->   ok | error           ("" clears)
status   {}                                    ->   status {arbiter, devices, host, models,
                                                            satellites, queue, foreign}
                                               <-   evict  {evict_id, model, reason}
                                               <-   notice {kind, message, data}
```

- `role` is `"satellite"` (default) or `"control"` (the CLI): control connections may acquire
  (consumer leases), pin, evict, switch profiles and ask for status, but do not register models and
  are not listed as satellites.
- `register` with `state: "resident"`, `vram_bytes` and `leases: n` is how a satellite re-registers
  after a reconnect or an arbiter restart: the arbiter believes it, marks the model resident and
  busy, and returns `n` adopted lease ids in `ok.leases` so the satellite's later `release`s match.
  `state: "loading"` does the same for a load granted on the previous connection and still
  running: the model is `LOADING`, with its reservation and load slot, until `loaded` arrives.
  (`loaded` and `load_failed` describe the model's state now and are sent on whichever connection
  is live; lease ids only mean something on their own connection.)
- A satellite addresses its own models by short name and any model by full id.
- Error replies are `{type: "error", id, code, message, detail}`. Codes: `vram_unavailable`,
  `host_ram_unavailable` (both with `detail` = need, free, reason, holders), `unknown_model`,
  `registration_error`, `name_in_use`, `protocol_error`, `unknown_type`, `arbiter_error`,
  `internal_error`. The client maps them back to exception classes.
- `notice` kinds: `waiting` (`data: {request, model, blockers}`, sent whenever the reason a queued
  request waits changes; the client hands it to the caller's `on_wait`), and `lease_lost` (a
  consumer's lease ended because the model's process went away).

`acquire` on a model the caller does not own (a consumer lease) is allowed for adapter-backed and
managed models: the arbiter loads it through the adapter or by starting the process if needed and
replies `ready` when it is resident. Consumer leases on another cooperative satellite's models are a
later extension.

Protocol version is 1. Unknown fields are ignored; unknown message types get an `error` reply,
never a disconnect.

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

- `connect()` returns an `ArbiterClient`, or a `NullArbiter` when no daemon answers.
  `$VRAMBITER_NAME` overrides the name given in code (so a launcher picks the identity),
  `wait_s` waits for a daemon that is still starting, `required=True` raises
  `ArbiterUnavailable` instead of falling back.
- The client runs one background thread with its own event loop and a single socket. Sync and
  async APIs bridge into it. Messages are written in the order they are submitted (submission is
  `call_soon_threadsafe`, never a task that might run later), which is what keeps `loaded` ahead of
  `release` and `evicted` ahead of the next `acquire`.
- `load` and `unload` run under a per-model lock, and the lease path takes the same lock to check
  residency, so an eviction can never race a lease: a lease that arrives during an unload waits for
  it and then loads again. For `alease()`, `load` runs in a worker thread.
- Local lease counting: a lease is counted locally *before* its `acquire` is sent, and an `evict`
  that finds the count non-zero is answered `evict_refused` immediately. This holds even when the
  arbiter's view is stale.
- A `ready` grant for a model that is not resident here (a voluntary unload or a late eviction
  raced the lease) is never loaded behind the arbiter's back: the client reports `unloaded`,
  releases the grant and asks again, which gets an admitted `load`.
- Cancelling an async acquire (or interrupting a sync one) sends `cancel`; a grant already on its
  way, or already arrived, is released.
- Requests are tracked per connection: when a connection drops, exactly its requests fail (the
  callers carry on standalone) and a single reconnect loop runs.
- **Standalone mode** (`NullArbiter`): `lease()` loads on first use and never unloads. Behaviour
  without an arbiter is exactly what a plain server would do.
- If the arbiter goes away mid-run, the client falls back to standalone behaviour and reconnects in
  the background with backoff, re-registering its models with their current state and live lease
  counts. A model the new arbiter rejects stays usable standalone; the others coordinate.

## Managed processes and configuration

`vrambiter.toml` (parsed strictly: an unknown key is an error, and every error names its location):

```toml
[arbiter]
socket = "/run/user/1000/vrambiter.sock"
headroom = "2GiB"               # never plan into the last 2 GiB
host_headroom = "3GiB"
max_concurrent_loads = 1
poll_interval_s = 1.0
evict_timeout_s = 30
# settle_timeout_s = 5, refuse_cooldown_s = 2, kill_grace_s = 10, kill_on_evict_timeout = false,
# socket_mode = "0600", log_level = "info", gpu = "nvml" | "fake", fake_vram = "24GiB"

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
  # router_model = "ggml-org/gemma-4-26b:Q4_K_M"   # llama-server's id, if different

[[satellite]]
name = "cerberus-tts"
command = ["/home/me/cerberus/services/tts/.venv/bin/cerberus-tts", "--port", "8101"]
autostart = true                # cooperative: registers its own models over the socket
```

A satellite's kind is inferred: `adapter` makes it adapter-backed; a `command` with exactly one
`[[satellite.model]]` and no adapter is a black box; a `command` with no models is cooperative.
Per-satellite options: `cwd`, `env`, `autostart` (default true; for a black box it means "load at
startup", through admission), `restart` (`never` | `on-failure` | `always`, exponential backoff
from `restart_backoff_s`; black boxes are never restarted behind the arbiter's back, they start on
demand), `health_url`, `start_timeout_s`, `kill_grace_s`, `load_timeout_s`.

Managed processes run in their own session, and every signal goes to the **process group**:
stopping llama-server's router must stop the per-model children that hold the VRAM, and children
left behind when a leader exits on its own are terminated too. The daemon stops all managed
processes on shutdown.

**Pins.** A model is pinned if an operator pinned it (`vrambiter pin`), or, absent an operator
override, if it was registered pinned or matches the active profile. `vrambiter unpin` overrides
both. Activating a profile clears operator overrides on the models it names: the most recent
operator action wins.

CLI:

```
vrambiter daemon [--config vrambiter.toml] [--fake-gpu SIZE]
vrambiter status [--json]
vrambiter run --name NAME -- command ...     # launch a satellite with VRAMBITER_SOCKET set
vrambiter pin|unpin MODEL
vrambiter evict MODEL
vrambiter warm MODEL                         # acquire and release: make it resident now
vrambiter profile NAME | --clear
```

## The llama-router adapter

llama-server in router mode lists models (`GET /models`, each with `status.value` in `loaded`,
`loading`, `unloaded`, `sleeping`, `downloading`, plus `failed`/`exit_code` after a failed load),
and loads and unloads them asynchronously (`POST /models/load` / `/models/unload` with
`{"model": id}` return at once; the status changes later). The adapter posts, then polls `/models`
until the model reaches the wanted state, failing on `failed` or on a load that falls back to
`unloaded`. Blocking HTTP (stdlib `urllib`) runs in worker threads.

Each loaded model is a child process of the router, so the adapter notes the router's children
before and after a load and attributes the new ones to the model: its VRAM is their NVML usage,
exactly. When that is not possible (the arbiter does not run the router, so has no pid; or the
driver hides per-process numbers) it falls back to the drop in free memory across the load, which
is right as long as nothing else moved meanwhile (loads are serialised, so usually nothing did).
Every poll the adapter also lists `/models` and reports loads and unloads the router made on its
own (autoload for a request, `--sleep-idle-seconds`), which the arbiter then tracks. Only declared
models are considered (the router also lists cache entries and duplicates). A poll result computed
before a transition the arbiter made meanwhile is dropped as stale (models carry a version that
every phase change bumps).

**Dead instances.** A child instance can die while the router still lists the model as `loaded`;
every request to it then fails with HTTP 500 `proxy error: Could not establish connection` until
the model is unloaded. So `/models` is never trusted alone: each poll checks that the child
processes recorded at load time are alive and, for a model loaded behind the arbiter's back (no
pids recorded), probes `GET /props?model=<id>&autoload=false` every few seconds (proxied to the
child: 200 alive, 500 `proxy error` dead, 400 not loaded). A dead instance is reported as
`"dead"`: the arbiter marks the model unloaded even if leased (consumers get a `lease_lost`
notice), the adapter unloads the stale router entry, and the next lease loads a fresh instance.
A load that finds the model "loaded" but dead does the same before loading.

Recommended router flags: `--models-max 0` (no router-side LRU; vrambiter decides what to evict)
and `--no-models-autoload` (a request for an unloaded model fails with HTTP 400 instead of loading
it behind the arbiter's back; consumers take a lease, which loads it). Measured on llama-server
b11379: loads of 30-40 GB Q8 models take 21-33 s and an unload returns the child's VRAM within
1-2 s; the router process itself holds no VRAM.

Managed processes are spawned with no inherited file descriptors (`close_fds`): a router started
while holding, say, an inherited `flock` descriptor would pass it to every model child, which then
keeps the lock open forever.

## Backends (seams for testing)

- `gpu.GpuBackend`: `devices()`, `mem_info(device) -> (free, total)`,
  `process_usage(device) -> {pid: bytes} | None` (`None`: the driver cannot say; `{}`: nobody).
  Implementations: `NvmlBackend` (`nvidia-ml-py`, optional extra, imported lazily), `FakeGpu`
  (in-memory, programmable, raises a fake OOM when over-committed; optionally shared between
  processes through a `flock`ed JSON file, with dead pids reaped as the driver would), and
  `daemon.SimulatedGpu` (for `--fake-gpu`: usage follows the models' declared sizes).
- `host.HostBackend`: `mem_available()`, `mem_total()`. Implementations: `/proc/meminfo` (Linux),
  `psutil` fallback, `UnknownHost` (host-RAM checks skipped), and `FakeHost`.
  `host.process_tree(pid)` finds a satellite's child processes (psutil, else `/proc`).
- `clock.Clock`: `now()` and `call_later()`. Every timeout and the poll are timers on the injected
  clock, so tests run with `FakeClock` and move time explicitly.

## Package layout

```
src/vrambiter/
  __init__.py          connect(), public types
  units.py             "17GiB" / "9GB" / ints  <->  bytes
  protocol.py          message dataclasses, NDJSON framing, version
  errors.py            VramUnavailable, HostRamUnavailable, ProtocolError, ...
  clock.py             Clock, LoopClock, FakeClock
  gpu.py               GpuBackend, NvmlBackend, FakeGpu
  host.py              HostBackend, ProcMeminfo, PsutilHost, FakeHost, process_tree
  policy.py            plan(), plan_queue(): pure admission and victim selection
  state.py             models, leases, satellites, settle records, snapshots (no I/O)
  arbiter.py           the daemon core: applies plans, drives evictions, timeouts, events
  server.py            Unix-socket server and connection handling
  client.py            ArbiterClient / NullArbiter, Model, Lease (sync + async), reconnect
  managed.py           process supervisor; BlackBoxDriver
  adapters/
    llama_router.py    llama-server router mode
  config.py            TOML config (tomllib; tomli on 3.10)
  daemon.py            assembles arbiter, server, processes and adapters from a config
  _http.py             minimal JSON-over-HTTP on urllib
  cli.py               the `vrambiter` command
  integrations/torch.py  unload helpers: drop refs, gc, empty_cache; reserved-bytes measurement
```

Python 3.10+. The client (`client.py`, `protocol.py`, `units.py`, `errors.py`) imports nothing outside
the standard library; `integrations/torch.py` never imports torch itself. The daemon optionally uses
`nvidia-ml-py` and `psutil`.

## Testing

- `policy.py`: table tests for every rule above, plus property tests (Hypothesis): never evict a
  busy, pinned, unresponsive or protected model; never admit beyond effective free (alone or summed
  over a queue pass); victims are the shortest covering prefix in (priority, LRU) order; equal
  inputs give equal plans whatever order models or requests arrive in; Fail only when nothing in
  flight could help.
- `state.py`: reservations (envelope), attribution, settling, snapshot building.
- The arbiter driven message by message with fake connections and a fake clock (precise timing),
  and the server with raw NDJSON (malformed input, stale sockets).
- Arbiter + server + client end to end over a real Unix socket with `FakeGpu` and a fake clock:
  admission, eviction round trips, refusals over the wire, timeouts, disconnect cleanup, queue
  fairness, host-RAM serialisation, working headroom, standalone fallback, reconnect.
- A complete daemon from a config with tiny Python scripts as cooperative, black-box and
  llama-router satellites (the last a faithful fake of router mode, spawning a child per model)
  sharing a file-backed `FakeGpu`; killed, restarted and evicted by the supervisor.
- The CLI, including `vrambiter daemon` as a subprocess.
- On a real GPU (`-m gpu`): a torch satellite allocates tensors, gets evicted, and NVML shows the
  memory returned.

## Known limitations

- **A cooperative load cannot be timed out by the arbiter.** The satellite owns it; a hung `load()`
  keeps its reservation and the load slot until it returns or the satellite disconnects.
  `status` shows how long each load has been running.
- **Consumer leases on another cooperative satellite's models** are not supported yet (adapter-
  backed and managed models only).
- **A satellite started before the daemon stays standalone** (`connect()` returns a
  `NullArbiter`); use `connect(wait_s=...)` or let the daemon start it.
- **`vram_peak` is per model, not per lease.** Several concurrent leases on one model share its
  peak; a satellite that runs several jobs on one model at once must declare a peak that covers
  them.
- **Attribution inside one process is approximate** (split by reports or declarations), and with
  `max_concurrent_loads > 1` two router children starting at once can be attributed together.
- **Priority can starve.** Under sustained load, low-priority requests may wait behind higher
  ones until their timeout. That is the policy (priority, then FIFO), not a bug; use timeouts.
- **Device indices are NVML's**; satellites should set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **No authentication** beyond the socket's file permissions.

## Changes from the first draft

- **Rule 0 (never fits) was added.** A request larger than `total - headroom` fails at once. The
  in-process predecessor waited forever for 79 GB of a card that could never offer more than 77.
- **The host gate is checked before VRAM eviction,** not after it, so models are never evicted for
  a load that cannot start yet.
- **Rule 3 (wait for memory already returning)** was added, with **settle records** replacing "wait
  up to a few seconds after `evicted`": the arbiter keeps planning while memory comes back, but
  counts it as pending instead of evicting more.
- **Busy models reserve their working headroom** (`peak - resident`), and reservations are an
  envelope (`peak - what the process already holds`) rather than the full peak. The draft reserved
  `vram_peak` only while loading, which left a model that grows during inference unprotected, and
  double-counted every byte a load had already allocated.
- **"Could free memory later" includes evicting and protected models and settling memory, but
  only counts what could help this requester** (in-flight models it will be allowed to evict,
  plus returning reservations), and the queue earmarks room in order. A randomised end-to-end
  test showed low-priority requests waiting out their timeouts behind busy higher-priority models
  that could never make room for them. Found by the end-to-end tests: with only busy/loading models
  counted, a third request behind two that had earmarked an in-flight eviction failed instead of
  waiting.
- **Refusal cooldown**: after `evict_refused` a model is not chosen again for 2 s, so a refusal is
  not followed by an immediate identical eviction.
- **Only requests older than the measurement are planned** in a pass.
- **Baselines** (per-satellite CUDA context and allocator floor) are learned when an unload settles,
  and zero NVML usage for a process with resident models is treated as "cannot see it". Both came
  from real failure modes: a reconnecting satellite's usage taken as its baseline zeroed its model's
  size, and a satellite in another PID namespace measured 0 B and so was never evicted.
- **Protocol additions**: `cancel` (withdraw a queued acquire), `evict_model` and `profile`
  (operator commands), `hello.role` (`control` connections for the CLI), re-registration fields on
  `register` with adopted lease ids in `ok.leases`, `notice` kinds `waiting` (for `on_wait`) and
  `lease_lost`, and a list of error codes. All additive; the protocol is still version 1.
- **Names**: the client classes are `ArbiterClient` and `NullArbiter` (the daemon core is
  `arbiter.Arbiter`); `clock.py`, `daemon.py` and `_http.py` were added to the layout.
- **Pins** have explicit precedence (operator override, then declaration or profile; activating a
  profile clears overrides on the models it names).
- **`--fake-gpu` simulates usage from declared sizes**, so the daemon can be tried meaningfully on
  a machine without an NVIDIA GPU.
- **Black boxes** are exactly one model, never auto-restarted; `autostart` loads them through
  admission. Signals always go to the process group.
- **Consumer leases** work for managed (black-box) models as well as adapter-backed ones.
- **Found in review and fixed** (each with a regression test): a grant arriving as its acquirer
  gave up leaked its lease (and with it the load slot); a link drop mid-reconnect started a second
  reconnect loop and could strand requests; a model registered during a reconnect was never
  registered; a lease racing a voluntary unload reloaded without admission; snapshots could count
  one allocation as both free and held; reservations ignored the learned baseline; one queue pass
  could lease a model it had just chosen as a victim, or evict twice for two requests for the same
  model; a driven model's `unresponsive` flag could stick; a driven load finishing after its process
  died left the request unanswered; a stale adapter poll could undo a transition.
- **llama-server dead instances** (from benchmarking b11379): a crashed child stays listed as
  loaded, so the adapter checks child pids and probes `/props` rather than trusting `/models`.
