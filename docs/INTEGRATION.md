# Integrating vrambiter

Two recipes: making an existing PyTorch/diffusers server a cooperative satellite, and putting
llama.cpp's `llama-server` under the arbiter. A third section covers black boxes, and the last one
the Linux specifics that bite on a real GPU box.

## 1. An existing PyTorch / diffusers server

### Decide what the models are

A *model* is the unit of eviction: whatever you can load and free independently. In a diffusers
server that is usually one pipeline (or one engine in a cache of engines). If several pipelines
share components (a VAE, a text encoder), either make the shared part its own model or treat the
group as one model; vrambiter does not track sharing.

When one slot can hold different variants (a checkpoint with or without a fused LoRA, a quantised
and an exact-precision transformer), register **one model per variant**. If the card cannot hold
two variants at once, the idle one is simply evicted when the other is needed, LRU as usual. If it
can, both stay resident until something else needs the room.

### Measure the sizes

Declare numbers you measured, rounded up, through the real server path:

| field | how to measure |
|---|---|
| `vram_peak` | driver-level peak (`torch.cuda.mem_get_info()` sampled during a cold load *followed by one inference*, or `nvidia-smi --query-gpu=memory.used -lms 20`). The peak covers loading **and working**. |
| `vram_resident` | the same reading once the model sits idle after a job (optional: vrambiter measures it). |
| `host_peak` | peak RSS of the process during a cold load (`/usr/bin/time -v`, `psutil`). |

Use driver-level numbers, not `torch.cuda.max_memory_allocated()`: the allocator's view misses the
CUDA context, its own cached blocks and everything outside the tensors you know about.

Get `vram_peak` right in both directions. Too low, and an inference can OOM next to a neighbour
admitted into the room it needed. Too high, and the arbiter evicts models it did not need to touch
or, once nothing is left to evict, fails a request that would have run. While a model is leased,
`vram_peak - resident` stays reserved, so the activations of a warm inference are protected without
any extra declaration: a model with a 17 GB cold peak that sits at 7.2 GB reserves 9.8 GB while it
works.

### Write `load` and `unload`

```python
import vrambiter
from vrambiter.integrations.torch import unload

state = {"pipe": None}

def load_sdxl():
    # device_map / low_cpu_mem_usage keep host RAM low: on a box with less RAM than VRAM,
    # `from_pretrained(...).to("cuda")` materialises the whole model on the CPU first.
    state["pipe"] = StableDiffusionXLPipeline.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to("cuda")

def free_sdxl():
    unload(state, "pipe")   # sets state["pipe"] = None; vrambiter runs gc.collect + empty_cache

arb = vrambiter.connect("diffusion")
sdxl = arb.register("sdxl", vram_peak="17GiB", vram_resident="8GiB", host_peak="14GiB",
                    load=load_sdxl, unload=free_sdxl)
```

Rules for the callbacks:

- **Drop every reference in `unload`.** Memory comes back only when the last reference goes:
  module caches, LoRA adapter registries, compiled graphs, a `self.last_pipe` someone kept for
  debugging. After `unload`, the client runs `gc.collect()` (reference cycles keep tensors alive)
  and `torch.cuda.empty_cache()` (the caching allocator keeps freed blocks until then). If the
  memory still does not come back, `vrambiter status` shows it as residue on the satellite.
- **Do not check "busy" yourself.** vrambiter never calls `unload` while a lease is held in this
  process: leases are counted locally and an eviction that finds one is refused on the spot.
- They run under a per-model lock, on a vrambiter thread (or a worker thread for `alease()`), not
  on your event loop. A long load does not block the server.
- `load` may raise; the lease then raises the same exception, and the arbiter releases the room.

### Lease around the whole job

```python
async def generate(req):
    async with sdxl.alease(timeout=600):          # admitted, loaded if needed, busy
        image = await asyncio.to_thread(state["pipe"], req.prompt, ...)
    return image
```

- **Busy means busy.** Hold the lease for the whole time the job uses the model, from before the
  first tensor op until after the last, not just around one `pipe(...)` call. The window between
  "got the pipeline" and "started generating" is exactly where an arbiter would otherwise evict
  weights from under a job about to use them.
- **Async servers use `alease()`.** Calling the sync `lease()` from an event loop blocks the loop
  while it waits for room.
- **Jobs that need several models**: acquire them in one fixed order everywhere (two jobs taking
  A then B and B then A can deadlock), and if two models do not fit on the card together, release
  the first before acquiring the second. A job holding a lease on A while waiting for B that needs
  A's room waits until its timeout, because A is busy, by its own hand.
- Pass a `timeout` and handle `vrambiter.VramUnavailable`: its message and `holders` say who has
  the memory, which is what a user needs to see instead of a hung job.
- Pass `on_wait=` to surface waits as they happen (`lease(on_wait=lambda r: job.status(f"waiting
  for the GPU: {r}"))`). It runs on vrambiter's thread: set a field or log, do not block.

### From an in-process arbiter

If the server already has an in-process tenancy (like the `gpu_tenancy.py` this project grew out
of), the mapping is direct:

| in-process | vrambiter |
|---|---|
| `register(name, size, is_busy, evict)` | `arb.register(name, vram_peak=..., load=..., unload=...)` |
| `ensure_room(need)` + load + claim counter | `async with model.alease():` (admission, load, claim) |
| `release()` (stays resident, becomes evictable) | leaving the `async with` |
| `evict_now()` / admin evict | `model.unload()` / `vrambiter evict satellite/model` |
| `is_busy()` callback | gone: a lease *is* busy |
| `reclaim()` (`gc` + `empty_cache`) after evicting | automatic after `unload` |
| `COLD` / `WARM` byte constants | `vram_peak` / `vram_peak - vram_resident` (reserved automatically while leased) |
| `VramUnavailable(need, free, floor)` | `vrambiter.VramUnavailable(need, free, holders)` |
| "wait forever unless nothing is busy" | the same, across processes, plus request timeouts |

Job scheduling (one heavy job at a time, a GPU slot) stays the server's business: vrambiter only
answers "is there room for this model, and if not whose idle weights can go".

### Several models in one process

Works as is. The client reports each model's growth of `torch.cuda.memory_reserved()` across its
load, and the arbiter splits the process's NVML usage among idle models in proportion to those
reports. The process's floor (the CUDA context, about 0.3-0.5 GB) is learned once its models have
all been unloaded and is then not attributed to any model.

### Trying it without a GPU

```sh
vrambiter daemon --fake-gpu 24GiB &
vrambiter run --name diffusion -- python server.py
vrambiter status
```

The fake card's usage follows the models' declared sizes, so admission, eviction and status behave
as they would on hardware, without loading anything for real (point `load` at a stub).

## 2. llama.cpp `llama-server` (router mode)

`llama-server` started without `-m` is a *router*: it serves several models, each in its own child
process, and loads and unloads them through its API. vrambiter's `llama-router` adapter drives that
API; the arbiter starts the router and treats each model as an evictable model of the `llama`
satellite.

### Configure the router

A preset file names the models (section names are the ids the router uses):

```ini
; models.ini
version = 1

[*]
n-gpu-layers = 999

[gemma-4-26b]
model = /models/gemma-4-26b-it-Q4_K_M.gguf
c = 32768

[qwen3-8b]
model = /models/Qwen3-8B-Q4_K_M.gguf
c = 16384
```

Two flags matter:

- `--models-max 0`: no router-side LRU. Otherwise the router evicts its own models at 4 loaded,
  behind the arbiter's back.
- `--no-models-autoload`: a request for a model that is not loaded fails instead of loading it.
  Without it, any chat request can start a load the arbiter did not admit. (The adapter polls
  `GET /models` and tracks such loads anyway, but only after the fact.)

Do not use `load-on-startup` in the preset; use `vrambiter warm llama/MODEL` or a consumer lease.
`--sleep-idle-seconds` is compatible: a sleeping model is tracked as unloaded.

**Crashed instances.** A model's child process can die while the router keeps listing the model
as `loaded`; requests to it then get HTTP 500 `proxy error: Could not establish connection`. The
adapter checks the child's pid every poll (and probes `GET /props?model=...&autoload=false` for
models it did not load itself), so within a poll or a few seconds the arbiter marks the model
unloaded, clears the router's stale entry, and the next lease loads a fresh instance. A consumer
that sees that 500 should treat it as transient: release the lease and take a new one.

### Configure vrambiter

```toml
[[satellite]]
name = "llama"
adapter = "llama-router"
url = "http://127.0.0.1:8000"
command = ["/opt/llama.cpp/build/bin/llama-server", "--models-preset", "/etc/llama/models.ini",
           "--models-max", "0", "--no-models-autoload", "--host", "127.0.0.1", "--port", "8000"]

  [[satellite.model]]
  name = "gemma-4-26b"          # vrambiter id llama/gemma-4-26b
  vram_peak = "30GiB"

  [[satellite.model]]
  name = "qwen"
  router_model = "qwen3-8b"     # when vrambiter's name differs from the router's id
  vram_peak = "9GiB"
```

Only declared models are managed; others the router lists (cache entries, duplicates) are ignored
(their memory still counts against free memory). `vram_peak` for a GGUF model is roughly the file
size plus the KV cache at the configured context (`c`), plus compute buffers: load it once, read
`vrambiter status`, and round up. For scale, measured child usage on a real box: Gemma 4 26B Q8
31.7 GB, Qwen3.6-35B Q8 38.9 GB; loads took 21-33 s, unloads returned the memory in 1-2 s. Let the arbiter run the router (`command`): that is what lets it attribute each model's child
process, so measurements are exact. With `url` only, the arbiter cannot know the router's pid and
falls back to the drop in free memory across each load.

### Use it from an app

Take a consumer lease by full id around each request (or each conversation turn):

```python
arb = vrambiter.connect("chat-app")
with arb.lease("llama/gemma-4-26b", timeout=300):
    r = httpx.post("http://127.0.0.1:8000/v1/chat/completions",
                   json={"model": "gemma-4-26b", "messages": msgs}, timeout=600)
```

The lease loads the model if it is not resident (waiting for room as usual) and keeps it busy until
the reply is done. With `--no-models-autoload`, a request without a lease for an unloaded model
gets HTTP 400: take the lease first. Without a running arbiter the lease is a no-op and the request goes straight to
the router. For tools that cannot take a lease, `vrambiter warm llama/gemma-4-26b` (or a profile
that pins it) makes it resident ahead of time.

### A single-model `llama-server`

A plain `llama-server -m model.gguf` is a black box: one process, one model (next section).

Launch the router directly (or through `vrambiter`), not under a wrapper that holds a lock file
descriptor: children inherit descriptors from their parent, and a model child holding a `flock`
keeps the lock forever. vrambiter itself spawns managed processes with no inherited descriptors.

## 3. Black boxes

Any server can be managed without code changes: loading it is starting it, evicting it is stopping
it (SIGTERM to its process group, SIGKILL after `kill_grace_s`).

```toml
[[satellite]]
name = "comfy"
command = ["/srv/comfy/.venv/bin/python", "main.py", "--port", "8188"]
cwd = "/srv/comfy"
health_url = "http://127.0.0.1:8188/"   # "loaded" once this answers 2xx
autostart = false                        # start on first lease instead of at daemon start

  [[satellite.model]]
  name = "sdxl"
  vram_peak = "17GiB"
```

Clients take a consumer lease on `comfy/sdxl` (or run `vrambiter warm comfy/sdxl`) before using
it. The process's GPU memory is measured from its whole process tree. If the server keeps
allocating after its health check passes, declare a `vram_resident` close to its steady state:
the arbiter only reserves the peak while the start is in flight.

## 4. Linux specifics

- **Device numbering.** vrambiter uses NVML indices. CUDA orders devices fastest-first unless
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set; set it for every satellite on a multi-GPU box (and do not
  remap with `CUDA_VISIBLE_DEVICES` without adjusting the `device` you register).
- **Containers.** NVML reports host pids. A satellite in its own PID namespace is invisible to
  per-process attribution: run it with `--pid=host`, or accept declared sizes (vrambiter detects
  the situation and logs it).
- **The socket** lives at `$XDG_RUNTIME_DIR/vrambiter.sock` (`/run/user/<uid>/...` under systemd).
  Services that run as another user need a shared path (`[arbiter] socket = ...`,
  `socket_mode = "0660"` and a common group).
- **Start order.** A satellite started before the daemon runs standalone; pass
  `connect(..., wait_s=30)` to wait for the daemon, or let the daemon start the satellite itself.
- **Persistence mode** (`nvidia-smi -pm 1`) keeps the driver from tearing down and re-initialising
  between processes, which makes NVML readings steadier and process starts faster.
