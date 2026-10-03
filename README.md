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

- **Leases, not guesses.** A model is busy exactly while a lease is held.
- **Pinning and priority** for latency-critical services.
- **Host-RAM admission.** Loads are serialised and checked against available RAM, because on many
  workstations loading two models at once kills the host before the GPU runs out.
- **Standalone mode.** Without an arbiter the same code loads on first use and keeps the model
  resident, just like a plain server.
- **Black boxes too.** Processes that don't use the library can be started, stopped and measured;
  `llama-server` in router mode is supported directly.

Status: alpha, under active development. See [docs/DESIGN.md](docs/DESIGN.md).

## License

Apache-2.0.
