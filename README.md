# KVBridge
**Stateful multi-backend router for `llama.cpp` with tiered KV cache reuse**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

KVBridge sits in front of one or more `llama-server` instances and routes requests toward the backend that can answer them with the least waste. Instead of treating each backend as an isolated endpoint, KVBridge tracks slot activity, reuses serialized KV snapshots, aggregates model discovery, and exposes a control plane for managing the fleet.

It is designed for rigs where PCIe is the bottleneck and where large prompts make prefill cost dominate throughput. The router keeps hot cache state in RAM, persists cold snapshots to SSD, and tries to restore the best matching state before forwarding inference.

## What It Does

- Tiered KV cache storage with an L1 RAM disk and L2 persistent cache directory.
- Partial cache reuse using prompt-prefix similarity scoring instead of exact-hit-only routing.
- Model-aware routing for OpenAI-compatible chat requests.
- Aggregated model discovery across multiple `llama-server` backends.
- Dashboard and settings APIs for live inspection and operational control.
- Managed backend lifecycle hooks with per-backend start, stop, restart, and log access.
- Rolling request logging for prompt inspection in `ghost_detector.txt`.

## Runtime Layout

- `server.js`: Fastify router, cache orchestration, backend management, OpenAI-compatible endpoints.
- `settings.js`: settings defaults, config loading, persistence, and live settings helpers.
- `dashboard.html`: control deck UI served by the router.
- `config/config.example.json`: example runtime config.
- `config/start_server.example.sh`: example managed backend launcher for the first worker.
- `config/start_server2.example.sh`: example managed backend launcher for the second worker.
- `runtime-logs/`: backend stdout/stderr logs.
- `ghost_detector.txt`: serialized request logging.

## Requirements

- Node.js 18 or newer.
- One or more `llama-server` backends reachable over HTTP.
- A writable RAM-backed cache directory for hot KV snapshots.
- A writable persistent cache directory for cold KV snapshots.
- Backend launch scripts if you want KVBridge to auto-start workers.

## Quick Start

1. Install dependencies.

	```bash
	npm install
	```

2. Copy the example config and adjust paths, ports, models, and backend launch scripts.

	```bash
	cp config/config.example.json config/config.json
	```

3. Review the storage paths in `config/config.json`.

	Recommended defaults:

	- `storage.l1Dir`: RAM disk or tmpfs path, for example `$HOME/Models/ramdisk_cache`
	- `storage.l2Dir`: persistent SSD/NVMe path, for example `$HOME/Models/slot_cache`
	- `storage.backendLogDir`: directory where managed backend logs are written

4. Point each backend entry at a real launch script.

	The example scripts in `config/` show the expected shape:

	- bind an HTTP port
	- enable `--metrics`
	- set `--slot-save-path`
	- optionally preload a model using `/models/load`

5. Start KVBridge.

	Foreground:

	```bash
	npm start
	```

	Background helper:

	```bash
	./start.sh
	```

6. Open the control deck at `http://127.0.0.1:8080/dashboard`.

## Configuration

The router reads `config/config.json` at startup. If the file is missing, built-in defaults from `settings.js` are used.

Main sections:

- `server`: HTTP bind host and port for KVBridge.
- `storage`: `modelsDir`, `l1Dir`, `l2Dir`, and `backendLogDir`.
- `backends`: managed worker definitions, including `id`, `host`, `port`, `gpuGroup`, `maxSlots`, `scriptPath`, and `modelName`.
- `cache`: L1 eviction thresholds and partial-cache scoring knobs.
- `health`: backend probe timings, startup grace period, and auto-start behavior.
- `gc`: periodic cleanup interval and max age for cache artifacts.
- `misc`: warmup, hashing, and large-input cooperative processing thresholds.

The example config defines two backends on ports `11434` and `11435` and is the best starting point for a multi-GPU setup.

When KV persistence is enabled, backend workers should be started with `--no-context-shift`. KVBridge reuses serialized slot snapshots by exact prompt-prefix identity; if a backend silently shifts or truncates its in-memory context and that mutated state is saved under the original hash, future restores will be corrupted.

## HTTP Surface

### User and Dashboard Endpoints

- `GET /dashboard`: serves the control deck UI.
- `GET /api/stats`: queue length, L1 usage, slot state, aggregated model state, backend health, and collected metrics.
- `GET /api/settings`: returns the active settings object.
- `POST /api/settings`: persists a new settings object.
- `POST /api/settings/reload`: reloads settings from disk.
- `POST /api/restart`: exits the KVBridge process so a supervisor can respawn it.
- `GET /api/backends/:backendId/logs?lines=120`: tails a managed backend log.
- `POST /api/backends/:backendId/start|stop|restart`: controls a managed backend.

### Model Discovery and Control

- `GET /models`: aggregated llama.cpp-style model inventory.
- `POST /models/load`: loads a model across eligible backends.
- `POST /models/unload`: unloads a model across eligible backends.
- `GET /v1/models`: OpenAI-compatible aggregated model list.
- `GET /v1/models/:modelId`: OpenAI-compatible single-model view.

### Inference

- `POST /v1/chat/completions`: OpenAI-compatible chat endpoint.

Routing behavior for chat completions:

- the request `model` field is respected
- only backends currently advertising that model are considered
- cache affinity, queueing, and slot assignment run inside that model-qualified pool
- unknown models return `400 invalid_request_error`
- known-but-unavailable models return `503 unavailable_error`

## Operations Notes

- Managed backend logs are written under `runtime-logs/` unless `storage.backendLogDir` points elsewhere.
- Request summaries are serialized into `ghost_detector.txt` to avoid interleaved writes under load.
- `start.sh` relaunches KVBridge under `nohup` and writes the main process output to `app.log`.
- The dashboard expects `/api/stats` and will reflect aggregated llama metrics when worker backends expose `/metrics`.
- Requests that overflow backend context are forwarded as client-facing errors and are never persisted to the KV cache. Responses that end with `finish_reason = length` are also treated conservatively and skip cache save.

## Example Flow

1. A client sends `POST /v1/chat/completions` with a model id.
2. KVBridge selects the subset of healthy backends advertising that model.
3. The router checks for an exact or partial cache match for the prompt context.
4. If needed, it restores the best candidate from L2 to L1 and injects state before inference.
5. The request runs on the chosen slot and updated cache state is persisted for reuse.

## Development

```bash
npm run dev
```

The dev script runs `node --watch server.js`.

## License

MIT
