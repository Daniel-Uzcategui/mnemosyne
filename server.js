#!/usr/bin/env node
/**
 * KV Cache Semantic Router — Tiered Storage (L1 RAM / L2 SSD)
 * ============================================================
 * Storage Orchestrator — never holds binary data in V8 heap.
 *
 * L1_DIR (RAMDisk, tmpfs):  $HOME/Models/ramdisk_cache  — volatile, fast
 * L2_DIR (SSD, persistent): $HOME/Models/slot_cache     — unlimited, slow
 *
 * Usage:
 *   node server.js
 *
 * Listens on :8080, proxies to llama.cpp at :11434.
 */

'use strict';

const { createHash } = require('crypto');
const { spawn } = require('child_process');
const fastify = require('fastify');
const fastifyCors = require('@fastify/cors');
const fastifyFormBody = require('@fastify/formbody');

const { copyFile, rm, readdir, stat, mkdir, appendFile, readFile, open } = require('fs/promises');
const { existsSync, createWriteStream } = require('fs');
const { join, extname } = require('path');

const settings = require('./settings');
const { loadSettings, getSettings, setSettings } = settings;

// ─── Tiered Storage Configuration ───────────────────────────────────
// All values loaded from settings at startup.

let MODELS_DIR, L1_DIR, L2_DIR, BACKEND_LOG_DIR;
let DEFAULT_BACKEND_MAX_SLOTS, BACKEND_HEALTHCHECK_INTERVAL_MS, BACKEND_HEALTHCHECK_TIMEOUT_MS;
let BACKEND_CONFIGS, BACKEND_AUTO_START, BACKEND_STARTUP_GRACE_MS;
let L1_EVICT_THRESHOLD, L1_EVICT_TARGET;
let GARBAGE_COLLECT_INTERVAL_MS, MAX_FILE_AGE_MS;
let RAG_CONTENT_LENGTH_THRESHOLD, WARMUP_TOP_N;
let SUBAGENT_MODEL, SUBAGENT_PROMPT_PATTERNS;

async function initSettings() {
  await loadSettings();
  const s = getSettings();

  MODELS_DIR = s.storage.modelsDir;
  L1_DIR = s.storage.l1Dir;
  L2_DIR = s.storage.l2Dir;
  BACKEND_LOG_DIR = s.storage.backendLogDir;

  DEFAULT_BACKEND_MAX_SLOTS = 2;
  BACKEND_HEALTHCHECK_INTERVAL_MS = s.health.backendHealthcheckIntervalMs;
  BACKEND_HEALTHCHECK_TIMEOUT_MS = s.health.backendHealthcheckTimeoutMs;
  BACKEND_CONFIGS = s.backends.map(b => ({
    id: b.id,
    host: b.host,
    port: b.port,
    gpuGroup: b.gpuGroup,
    maxSlots: b.maxSlots,
    scriptPath: b.scriptPath,
  }));
  BACKEND_AUTO_START = s.health.backendAutoStart;
  BACKEND_STARTUP_GRACE_MS = s.health.backendStartupGraceMs;

  L1_EVICT_THRESHOLD = s.cache.l1EvictThresholdBytes;
  L1_EVICT_TARGET = s.cache.l1EvictTargetBytes;

  GARBAGE_COLLECT_INTERVAL_MS = s.gc.gcIntervalMs;
  MAX_FILE_AGE_MS = s.gc.maxFileAgeMs;

  RAG_CONTENT_LENGTH_THRESHOLD = s.misc.ragContentLengthThreshold;
  WARMUP_TOP_N = s.misc.warmupTopN;
  SUBAGENT_MODEL = normalizeModelId(s.misc.subagentModel) || 'fast';
  SUBAGENT_PROMPT_PATTERNS = Array.isArray(s.misc.subagentPromptPatterns)
    ? s.misc.subagentPromptPatterns
        .filter(pattern => typeof pattern === 'string')
        .map(pattern => pattern.trim().toLowerCase())
        .filter(Boolean)
    : [];
}

// ─── Helpers ────────────────────────────────────────────────────────

const C = {
  reset:  '\x1b[0m',
  bold:   '\x1b[1m',
  grey:   '\x1b[90m',
  green:  '\x1b[32m',
  yellow: '\x1b[33m',
  red:    '\x1b[31m',
  cyan:   '\x1b[36m',
  magenta:'\x1b[35m',
};

function log(tag, msg) {
  console.log(`${C.grey}${new Date().toISOString().slice(11, 23)}${C.reset}  ${C.bold}[${tag}]${C.reset} ${msg}`);
}

function logHit(hash) {
  console.log(`${C.green}${C.bold}■ CACHE HIT${C.reset}  slot  →  ${hash}.bin  (restoring from L1 RAMDisk)`);
}

function logColdHit(hash) {
  console.log(
    `${C.yellow}${C.bold}■ COLD HIT${C.reset}  slot  →  ${hash}.bin  (copying L2→L1, then restoring)`
  );
}

function logMiss(hash) {
  console.log(
    `${C.yellow}${C.bold}■ CACHE MISS${C.reset}  slot  →  ${hash}.bin  (will save to L1, then persist to L2)`
  );
}

function logWarn(msg) {
  console.log(`${C.yellow}${C.bold}⚠ WARN${C.reset}    ${msg}`);
}

function logErr(msg) {
  console.error(`${C.red}${C.bold}✖ ERROR${C.reset}   ${msg}`);
}

// ─── Ghost Detector Logger ──────────────────────────────────────────
// Serial log writer — prevents interleaving under concurrent load.
// Each call enqueues a log entry; a single background drain serializes
// writes to disk so no two appendFile calls overlap.

const MAX_LOG_QUEUE = 10_000;
const logQueue = [];
let logDraining = false;

async function flushLogQueue() {
  if (logDraining || logQueue.length === 0) return;
  logDraining = true;

  try {
    while (logQueue.length > 0) {
      const entry = logQueue.shift();
      await appendFile(join(__dirname, 'ghost_detector.txt'), entry, 'utf8');
    }
  } finally {
    logDraining = false;
  }
}

function enqueueLogEntry(logContent) {
  if (logQueue.length >= MAX_LOG_QUEUE) {
    logWarn('Log queue full, dropping oldest entry');
    logQueue.shift();
  }
  logQueue.push(logContent);
  flushLogQueue().catch(err => {
    logWarn(`Log flush failed: ${err.message}`);
  });
}

function logRequestToFile(reqId, systemPrompt, firstUserMessage) {
  const timestamp = new Date().toISOString();
  let logContent = `\n========== [${timestamp}] REQ: ${reqId} ==========\n`;
  logContent += `[SYSTEM PROMPT]\n${systemPrompt ? systemPrompt.substring(0, 800) + '...' : 'NO SYSTEM PROMPT'}\n`;

  if (firstUserMessage) {
    logContent += `\n[FIRST USER MSG]\n${firstUserMessage.content ? firstUserMessage.content.substring(0, 500) + '...' : 'NONE'}\n`;
  }
  logContent += `======================================================\n`;

  enqueueLogEntry(logContent);
}
// ─── fs/promises convenience wrappers ───────────────────────────────

async function safeReaddir(dir) {
  try {
    return await readdir(dir);
  } catch {
    return [];
  }
}

async function safeStat(path) {
  try {
    return await stat(path);
  } catch {
    return null;
  }
}

// ─── 1. Backend Registry + Slot State ───────────────────────────────

function createBackendState(config) {
  const baseUrl = `http://${config.host}:${config.port}`;
  return {
    ...config,
    baseUrl,
    healthy: false,
    lastCheckAt: 0,
    lastError: null,
    slotQueue: [],
    availableSlots: Array.from({ length: config.maxSlots }, (_, i) => i),
    activeSlots: new Map(),
    slotContextIndex: new Map(),
    slotActivity: new Map(),
    discoveredModels: new Map(),
    modelsDiscoveryAt: 0,
    modelsDiscoveryError: null,
    lastMetrics: {},
    lastMetricsAt: 0,
    failureStreak: 0,
    degradedUntil: 0,
    requestFailures: 0,
    restoreFailures: 0,
    lastFailureKind: null,
    processInfo: {
      status: 'stopped',
      pid: null,
      startedAt: null,
      logPath: join(BACKEND_LOG_DIR, `${config.id}.log`),
      lastExitCode: null,
      lastExitSignal: null,
      lastStartAttemptAt: 0,
      autoStartDisabled: false,
    },
  };
}

let backends = null;
let backendsById = new Map();
let roundRobinCounter = 0;
const MODEL_OPERATION_TIMEOUT_MS = 90_000;
const MODEL_OPERATION_POLL_INTERVAL_MS = 1_000;
const backendModelOperationLocks = new Map();
const BACKEND_FAILURE_COOLDOWN_MS = 60_000;
const BACKEND_FAILURE_MAX_COOLDOWN_MS = 5 * 60_000;

function normalizeModelId(model) {
  if (typeof model !== 'string') return null;
  const value = model.trim();
  return value.length > 0 ? value : null;
}

function isSubagentPrompt(systemPrompt = '') {
  const normalized = normalizeText(systemPrompt);
  if (!normalized || SUBAGENT_PROMPT_PATTERNS.length === 0) {
    return false;
  }

  return SUBAGENT_PROMPT_PATTERNS.some(pattern => normalized.includes(pattern));
}

function createDiscoveredModelEntry(modelId, patch = {}) {
  return {
    id: modelId,
    object: 'model',
    created: patch.created || Math.floor(Date.now() / 1000),
    owned_by: patch.owned_by || 'llamacpp',
    meta: patch.meta || null,
    backendStatus: patch.backendStatus || null,
    raw: patch.raw || null,
  };
}

function setBackendDiscoveredModels(backend, models = [], error = null) {
  const next = new Map();
  for (const model of models) {
    const modelId = normalizeModelId(model?.id || model?.name || model?.model);
    if (!modelId) continue;
    next.set(modelId, createDiscoveredModelEntry(modelId, {
      created: model?.created,
      owned_by: model?.owned_by,
      meta: model?.meta || null,
      backendStatus: model?.status || null,
      raw: model,
    }));
  }

  backend.discoveredModels = next;
  backend.modelsDiscoveryAt = Date.now();
  backend.modelsDiscoveryError = error;
}

function getBackendDiscoveredModels(backend) {
  return [...backend.discoveredModels.values()];
}

function getBackendsForModel(modelId) {
  const normalizedModelId = normalizeModelId(modelId);
  if (!normalizedModelId) return [];
  return backends.filter(backend => backend.discoveredModels.has(normalizedModelId));
}

function isBackendDegraded(backend, now = Date.now()) {
  return Number.isFinite(backend.degradedUntil) && backend.degradedUntil > now;
}

function isBackendRoutable(backend, now = Date.now()) {
  return backend.healthy && !isBackendDegraded(backend, now);
}

function clearBackendDegradedState(backend) {
  backend.failureStreak = 0;
  backend.degradedUntil = 0;
  backend.lastFailureKind = null;
}

function markBackendRequestFailure(backend, kind, error = null) {
  backend.failureStreak = (backend.failureStreak || 0) + 1;
  backend.lastFailureKind = kind;

  if (kind === 'restore') {
    backend.restoreFailures = (backend.restoreFailures || 0) + 1;
  } else {
    backend.requestFailures = (backend.requestFailures || 0) + 1;
  }

  const cooldownMs = Math.min(
    BACKEND_FAILURE_COOLDOWN_MS * backend.failureStreak,
    BACKEND_FAILURE_MAX_COOLDOWN_MS
  );

  backend.degradedUntil = Date.now() + cooldownMs;
  backend.lastError = error?.message || kind;
  invalidateAffinityForBackend(backend.id);

  logWarn(
    `Backend degraded: ${backend.id} kind=${kind} cooldown=${Math.round(cooldownMs / 1000)}s`
      + (error?.message ? ` (${error.message})` : '')
  );
}

function markBackendRequestSuccess(backend) {
  clearBackendDegradedState(backend);
  backend.lastError = null;
}

function getHealthyBackendsForModel(modelId) {
  return getBackendsForModel(modelId).filter(backend => isBackendRoutable(backend));
}

function buildAggregatedModelRegistry() {
  const registry = new Map();

  for (const backend of backends) {
    for (const model of backend.discoveredModels.values()) {
      const existing = registry.get(model.id) || {
        id: model.id,
        object: 'model',
        created: model.created,
        owned_by: model.owned_by || 'llamacpp',
        meta: model.meta || null,
        backends: [],
      };

      if (!existing.meta && model.meta) {
        existing.meta = model.meta;
      }

      existing.backends.push({
        id: backend.id,
        url: backend.baseUrl,
        healthy: backend.healthy,
        routable: isBackendRoutable(backend),
        discoveredAt: backend.modelsDiscoveryAt,
        discoveryError: backend.modelsDiscoveryError,
        status: model.backendStatus,
      });
      registry.set(model.id, existing);
    }
  }

  for (const entry of registry.values()) {
    entry.healthyBackends = entry.backends.filter(backend => backend.routable);
    entry.loadedBackends = entry.backends.filter(backend => {
      const value = backend.status?.value;
      return value === 'loaded' || value === 'loading' || value === 'sleeping';
    });
    entry.available = entry.healthyBackends.length > 0;
  }

  return registry;
}

function getAggregatedModelEntry(modelId) {
  const normalizedModelId = normalizeModelId(modelId);
  if (!normalizedModelId) return null;
  return buildAggregatedModelRegistry().get(normalizedModelId) || null;
}

function toOpenAiModel(entry) {
  return {
    id: entry.id,
    object: 'model',
    created: entry.created || Math.floor(Date.now() / 1000),
    owned_by: entry.owned_by || 'llamacpp',
    meta: entry.meta || null,
    healthy_backends: entry.healthyBackends.length,
    loaded_backends: entry.loadedBackends.length,
    backends: entry.backends,
  };
}

async function fetchBackendModels(backend) {
  const res = await timedFetch(`${backend.baseUrl}/models`, BACKEND_HEALTHCHECK_TIMEOUT_MS);
  if (!res.ok) {
    const body = await res.text().catch(() => '(no body)');
    throw new Error(`models endpoint failed ${res.status}: ${body.slice(0, 200)}`);
  }

  const payload = await res.json();
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

async function fetchBackendOpenAiModels(backend) {
  try {
    const res = await timedFetch(`${backend.baseUrl}/v1/models`, BACKEND_HEALTHCHECK_TIMEOUT_MS);
    if (!res.ok) {
      return [];
    }

    const payload = await res.json();
    if (Array.isArray(payload?.data)) return payload.data;
    return [];
  } catch {
    return [];
  }
}

async function refreshBackendModels(backend) {
  try {
    const [models, openAiModels] = await Promise.all([
      fetchBackendModels(backend),
      fetchBackendOpenAiModels(backend),
    ]);

    const openAiById = new Map(openAiModels
      .map(model => [normalizeModelId(model?.id), model])
      .filter(([modelId]) => Boolean(modelId)));

    const enrichedModels = models.map(model => {
      const modelId = normalizeModelId(model?.id || model?.name || model?.model);
      const openAiModel = modelId ? openAiById.get(modelId) : null;
      if (!openAiModel) {
        return model;
      }

      return {
        ...model,
        created: model?.created || openAiModel.created,
        owned_by: model?.owned_by || openAiModel.owned_by,
        meta: model?.meta || openAiModel.meta || null,
      };
    });

    setBackendDiscoveredModels(backend, enrichedModels, null);
    return getBackendDiscoveredModels(backend);
  } catch (err) {
    setBackendDiscoveredModels(backend, [], err.message);
    throw err;
  }
}

function getBackendModelStatus(backend, modelId) {
  const normalizedModelId = normalizeModelId(modelId);
  if (!normalizedModelId) return null;
  return backend.discoveredModels.get(normalizedModelId)?.backendStatus || null;
}

function isBackendModelActive(status) {
  const value = status?.value;
  return value === 'loaded' || value === 'loading' || value === 'sleeping';
}

function isBackendModelReady(status) {
  const value = status?.value;
  return value === 'loaded' || value === 'sleeping';
}

function getBackendActiveModelIds(backend, excludeModelId = null) {
  const excludedModelId = normalizeModelId(excludeModelId);
  const activeModelIds = [];

  for (const model of backend.discoveredModels.values()) {
    if (excludedModelId && model.id === excludedModelId) continue;
    if (isBackendModelActive(model.backendStatus)) {
      activeModelIds.push(model.id);
    }
  }

  return activeModelIds;
}

async function waitForBackendModelCondition(backend, modelId, predicate, actionLabel, timeoutMs = MODEL_OPERATION_TIMEOUT_MS) {
  const normalizedModelId = normalizeModelId(modelId);
  const deadline = Date.now() + timeoutMs;
  let lastState = getBackendModelStatus(backend, normalizedModelId)?.value || 'unknown';
  let lastError = null;

  while (Date.now() < deadline) {
    try {
      await refreshBackendModels(backend);
      lastError = null;
    } catch (err) {
      lastError = err;
    }

    const status = getBackendModelStatus(backend, normalizedModelId);
    lastState = status?.value || 'unknown';
    if (predicate(status)) {
      return status;
    }

    await wait(MODEL_OPERATION_POLL_INTERVAL_MS);
  }

  const detail = lastError ? `, last refresh error: ${lastError.message}` : '';
  const err = new Error(`Timed out waiting for model ${normalizedModelId} on ${backend.id} to ${actionLabel} (last state=${lastState}${detail})`);
  err.name = 'BackendModelTimeoutError';
  err.backendId = backend.id;
  err.modelId = normalizedModelId;
  err.actionLabel = actionLabel;
  err.lastState = lastState;
  throw err;
}

function withBackendModelOperationLock(backend, operation) {
  const previous = backendModelOperationLocks.get(backend.id) || Promise.resolve();
  const current = previous.catch(() => {}).then(operation);
  const settled = current.finally(() => {
    if (backendModelOperationLocks.get(backend.id) === settled) {
      backendModelOperationLocks.delete(backend.id);
    }
  });

  backendModelOperationLocks.set(backend.id, settled);
  return current;
}

async function unloadModelOnBackend(backend, modelId) {
  const normalizedModelId = normalizeModelId(modelId);
  if (!normalizedModelId) {
    throw new Error('model is required');
  }

  await refreshBackendModels(backend).catch(() => {});
  const status = getBackendModelStatus(backend, normalizedModelId);
  if (!isBackendModelActive(status)) {
    return { model: normalizedModelId, status: status?.value || 'inactive' };
  }

  log('MODEL', `Unloading model=${normalizedModelId} backend=${backend.id}`);
  await llamaPost(backend, '/models/unload', { model: normalizedModelId });
  await waitForBackendModelCondition(
    backend,
    normalizedModelId,
    currentStatus => !isBackendModelActive(currentStatus),
    'unload'
  );

  return { model: normalizedModelId, status: 'unloaded' };
}

async function switchModelOnBackend(backend, modelId) {
  const normalizedModelId = normalizeModelId(modelId);
  if (!normalizedModelId) {
    throw new Error('model is required');
  }

  await refreshBackendModels(backend).catch(() => {});

  const currentStatus = getBackendModelStatus(backend, normalizedModelId);
  if (isBackendModelReady(currentStatus)) {
    return {
      model: normalizedModelId,
      status: currentStatus.value,
      unloadedModels: [],
      alreadyLoaded: true,
    };
  }

  if (!currentStatus) {
    throw new Error(`Model ${normalizedModelId} is not advertised by backend ${backend.id}`);
  }

  const unloadedModels = [];
  const activeModels = getBackendActiveModelIds(backend, normalizedModelId);
  for (const activeModelId of activeModels) {
    const result = await unloadModelOnBackend(backend, activeModelId);
    unloadedModels.push(result.model);
  }

  const refreshedStatus = getBackendModelStatus(backend, normalizedModelId);
  if (isBackendModelReady(refreshedStatus)) {
    return {
      model: normalizedModelId,
      status: refreshedStatus.value,
      unloadedModels,
      alreadyLoaded: true,
    };
  }

  if (!isBackendModelActive(refreshedStatus)) {
    log('MODEL', `Loading model=${normalizedModelId} backend=${backend.id}`);
    await llamaPost(backend, '/models/load', { model: normalizedModelId });
  }

  const readyStatus = await waitForBackendModelCondition(
    backend,
    normalizedModelId,
    status => isBackendModelReady(status),
    'load'
  ).catch(err => {
    if (err.name === 'BackendModelTimeoutError' && err.lastState === 'loading') {
      logWarn(`Model ${normalizedModelId} is still loading on ${backend.id} after ${Math.round(MODEL_OPERATION_TIMEOUT_MS / 1000)}s`);
      return { value: 'loading' };
    }
    throw err;
  });

  return {
    model: normalizedModelId,
    status: readyStatus?.value || 'loaded',
    unloadedModels,
    alreadyLoaded: false,
    pending: readyStatus?.value === 'loading',
  };
}

function createIdleActivitySnapshot(overrides = {}) {
  return {
    phase: 'idle',
    summary: 'Idle',
    detail: 'No recent backend activity.',
    updatedAt: 0,
    startedAt: null,
    requestId: null,
    taskId: null,
    cacheHash: null,
    restoredCacheHash: null,
    cacheStatus: null,
    partialScore: null,
    cacheReason: null,
    progress: null,
    tokens: null,
    ...overrides,
  };
}

function updateSlotActivity(backend, slotId, patch = {}) {
  if (!Number.isInteger(slotId) || slotId < 0 || slotId >= backend.maxSlots) {
    return createIdleActivitySnapshot();
  }

  const previous = backend.slotActivity.get(slotId) || createIdleActivitySnapshot();
  const next = {
    ...previous,
    ...patch,
    updatedAt: Date.now(),
  };
  backend.slotActivity.set(slotId, next);
  return next;
}

function getSlotActivitySnapshot(backend, slotId) {
  return {
    ...(backend.slotActivity.get(slotId) || createIdleActivitySnapshot()),
  };
}

function getBackendActivitySnapshot(backend) {
  const activities = [...backend.slotActivity.values()];
  if (activities.length === 0) {
    return createIdleActivitySnapshot();
  }

  const activeActivities = activities.filter(activity => activity.phase && activity.phase !== 'idle');
  const source = (activeActivities.length > 0 ? activeActivities : activities)
    .sort((left, right) => (right.updatedAt || 0) - (left.updatedAt || 0))[0];

  return {
    ...createIdleActivitySnapshot(),
    ...source,
  };
}

// ─── Affinity Index — bounded LRU to prevent unbounded memory growth ──

const MAX_AFFINITY_ENTRIES = 10_000;

function evictAffinityMap(map) {
  if (map.size > MAX_AFFINITY_ENTRIES) {
    const oldestKey = [...map.entries()].sort((a, b) => a[1].updatedAt - b[1].updatedAt)[0]?.[0];
    if (oldestKey) map.delete(oldestKey);
  }
}

const affinityIndex = {
  exact: new Map(),
  system: new Map(),
  prefix: new Map(),
};

// ─── Per-backend startup lock to prevent concurrent spawn races ──────

const backendStartupLocks = new Map();

function resetBackendRuntimeState(backend) {
  backend.slotQueue.length = 0;
  backend.availableSlots = Array.from({ length: backend.maxSlots }, (_, i) => i);
  backend.activeSlots.clear();
  backend.slotContextIndex.clear();
  backend.slotActivity.clear();
}

function getBackendProcessSnapshot(backend) {
  return {
    status: backend.processInfo.status,
    pid: backend.processInfo.pid,
    startedAt: backend.processInfo.startedAt,
    logPath: backend.processInfo.logPath,
    lastExitCode: backend.processInfo.lastExitCode,
    lastExitSignal: backend.processInfo.lastExitSignal,
    lastStartAttemptAt: backend.processInfo.lastStartAttemptAt,
    autoStartDisabled: backend.processInfo.autoStartDisabled,
  };
}

function isBackendProcessActive(backend) {
  return backend.processInfo.pid !== null && ['starting', 'running', 'stopping'].includes(backend.processInfo.status);
}

async function tailFile(path, lineCount = 120) {
  if (!existsSync(path)) return '';
  const content = await readFile(path, 'utf8');
  return content.split('\n').slice(-lineCount).join('\n').trim();
}

async function readTailBytes(path, maxBytes = 128 * 1024) {
  if (!existsSync(path)) return '';

  const fileStats = await safeStat(path);
  if (!fileStats || fileStats.size <= 0) return '';

  const start = Math.max(0, fileStats.size - maxBytes);
  const length = fileStats.size - start;
  const handle = await open(path, 'r');

  try {
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await handle.read(buffer, 0, length, start);
    return buffer.toString('utf8', 0, bytesRead);
  } finally {
    await handle.close();
  }
}

async function refreshBackendPrefillProgress(backend) {
  const activePrefillSlots = [...backend.slotActivity.entries()].filter(([, activity]) => (
    activity && ['prefill', 'prompt'].includes(activity.phase)
  ));

  if (activePrefillSlots.length === 0) {
    return;
  }

  const logTail = await readTailBytes(backend.processInfo.logPath);
  if (!logTail) {
    return;
  }

  const latestProgressBySlot = new Map();
  const progressPattern = /slot update_slots: id\s+(\d+)\s+\|\s+task\s+(\d+)\s+\|\s+prompt processing progress,.*?\bprogress\s*=\s*([0-9]*\.?[0-9]+)/;

  for (const line of logTail.split('\n')) {
    const match = line.match(progressPattern);
    if (!match) {
      continue;
    }

    const slotId = Number.parseInt(match[1], 10);
    const taskId = Number.parseInt(match[2], 10);
    const progress = Number.parseFloat(match[3]);
    if (!Number.isInteger(slotId) || !Number.isInteger(taskId) || !Number.isFinite(progress)) {
      continue;
    }

    latestProgressBySlot.set(slotId, {
      taskId,
      progress: Math.max(0, Math.min(progress, 1)),
    });
  }

  for (const [slotId, activity] of activePrefillSlots) {
    const parsed = latestProgressBySlot.get(slotId);
    if (!parsed) {
      continue;
    }

    if (Number.isInteger(activity.taskId) && activity.taskId !== parsed.taskId) {
      continue;
    }

    const currentProgress = Number.isFinite(activity.progress) ? activity.progress : 0;
    if (parsed.progress < currentProgress) {
      continue;
    }

    updateSlotActivity(backend, slotId, {
      taskId: parsed.taskId,
      progress: parsed.progress,
      detail: `Processing prompt (${Math.round(parsed.progress * 100)}%).`,
    });
  }
}

async function ensureBackendLogFile(backend) {
  if (!existsSync(BACKEND_LOG_DIR)) {
    await mkdir(BACKEND_LOG_DIR, { recursive: true });
  }

  if (!existsSync(backend.processInfo.logPath)) {
    await appendFile(backend.processInfo.logPath, '', 'utf8');
  }
}

async function startBackendProcess(backend, trigger = 'manual') {
  if (isBackendProcessActive(backend)) {
    return getBackendProcessSnapshot(backend);
  }

  await ensureBackendLogFile(backend);
  backend.processInfo.autoStartDisabled = false;
  backend.processInfo.status = 'starting';
  backend.processInfo.lastStartAttemptAt = Date.now();
  backend.processInfo.lastExitCode = null;
  backend.processInfo.lastExitSignal = null;

  const stdout = createWriteStream(backend.processInfo.logPath, { flags: 'a' });
  const stderr = createWriteStream(backend.processInfo.logPath, { flags: 'a' });
  stdout.write(`\n=== ${new Date().toISOString()} START ${backend.id} trigger=${trigger} ===\n`);

  const child = spawn('/bin/bash', [backend.scriptPath], {
    cwd: MODELS_DIR,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  });

  backend.processInfo.pid = child.pid;
  backend.processInfo.startedAt = Date.now();

  child.stdout.pipe(stdout);
  child.stderr.pipe(stderr);

  child.on('error', err => {
    backend.processInfo.status = 'error';
    backend.processInfo.lastExitSignal = null;
    backend.processInfo.lastExitCode = null;
    appendFile(backend.processInfo.logPath, `\n[manager:error] ${err.message}\n`, 'utf8').catch(err => logWarn(`File operation failed: ${err.message}`));
  });

  child.on('exit', (code, signal) => {
    backend.processInfo.pid = null;
    backend.processInfo.lastExitCode = code;
    backend.processInfo.lastExitSignal = signal;
    backend.processInfo.status = code === 0 || signal === 'SIGTERM' ? 'stopped' : 'error';
    resetBackendRuntimeState(backend);
    appendFile(
      backend.processInfo.logPath,
      `\n=== ${new Date().toISOString()} EXIT ${backend.id} code=${code} signal=${signal || 'none'} ===\n`,
      'utf8'
    ).catch(err => logWarn(`File operation failed: ${err.message}`));
  });

  log('PROC', `Starting backend=${backend.id} pid=${child.pid} trigger=${trigger}`);
  return getBackendProcessSnapshot(backend);
}

async function stopBackendProcess(backend, trigger = 'manual') {
  if (!backend.processInfo.pid) {
    backend.processInfo.status = 'stopped';
    if (trigger === 'dashboard' || trigger === 'manual') {
      backend.processInfo.autoStartDisabled = true;
    }
    return getBackendProcessSnapshot(backend);
  }

  backend.processInfo.status = 'stopping';
  if (trigger === 'dashboard' || trigger === 'manual') {
    backend.processInfo.autoStartDisabled = true;
  }
  try {
    process.kill(-backend.processInfo.pid, 'SIGTERM');
    log('PROC', `Stopping backend=${backend.id} pid=${backend.processInfo.pid} trigger=${trigger}`);
  } catch (err) {
    if (err.code === 'ESRCH') {
      backend.processInfo.pid = null;
      backend.processInfo.status = 'stopped';
    } else {
      throw err;
    }
  }

  return getBackendProcessSnapshot(backend);
}

async function restartBackendProcess(backend, trigger = 'manual') {
  await stopBackendProcess(backend, trigger);
  return startBackendProcess(backend, trigger);
}

async function ensureBackendProcess(backend, trigger = 'auto') {
  if (!BACKEND_AUTO_START || backend.processInfo.autoStartDisabled || isBackendProcessActive(backend)) {
    return;
  }

  let lock = backendStartupLocks.get(backend.id);
  if (lock) {
    try {
      await lock;
    } catch (err) {
      // Sibling thread failed to start, ignore the error and check state
    }
    if (isBackendProcessActive(backend)) return;
  }

  const lastAttemptAge = Date.now() - backend.processInfo.lastStartAttemptAt;
  if (lastAttemptAge >= 0 && lastAttemptAge < BACKEND_STARTUP_GRACE_MS) {
    return;
  }

  // Acquire lock to prevent concurrent spawns
  let resolveLock, rejectLock;
  const promiseLock = new Promise((resolve, reject) => {
    resolveLock = resolve;
    rejectLock = reject;
  });
  backendStartupLocks.set(backend.id, promiseLock);
  try {
    await startBackendProcess(backend, trigger);
    resolveLock();
  } catch (err) {
    if (rejectLock) rejectLock(err);
    throw err;
  } finally {
    backendStartupLocks.delete(backend.id);
  }
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function bootstrapBackendProcesses() {
  await Promise.all(backends.map(backend => ensureBackendProcess(backend, 'bootstrap')));
  await wait(Math.min(BACKEND_STARTUP_GRACE_MS, 4_000));
  await checkAllBackendsHealth();
}

function getHealthyBackends() {
  return backends.filter(backend => isBackendRoutable(backend));
}

function getTotalQueueLength() {
  return backends.reduce((sum, backend) => sum + backend.slotQueue.length, 0);
}

function pickAvailableSlot(backend, preferredSlotId = null) {
  if (preferredSlotId !== null) {
    const preferredIndex = backend.availableSlots.indexOf(preferredSlotId);
    if (preferredIndex !== -1) {
      return backend.availableSlots.splice(preferredIndex, 1)[0];
    }
  }

  return backend.availableSlots.shift();
}

function acquireSlotOnBackend(backend, reqId, signal, preferredSlotId = null) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      const err = new Error('aborted');
      err.name = 'AbortError';
      return reject(err);
    }

    if (backend.availableSlots.length > 0) {
      const slotId = pickAvailableSlot(backend, preferredSlotId);
      backend.activeSlots.set(reqId, slotId);
      log('SLOT', `${C.green}ACQUIRED${C.reset}  backend=${backend.id} slot=${slotId}  ${reqId}`);
      resolve(slotId);
    } else {
      log('SLOT', `${C.yellow}QUEUED${C.reset}  backend=${backend.id} queue=${backend.slotQueue.length + 1}  ${reqId}`);
      backend.slotQueue.push({ resolve, reject, reqId, signal, preferredSlotId });
    }
  });
}

function releaseSlotOnBackend(backend, reqId) {
  const queueIndex = backend.slotQueue.findIndex(item => item.reqId === reqId);
  if (queueIndex !== -1) {
    const [removed] = backend.slotQueue.splice(queueIndex, 1);
    const err = new Error('aborted');
    err.name = 'AbortError';
    removed.reject(err);
    log('SLOT', `${C.yellow}REMOVED FROM QUEUE${C.reset}  backend=${backend.id}  ${reqId}`);
    return;
  }

  const slotId = backend.activeSlots.get(reqId);
  if (slotId !== undefined && slotId !== null) {
    backend.activeSlots.delete(reqId);

    while (backend.slotQueue.length > 0) {
      let nextIndex = backend.slotQueue.findIndex(item => {
        if (item.signal?.aborted) return false;
        return item.preferredSlotId === null || item.preferredSlotId === slotId;
      });

      if (nextIndex === -1) {
        nextIndex = backend.slotQueue.findIndex(item => !item.signal?.aborted);
      }

      if (nextIndex === -1) {
        break;
      }

      const [next] = backend.slotQueue.splice(nextIndex, 1);
      if (!next.signal?.aborted) {
        backend.activeSlots.set(next.reqId, slotId);
        log('SLOT', `${C.green}PASSED${C.reset}  backend=${backend.id} slot=${slotId}  ${reqId} -> ${next.reqId}`);
        next.resolve(slotId);
        return;
      }

        const err = new Error('aborted');
        err.name = 'AbortError';
        next.reject(err);
    }

    backend.availableSlots.push(slotId);
    log('SLOT', `${C.green}RELEASED${C.reset}  backend=${backend.id} slot=${slotId}  ${reqId}`);
  }
}

function invalidateAffinityForBackend(backendId) {
  for (const map of [affinityIndex.exact, affinityIndex.system, affinityIndex.prefix]) {
    for (const [key, value] of map.entries()) {
      if (value.backendId === backendId) {
        map.delete(key);
      }
    }
  }
}

async function timedFetch(url, timeoutMs, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function checkBackendHealth(backend) {
  // If the backend has active slots running, it is guaranteed to be alive and busy.
  // We can skip the HTTP probe completely to avoid queueing requests that will time out,
  // preventing HTTP aborts and subsequent slot cancel warnings in llama.cpp logs.
  if (backend.activeSlots.size > 0) {
    backend.healthy = true;
    backend.lastError = null;
    backend.lastCheckAt = Date.now();
    return true;
  }

  const probePaths = ['/health', '/metrics', '/'];
  const wasHealthy = backend.healthy;
  let lastError = 'unreachable';

  for (const path of probePaths) {
    try {
      const res = await timedFetch(`${backend.baseUrl}${path}`, BACKEND_HEALTHCHECK_TIMEOUT_MS);
      if (res.ok || (path === '/' && res.status < 500)) {
        backend.healthy = true;
        if (backend.processInfo.status === 'starting' || backend.processInfo.status === 'stopped' || backend.processInfo.status === 'error') {
          backend.processInfo.status = 'running';
        }
        try {
          await refreshBackendModels(backend);
        } catch (err) {
          logWarn(`Model discovery failed for ${backend.id}: ${err.message}`);
        }
        if (backend.failureStreak > 0 && !isBackendDegraded(backend)) {
          clearBackendDegradedState(backend);
        }
        backend.lastError = null;
        backend.lastCheckAt = Date.now();
        if (!wasHealthy) {
          log('HEALTH', `${C.green}UP${C.reset}  backend=${backend.id}  ${backend.baseUrl}`);
        }
        return true;
      }
      lastError = `status=${res.status}`;
    } catch (err) {
      lastError = err.name === 'AbortError' ? 'timeout' : err.message;
    }
  }

  backend.healthy = false;
  backend.lastError = lastError;
  backend.lastCheckAt = Date.now();
  setBackendDiscoveredModels(backend, [], lastError);
  if (wasHealthy) {
    logWarn(`Backend down: ${backend.id} (${backend.baseUrl}) - ${lastError}`);
  }
  invalidateAffinityForBackend(backend.id);
  await ensureBackendProcess(backend, 'healthcheck');
  return false;
}

async function checkAllBackendsHealth() {
  await Promise.all(backends.map(checkBackendHealth));
}

function chooseRoundRobin(backendsPool) {
  if (backendsPool.length === 0) return null;
  const idx = roundRobinCounter % backendsPool.length;
  roundRobinCounter = (roundRobinCounter + 1) % Number.MAX_SAFE_INTEGER;
  return backendsPool[idx];
}

function hasFreeSlot(backend, preferredSlotId = null) {
  if (preferredSlotId !== null && backend.availableSlots.includes(preferredSlotId)) {
    return true;
  }
  return backend.availableSlots.length > 0;
}

function normalizeText(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function takeTokenPrefix(text, maxTokens = 128) {
  const tokens = normalizeText(text).split(' ').filter(Boolean);
  return tokens.slice(0, maxTokens).join(' ');
}

function prefixSimilarityScore(left, right) {
  const leftTokens = takeTokenPrefix(left, 256).split(' ').filter(Boolean);
  const rightTokens = takeTokenPrefix(right, 256).split(' ').filter(Boolean);
  if (leftTokens.length === 0 || rightTokens.length === 0) {
    return 0;
  }

  const maxLength = Math.max(leftTokens.length, rightTokens.length);
  let matched = 0;
  const limit = Math.min(leftTokens.length, rightTokens.length);
  while (matched < limit && leftTokens[matched] === rightTokens[matched]) {
    matched++;
  }

  return matched / maxLength;
}

function getLoadedSlotMatch(backend, requestContext) {
  let bestMatch = null;
  for (const [slotId, slotContext] of backend.slotContextIndex.entries()) {
    if (!backend.availableSlots.includes(slotId)) continue;

    if (requestContext.exactHash && slotContext.exactHash === requestContext.exactHash) {
      return { slotId, reason: 'exact-hot', score: 1 };
    }

    if (!bestMatch && requestContext.systemHash && slotContext.systemHash === requestContext.systemHash) {
      bestMatch = { slotId, reason: 'system-hot', score: 0.95 };
      continue;
    }

    if (requestContext.normalizedPrefix && slotContext.prefixText) {
      const score = prefixSimilarityScore(requestContext.normalizedPrefix, slotContext.prefixText);
      if (score >= 0.8 && (!bestMatch || score > bestMatch.score)) {
        bestMatch = { slotId, reason: 'prefix-hot', score };
      }
    }
  }

  return bestMatch;
}

function pickQueueBackend(healthyBackends, preferredBackend = null) {
  if (preferredBackend && healthyBackends.includes(preferredBackend)) {
    return preferredBackend;
  }

  const sorted = [...healthyBackends].sort((left, right) => left.slotQueue.length - right.slotQueue.length);
  const shortestLength = sorted[0]?.slotQueue.length;
  const shortest = sorted.filter(backend => backend.slotQueue.length === shortestLength);
  return chooseRoundRobin(shortest);
}

function findAffinityPreference(requestContext) {
  const healthyBackends = requestContext.model
    ? getHealthyBackendsForModel(requestContext.model)
    : getHealthyBackends();
  const isHealthy = backendId => healthyBackends.some(backend => backend.id === backendId);

  if (requestContext.exactHash) {
    const exact = affinityIndex.exact.get(requestContext.exactHash);
    if (exact && isHealthy(exact.backendId)) {
      return {
        backend: backendsById.get(exact.backendId),
        preferredSlotId: exact.slotId,
        reason: 'exact-affinity',
      };
    }
  }

  if (requestContext.systemHash) {
    const system = affinityIndex.system.get(requestContext.systemHash);
    if (system && isHealthy(system.backendId)) {
      return {
        backend: backendsById.get(system.backendId),
        preferredSlotId: system.slotId,
        reason: 'system-affinity',
      };
    }
  }

  let bestPrefix = null;
  if (requestContext.normalizedPrefix) {
    for (const candidate of affinityIndex.prefix.values()) {
      if (!isHealthy(candidate.backendId)) continue;
      const score = prefixSimilarityScore(requestContext.normalizedPrefix, candidate.prefixText);
      if (score >= 0.8 && (!bestPrefix || score > bestPrefix.score)) {
        bestPrefix = {
          backend: backendsById.get(candidate.backendId),
          preferredSlotId: candidate.slotId,
          reason: 'prefix-affinity',
          score,
        };
      }
    }
  }

  if (bestPrefix) {
    return bestPrefix;
  }

  return {
    backend: null,
    preferredSlotId: null,
    reason: 'round-robin',
  };
}

function chooseBackendForRequest(requestContext) {
  const healthyBackends = requestContext.model
    ? getHealthyBackendsForModel(requestContext.model)
    : getHealthyBackends();

  if (requestContext.model) {
    const knownBackends = getBackendsForModel(requestContext.model);
    if (knownBackends.length === 0) {
      const err = new Error(`Unknown model: ${requestContext.model}`);
      err.statusCode = 400;
      err.errorType = 'invalid_request_error';
      throw err;
    }
  }

  if (healthyBackends.length === 0) {
    const err = new Error(requestContext.model
      ? `No healthy backends available for model: ${requestContext.model}`
      : 'No healthy llama backends available');
    err.statusCode = 503;
    err.errorType = 'unavailable_error';
    throw err;
  }

  const preference = findAffinityPreference(requestContext);
  const freeBackends = healthyBackends.filter(backend => hasFreeSlot(backend));

  if (freeBackends.length === 0) {
    return {
      backend: pickQueueBackend(healthyBackends, preference.backend),
      preferredSlotId: preference.preferredSlotId,
      reason: `${preference.reason}-queued`,
      queued: true,
    };
  }

  if (preference.backend && hasFreeSlot(preference.backend, preference.preferredSlotId)) {
    return {
      backend: preference.backend,
      preferredSlotId: preference.preferredSlotId,
      reason: preference.reason,
      queued: false,
    };
  }

  const hotAlternatives = freeBackends
    .map(backend => ({ backend, match: getLoadedSlotMatch(backend, requestContext) }))
    .filter(candidate => candidate.match !== null);

  if (hotAlternatives.length > 0) {
    hotAlternatives.sort((left, right) => right.match.score - left.match.score);
    return {
      backend: hotAlternatives[0].backend,
      preferredSlotId: hotAlternatives[0].match.slotId,
      reason: `${preference.reason}-spillover-hot`,
      queued: false,
    };
  }

  if (preference.backend) {
    const spilloverPool = freeBackends.filter(backend => backend.id !== preference.backend.id);
    const target = chooseRoundRobin(spilloverPool.length > 0 ? spilloverPool : freeBackends);
    return {
      backend: target,
      preferredSlotId: null,
      reason: `${preference.reason}-spillover-restore`,
      queued: false,
    };
  }

  return {
    backend: chooseRoundRobin(freeBackends),
    preferredSlotId: null,
    reason: 'round-robin',
    queued: false,
  };
}

function upsertAffinity(requestContext, backendId, slotId) {
  const entry = {
    backendId,
    slotId,
    model: requestContext.model,
    prefixText: requestContext.normalizedPrefix,
    updatedAt: Date.now(),
  };

  // Evict oldest entries before inserting to prevent unbounded growth
  evictAffinityMap(affinityIndex.exact);
  evictAffinityMap(affinityIndex.system);
  evictAffinityMap(affinityIndex.prefix);

  if (requestContext.exactHash) {
    affinityIndex.exact.set(requestContext.exactHash, entry);
  }
  if (requestContext.systemHash) {
    affinityIndex.system.set(requestContext.systemHash, entry);
  }
  if (requestContext.prefixSignature) {
    affinityIndex.prefix.set(requestContext.prefixSignature, entry);
  }
}

function clearSlotContext(backend, slotId) {
  if (!Number.isInteger(slotId) || slotId < 0 || slotId >= backend.maxSlots) {
    return;
  }

  const previous = backend.slotContextIndex.get(slotId);
  if (!previous) {
    return;
  }

  backend.slotContextIndex.delete(slotId);

  for (const map of [affinityIndex.exact, affinityIndex.system, affinityIndex.prefix]) {
    for (const [key, value] of map.entries()) {
      if (value.backendId === backend.id && value.slotId === slotId) {
        map.delete(key);
      }
    }
  }
}

function rememberSlotContext(backend, slotId, requestContext, source) {
  if (slotId === null || slotId === undefined) return;

  backend.slotContextIndex.set(slotId, {
    exactHash: requestContext.exactHash,
    systemHash: requestContext.systemHash,
    prefixSignature: requestContext.prefixSignature,
    prefixText: requestContext.normalizedPrefix,
    source,
    updatedAt: Date.now(),
  });
  upsertAffinity(requestContext, backend.id, slotId);
}

// Global deduplication map: hash -> Array of resolver functions
const inFlightMisses = new Map();

// ─── Active cache file tracking — prevents GC from deleting files in use ─

const activeCacheFiles = new Set();  // Set of hashes currently being accessed
const inFlightDiskCopies = new Map(); // hash -> Promise (prevents concurrent identical L1 to L2 saves and GC evictions)

function markCacheActive(hash) {
  activeCacheFiles.add(hash);
}

function markCacheInactive(hash) {
  activeCacheFiles.delete(hash);
}

// ─── 2. L1 Eviction Sweeper ─────────────────────────────────────────

async function sweepL1Eviction() {
  try {
    const files = await safeReaddir(L1_DIR);
    const binFiles = files.filter(f => extname(f) === '.bin');

    // Calculate total L1 size — skip files currently in use or copying
    let totalSize = 0;
    const fileInfo = [];
    for (const f of binFiles) {
      const hash = f.replace(/\.bin$/, '');
      if (activeCacheFiles.has(hash) || inFlightDiskCopies.has(hash)) continue;

      const s = await safeStat(join(L1_DIR, f));
      if (s) {
        totalSize += s.size;
        fileInfo.push({ name: f, size: s.size, mtimeMs: s.mtimeMs, hash });
      }
    }

    if (totalSize <= L1_EVICT_THRESHOLD) {
      return;  // Under threshold, nothing to do
    }

    log('EVICT', `${C.bold}L1 over threshold${C.reset}  ${totalSize / 1024 / 1024 / 1024.0}GB > ${L1_EVICT_THRESHOLD / 1024 / 1024 / 1024.0}GB`);

    // Sort oldest first, evict until under target
    fileInfo.sort((a, b) => a.mtimeMs - b.mtimeMs);

    let evicted = 0;
    let freed = 0;
    for (const fi of fileInfo) {
      if (totalSize <= L1_EVICT_TARGET) break;
      // Race guard: skip if file became active or started copying since we scanned
      if (activeCacheFiles.has(fi.hash) || inFlightDiskCopies.has(fi.hash)) continue;
      const fpath = join(L1_DIR, fi.name);
      await rm(fpath, { force: true });  // atomic: no-op if file gone
      evicted++;
      freed += fi.size;
      totalSize -= fi.size;
      log('EVICT', `  ${C.red}DELETED${C.reset}  ${fi.name}  (${fi.size / 1024 / 1024}KB)`);
    }

    log('EVICT', `${C.green}Sweep done${C.reset}  evicted=${evicted}  freed=${freed / 1024 / 1024}MB  remaining=${totalSize / 1024 / 1024 / 1024.0}GB`);
  } catch (err) {
    logErr(`L1 eviction sweep failed: ${err.message}`);
  }
}

// LRU GC sweeper (L1 only — deletes old files regardless of size)
async function gcSweep() {
  log('GC', `${C.bold}Starting sweep${C.reset}  (max age=${(MAX_FILE_AGE_MS / 3600000).toFixed(0)}h)`);
  try {
    const files = await safeReaddir(L1_DIR);
    const binFiles = files.filter(f => extname(f) === '.bin');
    const now = Date.now();
    let deleted = 0;

    for (const f of binFiles) {
      const hash = f.replace(/\.bin$/, '');
      // Skip files currently in use by a route handler or actively copying
      if (activeCacheFiles.has(hash) || inFlightDiskCopies.has(hash)) continue;

      const s = await safeStat(join(L1_DIR, f));
      if (s && now - s.mtimeMs > MAX_FILE_AGE_MS) {
        const fpath = join(L1_DIR, f);
        await rm(fpath, { force: true });  // atomic: no-op if file gone
        deleted++;
        log('GC', `  ${C.red}DELETED${C.reset}  ${f}  (age=${Math.round((now - s.mtimeMs) / 3600000)}h)`);
      }
    }

    if (deleted > 0) {
      log('GC', `${C.green}Sweep done${C.reset}  deleted=${deleted}`);
    }
  } catch (err) {
    logErr(`GC sweep failed: ${err.message}`);
  }
}

// ─── 3. File Copy Utilities ─────────────────────────────────────────

async function copyFileL2ToL1(hash) {
  const src = join(L2_DIR, `${hash}.bin`);
  const dst = join(L1_DIR, `${hash}.bin`);
  await copyFile(src, dst);
  log('COPY', `${C.green}L2→L1${C.reset}  ${hash}.bin`);
}

// ─── 4. llama.cpp API Helpers ───────────────────────────────────────

const HOP_BY_HOP_HEADERS = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailers',
  'transfer-encoding',
  'upgrade',
  'content-length',
]);

function sha256(input) {
  return createHash('sha256').update(input, 'utf8').digest('hex');
}

async function llamaPost(backend, path, payload) {
  const res = await fetch(`${backend.baseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '(no body)');
    throw new Error(`llama.cpp ${backend.id}${path} failed ${res.status}: ${text.slice(0, 300)}`);
  }

  return res;
}

function buildProxyRequestHeaders(sourceHeaders, backend) {
  const headers = {};
  for (const [key, value] of Object.entries(sourceHeaders || {})) {
    const lower = key.toLowerCase();
    if (HOP_BY_HOP_HEADERS.has(lower)) continue;
    if (value === undefined) continue;
    headers[key] = Array.isArray(value) ? value.join(', ') : String(value);
  }

  headers.Host = `${backend.host}:${backend.port}`;
  headers['Content-Type'] = headers['Content-Type'] || headers['content-type'] || 'application/json';
  return headers;
}

function buildProxyResponseHeaders(headerEntries, extraHeaders = {}) {
  const headers = {};
  for (const [key, value] of headerEntries.entries()) {
    if (HOP_BY_HOP_HEADERS.has(key.toLowerCase())) continue;
    headers[key] = value;
  }

  return {
    ...headers,
    ...extraHeaders,
  };
}

async function fetchChatCompletion(backend, requestHeaders, payload, signal) {
  return fetch(`${backend.baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: buildProxyRequestHeaders(requestHeaders, backend),
    body: JSON.stringify(payload),
    signal,
  });
}

/**
 * Save KV cache — llama.cpp saves to its internal cache dir.
 * We copy the result to L1_DIR (RAMDisk), then persist to L2_DIR (SSD) in background.
 *
 * @returns {{ filename: string, cacheDir: string }}
 *   cacheDir = where llama.cpp saved the file (its internal cache dir)
 */
async function saveKVCache(hash, model, backend, slotId) {
  const filename = `${hash}.bin`;

  try {
    log('SAVE', `Starting  →  ${filename}  backend=${backend.id}`);

    const res = await llamaPost(backend, `/slots/${slotId}?action=save`, { filename, model });
    await res.text();

    log('SAVE', `${C.green}Saved${C.reset}  →  L1 RAMDisk (backend=${backend.id} slot=${slotId})`);
    return { filename, backendId: backend.id };
  } catch (err) {
    logErr(`Save failed for ${hash}.bin (backend=${backend.id} slot=${slotId}): ${err.message}`);
    return null;
  }
}

/**
 * Restore KV cache — copy from L1_DIR to llama.cpp's cache dir,
 * call the restore endpoint, then clean up the temp copy.
 *
 * @param {string} hash - Cache hash
 * @param {string} model - Model name
 */
async function restoreKVCache(hash, model, backend, slotId) {
  const filename = `${hash}.bin`;

  log('RESTORE', `Loading  →  ${filename}  backend=${backend.id} slot=${slotId}  (from L1 RAMDisk)`);

  try {
    const res = await llamaPost(backend, `/slots/${slotId}?action=restore`, { filename, model });
    log('RESTORE', `${C.green}Success${C.reset}  →  ${filename}  (backend=${backend.id} slot=${slotId}, ${res.status})`);
    return true;
  } catch (err) {
    markBackendRequestFailure(backend, 'restore', err);
    logErr(`Restore failed for ${hash}.bin (backend=${backend.id} slot=${slotId}): ${err.message}`);
    throw err;
  }
}

// ─── 5. SSE Streaming Proxy ────────────────────────────────────────

function createSlotTracker(onSlot) {
  let buffer = '';

  return (chunk, onEvent) => {
    buffer += Buffer.from(chunk).toString('utf8');
    if (buffer.length > 65536) {
      buffer = buffer.slice(-32768);
    }

    while (true) {
      const boundary = buffer.indexOf('\n\n');
      if (boundary === -1) break;

      const rawEvent = buffer.slice(0, boundary).replace(/\r/g, '');
      buffer = buffer.slice(boundary + 2);

      for (const line of rawEvent.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;

        try {
          const parsed = JSON.parse(payload);
          const slotId = parsed.slot_id ?? parsed.id_slot;
          if (Number.isInteger(slotId)) {
            onSlot(slotId);
          }
          if (onEvent) {
            onEvent(parsed);
          }
        } catch {
          // Ignore partial or non-JSON SSE frames.
        }
      }
    }
  };
}

function normalizeSlotId(slotId, fallbackSlotId, backend) {
  if (Number.isInteger(slotId) && slotId >= 0 && slotId < backend.maxSlots) {
    return slotId;
  }
  return fallbackSlotId;
}

async function proxyStreamWithHeaders(llamaRes, reply, extraHeaders, signal, onChunk) {
  reply.hijack();
  reply.raw.writeHead(llamaRes.status, extraHeaders);

  const reader = llamaRes.body?.getReader();
  if (reader) {
    try {
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        if (signal.aborted) break;
        if (onChunk) onChunk(value);

        // Handle backpressure: if write() returns false, wait for drain
        const canContinue = reply.raw.write(value);
        if (!canContinue) {
          await new Promise(resolve => {
            const finishWaiting = () => {
              reply.raw.off('drain', finishWaiting);
              reply.raw.off('close', finishWaiting);
              reply.raw.off('error', finishWaiting);
              resolve();
            };

            reply.raw.once('drain', finishWaiting);
            reply.raw.once('close', finishWaiting);
            reply.raw.once('error', finishWaiting);
          });
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        logErr(`Stream read error: ${err.message}`);
      }
    } finally {
      reader.releaseLock();
      if (!reply.raw.writableEnded) {
        reply.raw.end();
      }
    }
  } else {
    reply.raw.end();
  }
}

const observedLlamaMetrics = {
  'llamacpp:n_tokens_max': 0,
  'llamacpp:prompt_tokens_total': 0,
  'llamacpp:prompt_seconds_total': 0,
  'llamacpp:tokens_predicted_total': 0,
  'llamacpp:tokens_predicted_seconds_total': 0,
  'llamacpp:prompt_tokens_seconds': 0,
  'llamacpp:predicted_tokens_seconds': 0,
  'kvbridge:context_overflow_total': 0,
  'kvbridge:cache_save_skipped_total': 0,
};

function extractErrorMessage(payload) {
  if (!payload) return '';
  if (typeof payload === 'string') return payload;
  if (typeof payload?.error?.message === 'string') return payload.error.message;
  if (typeof payload?.message === 'string') return payload.message;
  return '';
}

function isContextOverflowMessage(message) {
  return /exceeds the available context size|context size|context length|n_ctx|prompt is too long|too large for the context|too many tokens/i.test(message || '');
}

function classifyContextOverflow(payload, statusCode = null) {
  const message = extractErrorMessage(payload).trim();
  const overflow = isContextOverflowMessage(message)
    || (Number(statusCode) === 400 && isContextOverflowMessage(JSON.stringify(payload || {})));

  return {
    overflow,
    message,
  };
}

function payloadHasLengthFinishReason(payload) {
  const choices = Array.isArray(payload?.choices) ? payload.choices : [];
  return choices.some(choice => choice?.finish_reason === 'length');
}

function markContextOverflow(cacheHash, source, detail = '', skippedSave = false) {
  observedLlamaMetrics['kvbridge:context_overflow_total'] += 1;
  if (skippedSave) {
    observedLlamaMetrics['kvbridge:cache_save_skipped_total'] += 1;
  }
  const hashLabel = cacheHash ? cacheHash.slice(0, 8) : 'no-hash';
  const message = detail ? `source=${source} hash=${hashLabel} detail=${detail}` : `source=${source} hash=${hashLabel}`;
  logWarn(`Context overflow detected; cache save skipped (${message})`);
}

function markCacheSaveSkipped(reason, cacheHash) {
  observedLlamaMetrics['kvbridge:cache_save_skipped_total'] += 1;
  const hashLabel = cacheHash ? cacheHash.slice(0, 8) : 'no-hash';
  logWarn(`Cache save skipped (${reason}) for hash=${hashLabel}`);
}

function normalizePositiveNumber(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue >= 0 ? numericValue : null;
}

function recordObservedChatMetrics(payload) {
  const usage = payload?.usage || null;
  const timings = payload?.timings || null;

  const promptTokens = normalizePositiveNumber(usage?.prompt_tokens ?? timings?.prompt_n);
  const completionTokens = normalizePositiveNumber(usage?.completion_tokens ?? timings?.predicted_n);
  const totalTokens = normalizePositiveNumber(usage?.total_tokens)
    ?? ((promptTokens !== null || completionTokens !== null)
      ? ((promptTokens || 0) + (completionTokens || 0))
      : null);
  const promptSeconds = normalizePositiveNumber(timings?.prompt_ms) !== null
    ? normalizePositiveNumber(timings?.prompt_ms) / 1000
    : null;
  const predictedSeconds = normalizePositiveNumber(timings?.predicted_ms) !== null
    ? normalizePositiveNumber(timings?.predicted_ms) / 1000
    : null;
  const promptRate = normalizePositiveNumber(timings?.prompt_per_second);
  const predictedRate = normalizePositiveNumber(timings?.predicted_per_second);

  if (totalTokens !== null) {
    observedLlamaMetrics['llamacpp:n_tokens_max'] = Math.max(
      observedLlamaMetrics['llamacpp:n_tokens_max'],
      totalTokens
    );
  }
  if (promptTokens !== null) {
    observedLlamaMetrics['llamacpp:prompt_tokens_total'] += promptTokens;
  }
  if (promptSeconds !== null) {
    observedLlamaMetrics['llamacpp:prompt_seconds_total'] += promptSeconds;
  }
  if (completionTokens !== null) {
    observedLlamaMetrics['llamacpp:tokens_predicted_total'] += completionTokens;
  }
  if (predictedSeconds !== null) {
    observedLlamaMetrics['llamacpp:tokens_predicted_seconds_total'] += predictedSeconds;
  }
  if (promptRate !== null) {
    observedLlamaMetrics['llamacpp:prompt_tokens_seconds'] = promptRate;
  }
  if (predictedRate !== null) {
    observedLlamaMetrics['llamacpp:predicted_tokens_seconds'] = predictedRate;
  }
}

// ─── 6. Request Processing ─────────────────────────────────────────

function extractRequestContext(body) {
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  const promptText = typeof body?.prompt === 'string'
    ? body.prompt
    : (body?.prompt ? JSON.stringify(body.prompt) : '');
  let systemPrompt = '';
  const userMessages = [];
  let firstUserMessage = null;

  let systemFound = false;
  for (const msg of messages) {
    if (!systemFound && msg.role === 'system' && msg.content) {
      systemPrompt = typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
      systemFound = true;
    } else if (msg.role === 'user' && msg.content && firstUserMessage === null) {
      firstUserMessage = {
        role: 'user',
        content: typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content),
      };
      userMessages.push(msg);
    } else {
      userMessages.push(msg);
    }
  }

  const requestedModel = normalizeModelId(body?.model);
  const normalizedSystemPrompt = normalizeText(systemPrompt);
  const subagentRequest = isSubagentPrompt(systemPrompt);
  const model = subagentRequest ? SUBAGENT_MODEL : requestedModel;
  const affinitySource = promptText || [systemPrompt, firstUserMessage?.content || ''].filter(Boolean).join('\n\n');
  const normalizedPrefix = takeTokenPrefix(affinitySource, 256);
  const modelPrefix = model ? `${model}_` : '';
  const exactHash = normalizedPrefix ? sha256(`${modelPrefix}${normalizedPrefix}`) : null;
  const systemHash = normalizedSystemPrompt ? sha256(`${modelPrefix}${normalizedSystemPrompt}`) : null;
  const prefixSignature = normalizedPrefix ? sha256(`${modelPrefix}${takeTokenPrefix(normalizedPrefix, 128)}`) : null;

  return {
    model,
    requestedModel,
    promptText,
    systemPrompt,
    normalizedSystemPrompt,
    firstUserMessage,
    userMessages,
    normalizedPrefix,
    exactHash,
    systemHash,
    prefixSignature,
    hasAffinity: Boolean(normalizedPrefix || normalizedSystemPrompt),
    canUseCache: Boolean(systemPrompt && exactHash),
    subagentRequest,
  };
}

// ─── 9. Server Startup & Pre-Warm ──────────────────────────────────

/**
 * Pre-warm L1 (RAMDisk) from L2 (SSD) on startup.
 * Copies the N most recently modified .bin files.
 */
async function warmupL1() {
  try {
    const files = await safeReaddir(L2_DIR);
    const binFiles = files.filter(f => extname(f) === '.bin');

    if (binFiles.length === 0) {
      log('WARMUP', `${C.green}No .bin files in L2 — cache is cold${C.reset}`);
      return;
    }

    // Sort by mtime descending (most recent first)
    const stats = [];
    for (const f of binFiles) {
      const s = await safeStat(join(L2_DIR, f));
      if (s) stats.push({ name: f, mtimeMs: s.mtimeMs });
    }
    stats.sort((a, b) => b.mtimeMs - a.mtimeMs);

    const toCopy = stats.slice(0, WARMUP_TOP_N);
    let copied = 0;

    for (const item of toCopy) {
      try {
        await copyFile(join(L2_DIR, item.name), join(L1_DIR, item.name));
        copied++;
      } catch (err) {
        logWarn(`Warmup copy failed for ${item.name}: ${err.message}`);
      }
    }

    log('WARMUP', `${C.green}Pre-warmed${C.reset}  ${copied}/${stats.length} files → L1 (${L1_DIR})`);
  } catch (err) {
    logWarn(`Warmup failed: ${err.message}`);
  }
}

// ─── Fastify Server ─────────────────────────────────────────────────

const server = fastify({
  bodyLimit: 50 * 1024 * 1024,   // 50 MB
  maxParamLength: 1024 * 1024,
  disableRequestLogging: true,
});

server.register(fastifyCors, { origin: true });
server.register(fastifyFormBody);

// ─── 7. Route Handler ──────────────────────────────────────────────

function parsePrometheusMetrics(text) {
  const metrics = {};
  text.split('\n').forEach(line => {
    if (!line || line.startsWith('#') || !line.includes(' ')) return;
    const [key, rawValue] = line.split(' ');
    const numericValue = parseFloat(rawValue);
    if (!Number.isNaN(numericValue)) {
      metrics[key] = numericValue;
    }
  });
  return metrics;
}

function getStatusArgValue(status, flagName) {
  const args = Array.isArray(status?.args) ? status.args : [];
  for (let index = 0; index < args.length - 1; index++) {
    if (args[index] === flagName) {
      return args[index + 1];
    }
  }
  return null;
}

function getBackendMetricsTargets(backend) {
  const targets = new Map();

  for (const model of backend.discoveredModels.values()) {
    const status = model.backendStatus;
    const state = status?.value;
    if (state !== 'loaded' && state !== 'loading' && state !== 'sleeping') {
      continue;
    }

    const rawPort = getStatusArgValue(status, '--port') || status?.port;
    const port = Number.parseInt(rawPort, 10);
    if (!Number.isInteger(port) || port <= 0) {
      continue;
    }

    const url = `http://${backend.host}:${port}`;
    if (!targets.has(url)) {
      targets.set(url, { url, port, source: 'worker' });
    }
  }

  if (targets.size === 0) {
    targets.set(backend.baseUrl, {
      url: backend.baseUrl,
      port: backend.port,
      source: 'control',
    });
  }

  return [...targets.values()];
}

function mergeMetrics(target, source) {
  for (const [key, value] of Object.entries(source)) {
    target[key] = (target[key] || 0) + value;
  }
  return target;
}

function rememberBackendMetrics(backend, metrics) {
  if (!metrics || Object.keys(metrics).length === 0) {
    return;
  }

  backend.lastMetrics = { ...metrics };
  backend.lastMetricsAt = Date.now();
}

async function fetchMetricsFromTarget(baseUrl) {
  try {
    const response = await timedFetch(`${baseUrl}/metrics`, BACKEND_HEALTHCHECK_TIMEOUT_MS);
    if (!response.ok) return {};
    return parsePrometheusMetrics(await response.text());
  } catch {
    return {};
  }
}

async function fetchBackendMetrics(backend, options = {}) {
  const { allowCached = true } = options;
  const metricsTargets = getBackendMetricsTargets(backend);
  const metricsSets = await Promise.all(metricsTargets.map(target => fetchMetricsFromTarget(target.url)));

  const aggregatedMetrics = {};
  for (const metrics of metricsSets) {
    mergeMetrics(aggregatedMetrics, metrics);
  }

  if (Object.keys(aggregatedMetrics).length > 0) {
    rememberBackendMetrics(backend, aggregatedMetrics);
    return aggregatedMetrics;
  }

  if (allowCached && backend.lastMetricsAt > 0) {
    return backend.lastMetrics;
  }

  return aggregatedMetrics;
}

async function captureLiveBackendMetrics(backend) {
  try {
    await refreshBackendModels(backend);
    await fetchBackendMetrics(backend, { allowCached: false });
  } catch (err) {
    logWarn(`Live metrics capture failed for ${backend.id}: ${err.message}`);
  }
}


// ─── GPU Telemetry ─────────────────────────────────────────────────

async function collectGPUMetrics() {
  try {
    const { stdout } = await new Promise((resolve, reject) => {
      const proc = spawn('nvidia-smi', [
        '--query-gpu=index,name,temperature.gpu,fan.speed,utilization.gpu,power.draw,memory.used,memory.total,pstate',
        '--format=csv,noheader,nounits',
      ]);
      let out = '';
      let err = '';
      proc.stdout.on('data', d => { out += d; });
      proc.stderr.on('data', d => { err += d; });
      proc.on('close', code => {
        if (code !== 0) reject(new Error(err.trim() || 'nvidia-smi exited ' + code));
        else resolve({ stdout: out });
      });
      proc.on('error', reject);
    });

    const devices = stdout.trim().split('\n').map(line => {
      const cols = line.split(',').map(c => c.trim());
      return {
        index: parseInt(cols[0], 10),
        name: cols[1],
        temperatureC: parseFloat(cols[2]),
        fanSpeedPercent: parseInt(cols[3], 10),
        utilizationGpu: parseFloat(cols[4]),
        powerDrawWatts: parseFloat(cols[5]),
        memoryUsedMiB: parseFloat(cols[6]),
        memoryTotalMiB: parseFloat(cols[7]),
        pstate: cols[8],
      };
    });

    const summary = {
      averageTempC: Math.round(devices.reduce((s, d) => s + d.temperatureC, 0) / devices.length),
      totalPowerWatts: Math.round(devices.reduce((s, d) => s + d.powerDrawWatts, 0)),
      busiestUtilization: Math.round(Math.max(...devices.map(d => d.utilizationGpu))),
    };

    return { devices, summary };
  } catch (err) {
    return { error: err.message };
  }
}
server.get('/dashboard', async (request, reply) => {
  try {
    const html = await readFile(join(__dirname, 'dashboard.html'), 'utf8');
    reply.type('text/html').send(html);
  } catch (e) {
    reply.status(500).send('Dashboard UI not found');
  }
});

server.get('/api/stats', async (request, reply) => {
  let l1UsageBytes = 0;
  const gpuMetrics = await collectGPUMetrics();
  await Promise.all(backends.map(backend => refreshBackendPrefillProgress(backend)));
  try {
    const files = await safeReaddir(L1_DIR);
    for (const f of files) {
      if (extname(f) === '.bin') {
        const s = await safeStat(join(L1_DIR, f));
        if (s) l1UsageBytes += s.size;
      }
    }
  } catch (e) {}

  const backendSnapshots = await Promise.all(backends.map(async backend => {
    const metrics = await fetchBackendMetrics(backend);
    return { backend, metrics };
  }));

  const llamaMetrics = {};
  for (const snapshot of backendSnapshots) {
    for (const [key, value] of Object.entries(snapshot.metrics)) {
      llamaMetrics[key] = (llamaMetrics[key] || 0) + value;
    }
  }
  for (const [key, value] of Object.entries(observedLlamaMetrics)) {
    if (!(key in llamaMetrics) || llamaMetrics[key] === 0) {
      llamaMetrics[key] = value;
    }
  }

  const slotsInfo = [];
  for (const backend of backends) {
    for (let slotId = 0; slotId < backend.maxSlots; slotId++) {
      let activeReq = null;
      for (const [requestId, activeSlotId] of backend.activeSlots.entries()) {
        if (activeSlotId === slotId) {
          activeReq = requestId;
          break;
        }
      }

      const slotContext = backend.slotContextIndex.get(slotId);
      const slotActivity = getSlotActivitySnapshot(backend, slotId);
      slotsInfo.push({
        id: `${backend.id}:${slotId}`,
        backendId: backend.id,
        backendLabel: `${backend.host}:${backend.port}`,
        hash: slotContext?.exactHash || null,
        activeReq,
        taskId: slotActivity.taskId,
        phase: slotActivity.phase,
        summary: slotActivity.summary,
        detail: slotActivity.detail,
        updatedAt: slotActivity.updatedAt,
        startedAt: slotActivity.startedAt,
        progress: slotActivity.progress,
        restoredCacheHash: slotActivity.restoredCacheHash,
        cacheStatus: slotActivity.cacheStatus,
        partialScore: slotActivity.partialScore,
        cacheReason: slotActivity.cacheReason,
      });
    }
  }

  reply.send({
    queueLength: getTotalQueueLength(),
    l1UsageBytes,
    l1EvictThresholdBytes: L1_EVICT_THRESHOLD,
    l1EvictTargetBytes: L1_EVICT_TARGET,
    gpu: gpuMetrics,
    models: [...buildAggregatedModelRegistry().values()].map(entry => ({
      id: entry.id,
      available: entry.available,
      meta: entry.meta,
      loadedBackends: entry.loadedBackends.length,
      healthyBackends: entry.healthyBackends.length,
      backends: entry.backends,
    })),
    slots: slotsInfo,
    llamaMetrics,
    backends: backendSnapshots.map(({ backend, metrics }) => ({
      id: backend.id,
      url: backend.baseUrl,
      gpuGroup: backend.gpuGroup,
      healthy: backend.healthy,
      routable: isBackendRoutable(backend),
      degradedUntil: backend.degradedUntil,
      failureStreak: backend.failureStreak,
      lastFailureKind: backend.lastFailureKind,
      requestFailures: backend.requestFailures,
      restoreFailures: backend.restoreFailures,
      activity: getBackendActivitySnapshot(backend),
      queueLength: backend.slotQueue.length,
      activeRequests: backend.activeSlots.size,
      availableSlots: backend.availableSlots.length,
      lastCheckAt: backend.lastCheckAt,
      lastError: backend.lastError,
      modelsDiscoveryAt: backend.modelsDiscoveryAt,
      modelsDiscoveryError: backend.modelsDiscoveryError,
      models: getBackendDiscoveredModels(backend),
      process: getBackendProcessSnapshot(backend),
      metrics,
    })),
  });
});

server.get('/models', async (request, reply) => {
  const registry = buildAggregatedModelRegistry();
  reply.send({
    data: [...registry.values()].map(entry => ({
      id: entry.id,
      object: 'model',
      meta: entry.meta || null,
      status: {
        value: entry.loadedBackends.length > 0
          ? 'loaded'
          : (entry.available ? 'unloaded' : 'unavailable'),
      },
      backends: entry.backends,
    })),
  });
});

server.post('/models/load', async (request, reply) => {
  const modelId = normalizeModelId(request.body?.model);
  if (!modelId) {
    reply.status(400).send({
      error: {
        code: 400,
        message: 'model is required',
        type: 'invalid_request_error',
      },
    });
    return;
  }

  const candidateBackends = getBackendsForModel(modelId);
  if (candidateBackends.length === 0) {
    reply.status(404).send({
      error: {
        code: 404,
        message: `Unknown model: ${modelId}`,
        type: 'not_found_error',
      },
    });
    return;
  }

  const results = await Promise.allSettled(candidateBackends.map(backend => (
    withBackendModelOperationLock(backend, () => switchModelOnBackend(backend, modelId))
      .then(result => ({ backendId: backend.id, ...result }))
  )));

  const settledResults = results.map((result, index) => ({
    backendId: candidateBackends[index].id,
    result,
  }));

  const failed = settledResults
    .filter(entry => entry.result.status === 'rejected')
    .map(entry => ({
      backendId: entry.backendId,
      message: entry.result.reason?.message || String(entry.result.reason),
    }));

  if (failed.length > 0) {
    reply.status(502).send({
      error: {
        code: 502,
        message: `Failed to load model ${modelId} on ${failed.map(item => `${item.backendId}: ${item.message}`).join('; ')}`,
        type: 'backend_model_load_error',
      },
      failures: failed,
    });
    return;
  }

  const backendResults = settledResults.map(entry => entry.result.value);
  const pendingBackends = backendResults.filter(result => result.pending);

  reply.status(pendingBackends.length > 0 ? 202 : 200).send({
    success: pendingBackends.length === 0,
    pending: pendingBackends.length > 0,
    model: modelId,
    backends: backendResults,
  });
});

server.post('/models/unload', async (request, reply) => {
  const modelId = normalizeModelId(request.body?.model);
  if (!modelId) {
    reply.status(400).send({
      error: {
        code: 400,
        message: 'model is required',
        type: 'invalid_request_error',
      },
    });
    return;
  }

  const candidateBackends = getBackendsForModel(modelId);
  if (candidateBackends.length === 0) {
    reply.status(404).send({
      error: {
        code: 404,
        message: `Unknown model: ${modelId}`,
        type: 'not_found_error',
      },
    });
    return;
  }

  const results = await Promise.allSettled(candidateBackends.map(backend => (
    withBackendModelOperationLock(backend, () => unloadModelOnBackend(backend, modelId))
      .then(result => ({ backendId: backend.id, ...result }))
  )));

  const failed = results
    .filter(result => result.status === 'rejected')
    .map((result, index) => ({
      backendId: candidateBackends[index].id,
      message: result.reason?.message || String(result.reason),
    }));

  if (failed.length > 0) {
    reply.status(502).send({
      error: {
        code: 502,
        message: `Failed to unload model ${modelId} on ${failed.map(item => `${item.backendId}: ${item.message}`).join('; ')}`,
        type: 'backend_model_unload_error',
      },
      failures: failed,
    });
    return;
  }

  reply.send({
    success: true,
    model: modelId,
    backends: results.map(result => result.value),
  });
});

server.get('/v1/models', async (request, reply) => {
  const registry = buildAggregatedModelRegistry();
  reply.send({
    object: 'list',
    data: [...registry.values()].map(toOpenAiModel),
  });
});

server.get('/v1/models/:modelId', async (request, reply) => {
  const entry = getAggregatedModelEntry(request.params.modelId);
  if (!entry) {
    reply.status(404).send({
      error: {
        code: 404,
        message: `Unknown model: ${request.params.modelId}`,
        type: 'not_found_error',
      },
    });
    return;
  }

  reply.send(toOpenAiModel(entry));
});

// ─── Settings API ────────────────────────────────────────────────────

server.get('/api/settings', (request, reply) => {
  reply.send(getSettings());
});

server.post('/api/settings', async (request, reply) => {
  try {
    const newSettings = request.body;
    setSettings(newSettings);
    reply.send({ ok: true });
  } catch (err) {
    reply.status(500).send({ error: err.message });
  }
});

server.post('/api/settings/reload', async (request, reply) => {
  try {
    await loadSettings();
    reply.send({ ok: true });
  } catch (err) {
    reply.status(500).send({ error: err.message });
  }
});

// ─── Restart API ─────────────────────────────────────────────────────

server.post('/api/restart', async (request, reply) => {
  try {
    log('SERVER', 'Restart requested via Control Deck. Signaling graceful teardown...');
    reply.send({ ok: true });
    
    // Instead of process.exit(0), trigger self-directed SIGTERM 
    // to let your existing process event listeners kill worker nodes.
    setImmediate(() => process.emit('SIGTERM'));
  } catch (err) {
    reply.status(500).send({ error: err.message });
  }
});


// ─── Backend Logs ────────────────────────────────────────────────────

server.get('/api/backends/:backendId/logs', async (request, reply) => {
  const backend = backendsById.get(request.params.backendId);
  if (!backend) {
    reply.status(404).send({ error: 'backend_not_found' });
    return;
  }

  const requestedLines = Number.parseInt(request.query?.lines, 10);
  const lineCount = Number.isFinite(requestedLines) ? Math.min(Math.max(requestedLines, 20), 500) : 120;
  const content = await tailFile(backend.processInfo.logPath, lineCount);
  reply.send({
    backendId: backend.id,
    logPath: backend.processInfo.logPath,
    lines: lineCount,
    content,
  });
});

server.post('/api/backends/:backendId/:action', async (request, reply) => {
  const backend = backendsById.get(request.params.backendId);
  if (!backend) {
    reply.status(404).send({ error: 'backend_not_found' });
    return;
  }

  const { action } = request.params;
  if (!['start', 'stop', 'restart'].includes(action)) {
    reply.status(400).send({ error: 'invalid_action' });
    return;
  }

  if (action === 'start') {
    await startBackendProcess(backend, 'dashboard');
  } else if (action === 'stop') {
    await stopBackendProcess(backend, 'dashboard');
  } else {
    await restartBackendProcess(backend, 'dashboard');
  }

  await checkBackendHealth(backend);
  reply.send({
    backendId: backend.id,
    action,
    healthy: backend.healthy,
    process: getBackendProcessSnapshot(backend),
  });
});

server.post('/v1/chat/completions', async (request, reply) => {
  const body = request.body || {};
  const reqId = (body?.metadata && body.metadata?.request_id)
    || body?.id
    || request.id;

  const requestContext = extractRequestContext(body);
  logRequestToFile(reqId, requestContext.systemPrompt || requestContext.promptText, requestContext.firstUserMessage);

  // ─── AbortController + disconnect handler (Fix #1: Zombie Slots) ─
  const abortController = new AbortController();
  const { signal } = abortController;

  let slotReleased = false;
  let selectedBackend = null;
  let assignedSlotId = null;
  let observedSlotId = null;
  let resolveInFlight = null;
  let cacheMarkedActive = false;
  const requestStartedAt = Date.now();
  let cacheStatus = 'BYPASS';
  let shouldSaveCache = false;
  let skipCacheSaveReason = null;
  let streamedChunkCount = 0;
  let metricsCaptureStarted = false;

  const updateActivity = patch => {
    if (!selectedBackend || assignedSlotId === null) {
      return;
    }

    const slotId = normalizeSlotId(observedSlotId, assignedSlotId, selectedBackend);
    updateSlotActivity(selectedBackend, slotId, {
      requestId: reqId,
      startedAt: requestStartedAt,
      cacheHash: requestContext.exactHash,
      ...patch,
    });
  };

  const safeReleaseSlot = () => {
    if (!slotReleased && selectedBackend && assignedSlotId !== null) {
      slotReleased = true;
      releaseSlotOnBackend(selectedBackend, reqId);
    }

    if (resolveInFlight) {
      const releaseInFlight = resolveInFlight;
      resolveInFlight = null;
      releaseInFlight();
    }
  };

  const markCacheBusy = () => {
    if (!cacheMarkedActive && requestContext.exactHash) {
      markCacheActive(requestContext.exactHash);
      cacheMarkedActive = true;
    }
  };

  const handleClientDisconnect = () => {
    if (!signal.aborted) {
      log('DISCONNECT', `Client dropped  ${reqId}`);
      abortController.abort();
    }
    safeReleaseSlot();
  };

  request.raw.on('aborted', () => {
    if (!reply.raw.writableEnded) {
      handleClientDisconnect();
    }
  });

  reply.raw.on('close', () => {
    if (!reply.raw.writableEnded && !signal.aborted) {
      handleClientDisconnect();
    }
  });

  reply.raw.on('error', () => {
    if (!signal.aborted) {
      handleClientDisconnect();
    }
  });

  const filename = requestContext.exactHash ? `${requestContext.exactHash}.bin` : null;
  const l1Path = filename ? join(L1_DIR, filename) : null;
  const l2Path = filename ? join(L2_DIR, filename) : null;

  if (requestContext.canUseCache && requestContext.exactHash && inFlightMisses.has(requestContext.exactHash)) {
    log('DEDUP', `Waiting for in-flight cache creation for ${requestContext.exactHash.slice(0, 8)}`);
    try {
      await new Promise((resolve, reject) => {
        // 1. Declare and define the resolver first so the reference is completely solid
        const resolver = () => {
          signal.removeEventListener('abort', onAbort);
          resolve();
        };

        // 2. Define the abort loop which can now confidently target the pointer
        const onAbort = () => {
          const list = inFlightMisses.get(requestContext.exactHash);
          if (list) {
            const idx = list.indexOf(resolver); 
            if (idx !== -1) {
              list.splice(idx, 1);
            }
            if (list.length === 0) {
              inFlightMisses.delete(requestContext.exactHash);
            }
          }
          const err = new Error('aborted');
          err.name = 'AbortError';
          reject(err);
        };

        if (signal.aborted) return onAbort();
        signal.addEventListener('abort', onAbort);
        inFlightMisses.get(requestContext.exactHash).push(resolver);
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      throw err;
    }
  }

  const l1Exists = Boolean(filename && existsSync(l1Path));
  const l2Exists = Boolean(filename && existsSync(l2Path));
  const effectiveBody = requestContext.model && requestContext.model !== body?.model
    ? { ...body, model: requestContext.model }
    : body;
  const routing = chooseBackendForRequest(requestContext);
  selectedBackend = routing.backend;

  log(
    'ROUTE',
    `backend=${selectedBackend.id} reason=${routing.reason} req=${reqId}`
      + (requestContext.subagentRequest
        ? ` requestedModel=${requestContext.requestedModel || 'none'} effectiveModel=${requestContext.model}`
        : '')
  );

  try {
    assignedSlotId = await acquireSlotOnBackend(selectedBackend, reqId, signal, routing.preferredSlotId);
    observedSlotId = assignedSlotId;
    const trackedSlot = selectedBackend.slotContextIndex.get(assignedSlotId);
    // Revoke the previous residency claim before this request can mutate the slot.
    // We keep a local snapshot only to validate whether this request can treat the slot
    // as an exact hot hit; the authoritative index is restored only after clean success.
    clearSlotContext(selectedBackend, assignedSlotId);
    updateActivity({
      phase: 'prefill',
      summary: 'Prefilling prompt',
      detail: 'Submitting prompt to backend.',
      progress: 0,
      cacheStatus,
    });

    let proxyBody = { ...effectiveBody, id_slot: assignedSlotId };

    if (requestContext.canUseCache && (l1Exists || l2Exists)) {
      markCacheBusy();
      let cacheReady = trackedSlot?.exactHash === requestContext.exactHash;

      if (!cacheReady) {
        try {
          cacheStatus = l1Exists ? 'HIT' : 'COLD_HIT';
          updateActivity({
            phase: 'restoring',
            summary: l1Exists ? 'Restoring cache' : 'Restoring cold cache',
            detail: l1Exists
              ? 'Loading cached prefix from L1.'
              : 'Copying cached prefix from L2 and restoring it.',
            restoredCacheHash: requestContext.exactHash,
            cacheStatus,
            progress: null,
          });
          if (!l1Exists && l2Exists) {
            logColdHit(requestContext.exactHash);
            await copyFileL2ToL1(requestContext.exactHash);
          } else {
            logHit(requestContext.exactHash);
          }

          await restoreKVCache(requestContext.exactHash, requestContext.model, selectedBackend, assignedSlotId);
          cacheReady = true;
        } catch (err) {
          cacheStatus = 'BYPASS';
          updateActivity({
            phase: 'prefill',
            summary: 'Prefilling prompt',
            detail: 'Cache restore failed, continuing without cache.',
            restoredCacheHash: null,
            cacheStatus,
            progress: 0,
          });
          log('WARN', `Restore failed (${err.message}) — falling back to full request`);
          
          // Clean up potentially corrupted cache files so they aren't tried again
          if (l1Path) rm(l1Path, { force: true }).catch(() => {});
          if (l2Path) rm(l2Path, { force: true }).catch(() => {});
        }
      }

      if (cacheReady) {
        proxyBody = { ...effectiveBody, messages: requestContext.userMessages, id_slot: assignedSlotId };
        
        // RE-PRIME INDEX: Since we verified this slot actively holds valid warm cache data,
        // restore its lookups immediately so concurrent in-flight requests can reuse it.
        rememberSlotContext(selectedBackend, assignedSlotId, requestContext, 'hit');

        updateActivity({
          phase: 'prefill',
          summary: 'Prefilling prompt',
          detail: 'Cached prefix restored. Replaying request delta.',
          restoredCacheHash: requestContext.exactHash,
          cacheStatus,
          progress: null,
        });
      }
    } else if (requestContext.canUseCache && !l1Exists && !l2Exists) {
      markCacheBusy();
      logMiss(requestContext.exactHash);
      inFlightMisses.set(requestContext.exactHash, []);
      resolveInFlight = () => {
        const waiters = inFlightMisses.get(requestContext.exactHash) || [];
        inFlightMisses.delete(requestContext.exactHash);
        waiters.forEach(waiter => waiter());
      };
      cacheStatus = 'MISS';
      shouldSaveCache = true;
      updateActivity({
        phase: 'prefill',
        summary: 'Prefilling prompt',
        detail: 'Cache miss. Evaluating full prompt.',
        cacheStatus,
        progress: 0,
      });
    }

    const observeSlot = createSlotTracker(slotId => {
      observedSlotId = slotId;
    });
    const observeStreamChunk = chunk => {
      observeSlot(chunk, parsed => {
        recordObservedChatMetrics(parsed);
        if (shouldSaveCache && !skipCacheSaveReason && payloadHasLengthFinishReason(parsed)) {
          skipCacheSaveReason = 'finish_reason_length';
          shouldSaveCache = false;
          markCacheSaveSkipped(skipCacheSaveReason, requestContext.exactHash);
        }
      });
      if (!metricsCaptureStarted && selectedBackend) {
        metricsCaptureStarted = true;
        captureLiveBackendMetrics(selectedBackend).catch(err => {
          logWarn(`Deferred metrics capture failed for ${selectedBackend.id}: ${err.message}`);
        });
      }
      streamedChunkCount += 1;
      updateActivity({
        phase: 'streaming',
        summary: 'Generating response',
        detail: 'Streaming tokens to client.',
        cacheStatus,
        progress: null,
        tokens: streamedChunkCount,
      });
    };

    const llamaRes = await fetchChatCompletion(selectedBackend, request.headers, proxyBody, signal);
    if (!llamaRes.ok) {
      const errText = await llamaRes.text().catch(() => '');
      let parsedError = null;
      try {
        parsedError = errText ? JSON.parse(errText) : null;
      } catch {
        parsedError = null;
      }

      const classification = classifyContextOverflow(parsedError || errText, llamaRes.status);
      if (classification.overflow) {
        const skippedSave = shouldSaveCache;
        skipCacheSaveReason = 'context_overflow';
        shouldSaveCache = false;
        markContextOverflow(
          requestContext.exactHash,
          `http_${llamaRes.status}`,
          classification.message || errText.slice(0, 160),
          skippedSave
        );

        const err = new Error(classification.message || `Request exceeds the available context size on backend ${selectedBackend.id}`);
        err.statusCode = 400;
        err.errorType = 'invalid_request_error';
        throw err;
      }

      logErr(`llama.cpp ${selectedBackend.id} returned ${llamaRes.status}: ${errText.slice(0, 500)}`);
      throw new Error(`llama.cpp ${selectedBackend.id} returned ${llamaRes.status}: ${errText.slice(0, 300)}`);
    }
    updateActivity({
      phase: 'awaiting-output',
      summary: 'Waiting for first token',
      detail: 'Backend accepted request.',
      cacheStatus,
      progress: null,
      tokens: streamedChunkCount,
    });

    const responseHeaders = buildProxyResponseHeaders(llamaRes.headers, {
      'X-Accel-Buffering': 'no',
      'X-Backend-Id': selectedBackend.id,
      'X-Backend-Url': selectedBackend.baseUrl,
    });
    if (cacheStatus !== 'BYPASS') {
      responseHeaders['X-Cache-Status'] = cacheStatus;
    }

    const responseContentType = llamaRes.headers.get('content-type') || '';
    if (responseContentType.includes('application/json')) {
      const responseText = await llamaRes.text();
      try {
        const parsed = JSON.parse(responseText);
        recordObservedChatMetrics(parsed);
        if (shouldSaveCache && !skipCacheSaveReason && payloadHasLengthFinishReason(parsed)) {
          skipCacheSaveReason = 'finish_reason_length';
          shouldSaveCache = false;
          markCacheSaveSkipped(skipCacheSaveReason, requestContext.exactHash);
        }
        const slotId = parsed?.slot_id ?? parsed?.id_slot;
        if (Number.isInteger(slotId)) {
          observedSlotId = slotId;
        }
      } catch {
        // Ignore invalid JSON payloads and pass them through unchanged.
      }

      reply.headers(responseHeaders).status(llamaRes.status).send(responseText);
    } else {
      await proxyStreamWithHeaders(llamaRes, reply, responseHeaders, signal, observeStreamChunk);
    }

    const effectiveSlotId = normalizeSlotId(observedSlotId, assignedSlotId, selectedBackend);
    if (requestContext.hasAffinity) {
      // Re-advertise the slot only after the request completed without error. Doing this
      // earlier would let a later overflow or abort leave behind a false hot-slot claim.
      rememberSlotContext(selectedBackend, effectiveSlotId, requestContext, cacheStatus === 'HIT' ? 'hit' : 'stream');
    }

    if (shouldSaveCache && !signal.aborted) {
      try {
        updateActivity({
          phase: 'saving-cache',
          summary: 'Saving cache',
          detail: 'Persisting fresh KV cache for reuse.',
          cacheStatus,
          progress: null,
          tokens: streamedChunkCount,
        });

        // Proactive Defrag Cushions: Run an immediate check before saving if VRAM/L1 limits are close
        await sweepL1Eviction();

        // RACE OPTIMIZATION: If a copy of this exact hash is already in flight, drop out safely
        if (inFlightDiskCopies.has(requestContext.exactHash)) {
          log('SAVE', `Cache replication in-flight; concurrent L1/L2 save dropped safely`);
        } else if (existsSync(l1Path) && existsSync(l2Path)) {
          log('SAVE', `Cache already exists on L1 and L2; skip approved`);
        } else {
          const saveResult = await saveKVCache(requestContext.exactHash, requestContext.model, selectedBackend, assignedSlotId);
          if (saveResult) {
            rememberSlotContext(selectedBackend, effectiveSlotId, requestContext, 'save');

            // FATIGUE OPTIMIZATION: If the target file is already warm on L2, do not burn drive lifespan re-writing it
            if (existsSync(l2Path)) {
              log('SAVE', `Cache preserved on L1; L2 SSD skip approved (Invariance Validated)`);
            } else {
              // RACE OPTIMIZATION: Handle concurrent block copying requests for identical hashes seamlessly
              if (!inFlightDiskCopies.has(requestContext.exactHash)) {
                const copyWithTimeout = (src, dst, timeoutMs = 45_000) => {
                  const controller = new AbortController();
                  const timer = setTimeout(() => controller.abort(), timeoutMs);
                  return copyFile(src, dst).finally(() => clearTimeout(timer));
                };

                const copyPromise = copyWithTimeout(l1Path, l2Path)
                  .then(() => {
                    log('SAVE', `Successfully Persisted to L2 SSD Layer`);
                  })
                  .catch(err => logWarn(`Background replication aborted: ${err.message}`))
                  .finally(() => {
                    inFlightDiskCopies.delete(requestContext.exactHash);
                  });

                inFlightDiskCopies.set(requestContext.exactHash, copyPromise);
              }
            }
          }
        }
      } catch (err) {
        logWarn(`Save tracking pipe failed: ${err.message}`);
      }
    } else if (skipCacheSaveReason && cacheStatus === 'MISS') {
      updateActivity({
        phase: 'idle',
        summary: 'Cache save skipped',
        detail: `Skipped KV persistence: ${skipCacheSaveReason}`,
        cacheStatus,
        progress: null,
        tokens: streamedChunkCount,
      });
    }
    if (!signal.aborted && selectedBackend) {
      markBackendRequestSuccess(selectedBackend);
    }
  } catch (err) {
    if (selectedBackend && err.name !== 'AbortError' && !String(err.message || '').includes('aborted')) {
      markBackendRequestFailure(selectedBackend, 'request', err);
      const slotId = normalizeSlotId(observedSlotId, assignedSlotId, selectedBackend);
      updateSlotActivity(selectedBackend, slotId, {
        phase: 'error',
        summary: 'Request failed',
        detail: err.message,
        progress: null,
        cacheStatus,
      });
    }
    throw err;
  } finally {
    if (selectedBackend && assignedSlotId !== null) {
      const slotId = normalizeSlotId(observedSlotId, assignedSlotId, selectedBackend);
      updateSlotActivity(selectedBackend, slotId, signal.aborted
        ? {
            phase: 'canceled',
            summary: 'Canceled',
            detail: 'Client disconnected before completion.',
            startedAt: null,
            cacheStatus,
            progress: null,
            tokens: streamedChunkCount || null,
          }
        : {
            phase: 'idle',
            summary: 'Idle',
            detail: streamedChunkCount > 0 ? 'Last request completed.' : 'No recent backend activity.',
            startedAt: null,
            restoredCacheHash: cacheStatus === 'HIT' || cacheStatus === 'COLD_HIT'
              ? requestContext.exactHash
              : null,
            cacheStatus: cacheStatus === 'BYPASS' ? null : cacheStatus,
            progress: null,
            tokens: streamedChunkCount || null,
          });
    }
    if (cacheMarkedActive && requestContext.exactHash) {
      markCacheInactive(requestContext.exactHash);
    }
    safeReleaseSlot();
  }

  return;
});

// ─── 8. Error Handler ──────────────────────────────────────────────

server.setErrorHandler((error, request, reply) => {
  // IGNORE ABORT ERRORS
  if (error.name === 'AbortError' || error.message.includes('aborted')) {
    return; // Client disconnected, nothing to do.
  }

  logErr(`${error.message}  [${request.id || 'no-id'}]`);
  if (!reply.sent) {
    const statusCode = Number.isInteger(error.statusCode) ? error.statusCode : 500;
    reply.status(statusCode).send({
      error: {
        code: statusCode,
        message: error.message,
        type: error.errorType || 'internal_error',
      },
    });
  }
});

// ─── Graceful Shutdown ──────────────────────────────────────────────

process.on('SIGTERM', async () => {
  log('SHUTDOWN', 'SIGTERM received, shutting down...');

  // Kill all backend child processes
  for (const backend of backends) {
    if (backend.processInfo.pid) {
      try { process.kill(-backend.processInfo.pid, 'SIGTERM'); } catch {}
    }
  }

  // Close Fastify server
  await server.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  log('SHUTDOWN', 'SIGINT received');
  process.exit(130);
});

const start = async () => {
  // Load settings first
  await initSettings();

  // Create backend registry from loaded settings
  backends = BACKEND_CONFIGS.map(createBackendState);
  backendsById = new Map(backends.map(backend => [backend.id, backend]));

  // Ensure both directories exist
  if (!existsSync(L1_DIR)) {
    mkdir(L1_DIR, { recursive: true });
    log('INIT', `Created L1 (RAMDisk): ${L1_DIR}`);
  }
  if (!existsSync(L2_DIR)) {
    mkdir(L2_DIR, { recursive: true });
    log('INIT', `Created L2 (SSD): ${L2_DIR}`);
  }

  // Pre-warm L1 from L2
  await warmupL1();

  // Prime backend health state before accepting traffic.
  await checkAllBackendsHealth();
  await bootstrapBackendProcesses();

  // Start GC sweeper (L1 eviction + age-based cleanup)
  const gcInterval = setInterval(async () => {
    await sweepL1Eviction();
    await gcSweep();
  }, GARBAGE_COLLECT_INTERVAL_MS);
  if (gcInterval.unref) gcInterval.unref();

  const healthInterval = setInterval(() => {
    checkAllBackendsHealth().catch(err => {
      logWarn(`Health sweep failed: ${err.message}`);
    });
  }, BACKEND_HEALTHCHECK_INTERVAL_MS);
  if (healthInterval.unref) healthInterval.unref();

  try {
    await server.listen({ port: 8080, host: '0.0.0.0' });

    const PORT = 8080;
    const stripAnsi = (str) => str.replace(/\x1b\[[0-9;]*m/g, '');
    const backendSummary = backends
      .map(backend => `${backend.id}=${backend.baseUrl} gpu:${backend.gpuGroup} slots:${backend.maxSlots}`)
      .join(' | ');
    const lines = [
      `  ${C.bold}${C.green}KV Cache Semantic Router — Tiered Storage${C.reset}`,
      `  Listening on  ${C.bold}http://0.0.0.0:${PORT}${C.reset}`,
      `  Dashboard UI  ${C.bold}${C.magenta}http://127.0.0.1:${PORT}/dashboard${C.reset}`,
      `  Backends      ${C.bold}${backendSummary}${C.reset}`,
      `  L1 (RAMDisk)  ${C.bold}${L1_DIR}${C.reset} (tmpfs)`,
      `  L2 (SSD)      ${C.bold}${L2_DIR}${C.reset} (persistent)`,
      `  Health check  ${C.bold}${BACKEND_HEALTHCHECK_INTERVAL_MS / 1000}s${C.reset} timeout ${C.bold}${BACKEND_HEALTHCHECK_TIMEOUT_MS}ms${C.reset}`,
      `  Evict >       ${C.bold}${(L1_EVICT_THRESHOLD / 1024 / 1024 / 1024).toFixed(1)}GB${C.reset} down to ${C.bold}${(L1_EVICT_TARGET / 1024 / 1024 / 1024).toFixed(1)}GB${C.reset}`,
      `  GC interval   ${C.bold}${GARBAGE_COLLECT_INTERVAL_MS / 60000}min${C.reset} maxAge ${C.bold}${MAX_FILE_AGE_MS / 3600000}h${C.reset}`,
      `  RAG threshold ${C.bold}${RAG_CONTENT_LENGTH_THRESHOLD} chars${C.reset}`
    ];

    const boxWidth = 68;
    console.log(`\n${C.bold}${C.cyan}╔${'═'.repeat(boxWidth)}╗${C.reset}`);
    for (const line of lines) {
      const visibleLen = stripAnsi(line).length;
      const padding = Math.max(0, boxWidth - visibleLen);
      console.log(`${C.bold}${C.cyan}║${C.reset}${line}${' '.repeat(padding)}${C.bold}${C.cyan}║${C.reset}`);
    }
    console.log(`${C.bold}${C.cyan}╚${'═'.repeat(boxWidth)}╝${C.reset}\n`);
  } catch (err) {
    logErr(`Cannot start server: ${err.message}`);
    process.exit(1);
  }
};

start();
