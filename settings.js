#!/usr/bin/env node
'use strict';

const { readFile, writeFile, mkdir, rename } = require('fs/promises');
const { existsSync } = require('fs');
const { join, dirname } = require('path');
const { createHash } = require('crypto');
const os = require('os');
const { homedir } = os;

// ─── Schema defaults ─────────────────────────────────────────────────

function defaultSettings() {
  return {
    server: {
      port: 8080,
      host: '0.0.0.0',
    },
    storage: {
      modelsDir: join(homedir(), 'Models'),
      l1Dir: join(homedir(), 'Models', 'ramdisk_cache'),
      l2Dir: join(homedir(), 'Models', 'slot_cache'),
      backendLogDir: join(__dirname, 'runtime-logs'),
    },
    backends: [
      {
        id: 'llama-a',
        host: '127.0.0.1',
        port: 11434,
        gpuGroup: '0,1',
        maxSlots: 1,
        scriptPath: join(__dirname, 'config', 'start_server.example.sh'),
        modelName: 'qwen',
      },
    ],
    cache: {
      l1EvictThresholdBytes: 2_684_354_560,
      l1EvictTargetBytes: 2_147_483_648,
      partialCacheMinScore: 0.3,
      partialCacheSystemBonus: 0.12,
      partialCacheSuccessBonus: 0.08,
      partialCacheFailurePenalty: 0.18,
    },
    health: {
      backendHealthcheckIntervalMs: 5_000,
      backendHealthcheckTimeoutMs: 1_500,
      dedupWaitTimeoutMs: 15_000,
      backendStartupGraceMs: 12_000,
      backendAutoStart: true,
    },
    gc: {
      gcIntervalMs: 5 * 60 * 1000,
      maxFileAgeMs: 12 * 60 * 60 * 1000,
    },
    misc: {
      warmupTopN: 15,
      ragContentLengthThreshold: 1000,
      hashChunkSize: 64 * 1024,
      hashYieldInterval: 8,
      largeInputCooperativeThreshold: 256 * 1024,
      subagentModel: 'fast',
      subagentPromptPatterns: [
        'managed memory extraction subagent',
        'you are now acting as',
        'specialist agent',
        'read-only exploration task',
      ],
    },
  };
}

const SETTINGS_PATH = join(__dirname, 'config', 'config.json');
let _settings = defaultSettings();
let _dirty = false;
let _saveTimer = null;
let _backends = [];
let _gcInterval = null;
let _healthInterval = null;

// ─── Load / Save ─────────────────────────────────────────────────────

async function loadSettings() {
  try {
    const content = await readFile(SETTINGS_PATH, 'utf8');
    const parsed = JSON.parse(content);
    _settings = {
      ...defaultSettings(),
      ...parsed,
      storage: { ...defaultSettings().storage, ...(parsed.storage || {}) },
      backends: Array.isArray(parsed.backends) ? parsed.backends : defaultSettings().backends,
      cache: { ...defaultSettings().cache, ...(parsed.cache || {}) },
      health: { ...defaultSettings().health, ...(parsed.health || {}) },
      gc: { ...defaultSettings().gc, ...(parsed.gc || {}) },
      misc: { ...defaultSettings().misc, ...(parsed.misc || {}) },
    };
    return _settings;
  } catch (err) {
    if (err.code !== 'ENOENT') {
      console.error(`Failed to load settings: ${err.message}`);
    }
    return _settings;
  }
}

function scheduleSave() {
  _dirty = true;
  if (_saveTimer) {
    clearTimeout(_saveTimer);
  }
  _saveTimer = setTimeout(() => {
    _saveTimer = null;
    saveSettings().catch(err => {
      console.error(`Settings save failed: ${err.message}`);
    });
  }, 500);
  if (_saveTimer.unref) _saveTimer.unref();
}

async function saveSettings() {
  if (!_dirty) return;
  _dirty = false;
  const tmpPath = `${SETTINGS_PATH}.tmp`;
  const content = JSON.stringify(_settings, null, 2);
  await writeFile(tmpPath, content, 'utf8');
  await rename(tmpPath, SETTINGS_PATH);
}

// ─── Accessors ───────────────────────────────────────────────────────

function getSettings() {
  return _settings;
}

function setSettings(newSettings) {
  _settings = newSettings;
  scheduleSave();
}

function setBackend(id, config) {
  const idx = _settings.backends.findIndex(b => b.id === id);
  if (idx < 0) {
    _settings.backends.push(config);
  } else {
    _settings.backends[idx] = { ..._settings.backends[idx], ...config };
  }
  scheduleSave();
}

function removeBackend(id) {
  _settings.backends = _settings.backends.filter(b => b.id !== id);
  scheduleSave();
}

function setCache(key, value) {
  _settings.cache[key] = value;
  scheduleSave();
}

function setHealth(key, value) {
  _settings.health[key] = value;
  scheduleSave();
}

function setGc(key, value) {
  _settings.gc[key] = value;
  scheduleSave();
}

function setMisc(key, value) {
  _settings.misc[key] = value;
  scheduleSave();
}

// ─── Apply to running server ─────────────────────────────────────────

function applySettings(server, backendsById) {
  const s = _settings;

  // Update storage directories
  const { storage } = s;
  if (!existsSync(storage.l1Dir)) {
    mkdir(storage.l1Dir, { recursive: true });
  }
  if (!existsSync(storage.l2Dir)) {
    mkdir(storage.l2Dir, { recursive: true });
  }

  // Update backends
  for (const backend of s.backends) {
    const existing = backendsById.get(backend.id);
    if (existing) {
      existing.host = backend.host;
      existing.port = backend.port;
      existing.baseUrl = `http://${backend.host}:${backend.port}`;
      existing.gpuGroup = backend.gpuGroup;
      existing.maxSlots = backend.maxSlots;
      existing.scriptPath = backend.scriptPath;
      existing.modelName = backend.modelName;
    }
  }

  return s;
}

// ─── Export ──────────────────────────────────────────────────────────

module.exports = {
  loadSettings,
  saveSettings,
  getSettings,
  setSettings,
  setBackend,
  removeBackend,
  setCache,
  setHealth,
  setGc,
  setMisc,
  applySettings,
  defaultSettings,
};
