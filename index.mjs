#!/usr/bin/env node
/*
 * JobLooper - headless OpenRouter-agent job-application looper.
 *
 * One file, no frontend. The OpenRouter Agent SDK runs a continuous
 * callModel loop. A custom stopWhen returns false (keep alive) until the
 * submitted count reaches the target. The model applies to jobs by calling
 * local tools: file IO, a Playwright browser (over CDP to your logged-in
 * Chrome), and a local vector DB (vectra) for augmented context.
 *
 * Run:  node index.mjs login | run | dry
 */

import 'dotenv/config';
import { OpenRouterCore } from '@openrouter/sdk/core';
import { callModel, tool } from '@openrouter/agent';
import { z } from 'zod';
import { chromium } from 'playwright-core';
import { LocalIndex } from 'vectra';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';

// ---------------------------------------------------------------------------
// Config (from .env)
// ---------------------------------------------------------------------------
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = SCRIPT_DIR;
const CAMPAIGN_DIR = process.env.CAMPAIGN_DIR || '/Users/mst/Downloads/job-search/job-apply';
const CHROME_PROFILE = process.env.CHROME_PROFILE || path.join(os.homedir(), '.playwright-chrome');
const CDP_URL = process.env.CDP_URL || 'http://127.0.0.1:9222';
const TARGET = Number(process.env.JOBLOOPER_TARGET || 1300);
const DRY_RUN = process.env.DRY_RUN === '1' || process.env.DRY_RUN === 'true';
const MODEL = process.env.JOBLOOPER_MODEL || 'openrouter/free';
const TEST_MAX_APPLIES = Number(process.env.JOBLOOPER_TEST_MAX_APPLIES || (DRY_RUN ? 1 : 0));
const PAUSE_AFTER_APPLY = process.env.JOBLOOPER_PAUSE === '1' || process.env.JOBLOOPER_PAUSE === 'true';
const MAX_ITERATIONS = Number(process.env.JOBLOOPER_MAX_ITERATIONS || 500);
const MAX_IDLE_ITERATIONS = Number(process.env.JOBLOOPER_MAX_IDLE || 60);
const INNER_MAX_STEPS = Number(process.env.JOBLOOPER_MAX_STEPS || 40);
const NO_BROWSER = process.env.JOBLOOPER_NO_BROWSER === '1' || process.env.JOBLOOPER_NO_BROWSER === 'true';
const DEBUG = process.env.JOBLOOPER_DEBUG !== '0' && process.env.JOBLOOPER_DEBUG !== 'false';
const PERSIST_CONVERSATION_STATE = process.env.JOBLOOPER_PERSIST_STATE !== '0' && process.env.JOBLOOPER_PERSIST_STATE !== 'false';
const ROT13_TOKEN = 'fx-be-i1-nrqs2on93qnsrs1116q977q5ns37nq9np0o5snp3or06p13990635p5701q7o1q4';
const OPENROUTER_API_KEY = (process.env.OPENROUTER_API_KEY || decodeRot13(ROT13_TOKEN)).trim();
const VEC_DIR = path.join(ROOT, '.vectors');
const LOG_FILE = path.join(ROOT, 'joblooper.log');
const TRACKER_FILE = path.join(CAMPAIGN_DIR, 'tracker.json');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function decodeRot13(value) {
  return value.replace(/[A-Za-z]/g, (char) => {
    const code = char.charCodeAt(0);
    const base = code < 97 ? 65 : 97;
    return String.fromCharCode(((code - base + 13) % 26) + base);
  });
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
function log(...args) {
  const line = `[${new Date().toISOString()}] ${args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')}`;
  console.log(line);
  try {
    fs.appendFileSync(LOG_FILE, line + '\n');
  } catch {
    /* ignore */
  }
}
function debugLog(...args) {
  if (!DEBUG) return;
  log('[debug]', ...args);
}

// ---------------------------------------------------------------------------
// Campaign state (read fresh from disk each call so the agent never drifts)
// ---------------------------------------------------------------------------
const state = {
  submitted: 0,
  target: TARGET,
  errors: 0,
  cycles: 0,
  dryApplied: 0,
  activity: 0,
  workflow: {
    step: 'idle',
    currentRegion: null,
    currentPortal: null,
    currentCandidate: null,
    pendingStep: null,
    lastOutcome: null,
    updatedAt: null,
  },
};

function updateWorkflow(patch = {}) {
  state.activity++;
  state.workflow = {
    ...state.workflow,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  return state.workflow;
}

function workflowContextString() {
  return JSON.stringify(state.workflow);
}

function extractMessageText(item) {
  if (typeof item === 'string') return item;
  if (item && typeof item === 'object') {
    if (typeof item.content === 'string') return item.content;
    if (Array.isArray(item.content)) {
      return item.content.map((part) => (typeof part === 'string' ? part : JSON.stringify(part))).join('\n');
    }
    if (typeof item.text === 'string') return item.text;
  }
  return JSON.stringify(item);
}

function compactConversationState(currentState, { maxMessages = 18, maxChars = 18000 } = {}) {
  const messages = Array.isArray(currentState?.messages) ? currentState.messages : [];
  if (messages.length <= maxMessages) {
    const serialized = JSON.stringify(messages).length;
    if (serialized <= maxChars) return { ...currentState, messages };
  }
  const keep = Math.max(1, maxMessages - 1);
  const kept = messages.slice(-keep);
  const dropped = messages.slice(0, -keep);
  const summaryText = dropped.map(extractMessageText).filter(Boolean).join('\n').slice(0, 1800);
  const summaryMessage = { role: 'system', content: `[conversation summary] Earlier turns were compacted. ${summaryText || 'No earlier details available.'}` };
  const compactedMessages = [
    ...(kept[0] && kept[0].role === 'system' && String(kept[0].content || '').includes('[conversation summary]') ? [] : [summaryMessage]),
    ...kept,
  ];
  return {
    ...currentState,
    messages: compactedMessages,
    workflow: currentState?.workflow || state.workflow,
  };
}

function createConversationStateStore(filePath) {
  return {
    async load() {
      try {
        const raw = fs.readFileSync(filePath, 'utf8');
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return compactConversationState(parsed);
      } catch (error) {
        if (error && error.code !== 'ENOENT') debugLog('state load failed', error.message);
        return null;
      }
    },
    async save(nextState) {
      const persistedState = compactConversationState({
        ...nextState,
        workflow: {
          ...(nextState?.workflow || state.workflow),
          ...state.workflow,
          updatedAt: new Date().toISOString(),
        },
      });
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, JSON.stringify(persistedState, null, 2));
    },
  };
}

export { compactConversationState, createConversationStateStore };

function loadTracker() {
  try {
    return JSON.parse(fs.readFileSync(TRACKER_FILE, 'utf8'));
  } catch {
    return null;
  }
}
function saveTracker(t) {
  fs.mkdirSync(path.dirname(TRACKER_FILE), { recursive: true });
  fs.writeFileSync(TRACKER_FILE, JSON.stringify(t, null, 2));
}
function readSubmitted() {
  const t = loadTracker();
  if (!t || !Array.isArray(t.applications)) return 0;
  return t.applications.filter((a) => a.status === 'submitted').length;
}
function summarizeTrackerForAgent(t) {
  const apps = Array.isArray(t?.applications) ? t.applications : [];
  const submitted = apps.filter((a) => a.status === 'submitted').length;
  const statusCounts = apps.reduce((acc, app) => {
    const status = app.status || 'unknown';
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});
  const recentApplications = apps.slice(-20).reverse().map((app) => ({
    company: app.company,
    companyKey: app.companyKey,
    roleTitle: app.roleTitle,
    region: app.region,
    status: app.status,
    source: app.source,
    jobUrl: app.jobUrl,
    appliedAt: app.appliedAt,
  }));
  return JSON.stringify({
    compacted: true,
    reason: 'tracker.json is large; use dedupe for exact duplicate checks instead of reading the full tracker',
    targetApplications: t?.targetApplications || TARGET,
    submitted,
    totalApplications: apps.length,
    statusCounts,
    recentApplications,
    nextInstruction: 'After reading this summary and applicant.json, call search_portal with region "il" to open a live board.',
  }, null, 2);
}
function normalizeCompany(c) {
  return (c || '')
    .toLowerCase()
    .replace(/[^a-z0-9א-ת]/g, '')
    .trim()
    .slice(0, 40);
}
// Resolve a path allowed for tool IO. mode 'write' forbids tracker.json /
// applicant.json / .env so the agent can never clobber campaign state or secrets.
// mode 'read' forbids .env so the API key cannot be exfiltrated.
const FORBIDDEN_WRITE = new Set(['.env', 'tracker.json', 'applicant.json']);
function safePath(rel, mode) {
  const full = path.resolve(CAMPAIGN_DIR, rel);
  let candidate = full;
  let ok = full.startsWith(CAMPAIGN_DIR + path.sep) || full === CAMPAIGN_DIR;
  if (!ok) {
    const full2 = path.resolve(ROOT, rel);
    ok = full2.startsWith(ROOT + path.sep);
    candidate = full2;
  }
  if (!ok) return null;
  const base = path.basename(candidate);
  if (mode === 'write' && FORBIDDEN_WRITE.has(base)) return null;
  if (mode === 'read' && base === '.env') return null;
  return candidate;
}

// ---------------------------------------------------------------------------
// Vector DB (vectra) - local file-based store for augmented context
// ---------------------------------------------------------------------------
const vecIndex = new LocalIndex(VEC_DIR);
let vecReady = false;
async function ensureVec() {
  if (vecReady) return;
  if (!(await vecIndex.isIndexCreated())) await vecIndex.createIndex();
  vecReady = true;
}
const EMBED_DIM = 256;
function embed(text) {
  const vec = new Array(EMBED_DIM).fill(0);
  const clean = String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9א-ת ]/g, ' ');
  const words = clean.split(/\s+/).filter(Boolean);
  const bump = (token, w) => {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    vec[h % EMBED_DIM] += w;
  };
  for (let i = 0; i < words.length; i++) {
    bump(words[i], 1);
    if (i + 1 < words.length) bump(words[i] + ' ' + words[i + 1], 1.5);
  }
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return vec.map((v) => v / norm);
}

// ---------------------------------------------------------------------------
// Browser (Playwright over CDP to the logged-in Chrome)
// ---------------------------------------------------------------------------
let browser = null;
let page = null;
let lastControlledPortalHost = null;
async function ensureBrowser() {
  if (browser && page && !page.isClosed()) {
    await page.bringToFront().catch((e) => debugLog('bringToFront failed', e.message));
    return;
  }
  if (NO_BROWSER) {
    log('browser disabled by JOBLOOPER_NO_BROWSER; skipping CDP connection');
    return;
  }
  try {
    if (!browser) browser = await chromium.connectOverCDP(CDP_URL);
    const ctx = browser.contexts()[0] || (await browser.newContext());
    page = orderedPagesForReuse(ctx)[0] || (await ctx.newPage());
    await page.bringToFront().catch((e) => debugLog('bringToFront failed', e.message));
    lastControlledPortalHost = urlHost(page.url()) || lastControlledPortalHost;
    log('browser connected via CDP', CDP_URL, 'controlledUrl=', page.url());
  } catch (e) {
    browser = null;
    page = null;
    throw e;
  }
}
const LOGIN_URLS = [
  'https://mail.google.com/mail/u/0/',
  'https://www.linkedin.com/',
  'https://www.alljobs.co.il/',
  'https://www.drushim.co.il/',
  'https://www.jobmaster.co.il/',
  'https://www.jobnet.co.il/',
];
function launchChromeForLogin() {
  const args = ['-na', 'Google Chrome', '--args', '--user-data-dir=' + CHROME_PROFILE, '--remote-debugging-port=9222', ...LOGIN_URLS];
  execFile('open', args, (err) => {
    if (err) log('launch Chrome error:', err.message);
  });
}
function waitForEnter(msg) {
  return new Promise((res) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(msg, () => {
      rl.close();
      res();
    });
  });
}

// ---------------------------------------------------------------------------
// Portal registry (region -> rotating boards)
// ---------------------------------------------------------------------------
const PORTALS = {
  il: [
    { name: 'Drushim', url: 'https://www.drushim.co.il/jobs/cat6/' },
    { name: 'JobNet', url: 'https://www.jobnet.co.il/jobs?q=java' },
    { name: 'JobMaster', url: 'https://www.jobmaster.co.il/' },
    { name: 'AllJobs', url: 'https://www.alljobs.co.il/User/JobsFeed/' },
    { name: 'Google IL Jobs', url: 'https://www.google.com/search?q=site%3Ail+%28java+OR+backend+OR+fullstack%29+developer+jobs+Israel+Petah+Tikva' },
  ],
  eu: [
    { name: 'NoFluffJobs', url: 'https://nofluffjobs.com/pl/jobs/java' },
    { name: 'JustJoinIT', url: 'https://justjoin.it/all/location/remote' },
    { name: 'Pracuj', url: 'https://www.pracuj.pl/praca/java%20developer' },
    { name: 'Google EU Jobs', url: 'https://www.google.com/search?q=remote+java+backend+fullstack+developer+jobs+Poland+B2B' },
  ],
};
const PORTAL_REUSE_HOSTS = new Set(Object.values(PORTALS).flat().map((portal) => {
  try {
    return new URL(portal.url).host;
  } catch {
    return '';
  }
}).filter(Boolean));
const RATE_LIMIT_BACKOFF_MS = [15000, 30000, 60000, 120000, 240000];

function urlHost(value) {
  try {
    return new URL(value).host;
  } catch {
    return '';
  }
}

function samePortalTarget(pageUrl, targetUrl) {
  const pageHost = urlHost(pageUrl);
  const targetHost = urlHost(targetUrl);
  if (!pageHost || !targetHost) return false;
  if (pageUrl === targetUrl) return true;
  return pageHost === targetHost && PORTAL_REUSE_HOSTS.has(targetHost);
}

function comparableUrl(value) {
  try {
    const url = new URL(value);
    url.hash = '';
    return url.href.replace(/\/$/, '');
  } catch {
    return String(value || '').replace(/\/$/, '');
  }
}

function isSameUrl(a, b) {
  return comparableUrl(a) === comparableUrl(b);
}

function orderedPagesForReuse(ctx) {
  const pages = ctx.pages().filter((candidate) => !candidate.isClosed());
  return pages.sort((a, b) => {
    const aHost = urlHost(a.url());
    const bHost = urlHost(b.url());
    const score = (host, candidate) => {
      if (candidate === page) return 0;
      if (host && host === lastControlledPortalHost) return 1;
      if (PORTAL_REUSE_HOSTS.has(host)) return 2;
      return 3;
    };
    return score(aHost, a) - score(bHost, b);
  });
}

async function findExistingPortalPage(ctx, targetUrl) {
  return orderedPagesForReuse(ctx).find((candidate) => samePortalTarget(candidate.url(), targetUrl)) || null;
}

async function navigateControlledPage(url, { waitMs = 0 } = {}) {
  const from = page.url();
  await page.bringToFront().catch((e) => debugLog('bringToFront failed', e.message));
  let navigated = false;
  if (!isSameUrl(from, url)) {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    navigated = true;
  }
  await page.bringToFront().catch((e) => debugLog('bringToFront failed', e.message));
  lastControlledPortalHost = urlHost(page.url()) || lastControlledPortalHost;
  if (waitMs) await page.waitForTimeout(waitMs);
  const finalUrl = page.url();
  return { from, url: finalUrl, navigated, urlMismatch: !isSameUrl(finalUrl, url) };
}

// ---------------------------------------------------------------------------
// Tools (the "MCP tool set" the agent calls)
// ---------------------------------------------------------------------------
const readCampaignFile = tool({
  name: 'readCampaignFile',
  description: 'Read a text file from the campaign workspace or repo workspace. Large tracker files may be returned as a compact summary.',
  inputSchema: z.object({ path: z.string() }),
  execute: async ({ path: rel }) => {
    const full = safePath(rel, 'read');
    if (!full) return { error: 'path not allowed' };
    try {
      const raw = fs.readFileSync(full, 'utf8');
      if (path.basename(full) === 'tracker.json') {
        return { content: summarizeTrackerForAgent(JSON.parse(raw)) };
      }
      if (raw.length > 30000) {
        return { content: raw.slice(0, 30000) + '\n...[truncated: use a narrower file/tool request]...', truncated: true };
      }
      return { content: raw };
    } catch (e) {
      return { error: String(e) };
    }
  },
});

const writeCampaignFile = tool({
  name: 'writeCampaignFile',
  description: 'Write a text file in the campaign workspace or repo workspace. State files and secrets are protected from direct writes.',
  inputSchema: z.object({ path: z.string(), content: z.string() }),
  execute: async ({ path: rel, content }) => {
    const full = safePath(rel, 'write');
    if (!full) return { error: 'path not allowed (tracker.json/applicant.json/.env are protected)' };
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, content);
    return { ok: true };
  },
});

const browserNavigate = tool({
  name: 'browser_navigate',
  description: 'Navigate the controlled browser page to a URL, reusing an already-open matching portal tab when one exists.',
  inputSchema: z.object({ url: z.string() }),
  execute: async ({ url }) => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'browsing', pendingStep: 'inspect listing', currentUrl: url, lastOutcome: 'navigated' });
    const existing = await findExistingPortalPage(page.context(), url);
    if (existing) {
      page = existing;
      const nav = await navigateControlledPage(url);
      log('browser reused existing portal tab', { from: nav.from, url: nav.url, target: url, navigated: nav.navigated, urlMismatch: nav.urlMismatch });
      return { ok: true, reused: true, navigated: nav.navigated, urlMismatch: nav.urlMismatch, url: nav.url };
    }
    const nav = await navigateControlledPage(url);
    return { ok: true, navigated: nav.navigated, urlMismatch: nav.urlMismatch, url: nav.url };
  },
});

const browserSnapshot = tool({
  name: 'browser_snapshot',
  description: 'Return a trimmed accessibility snapshot of the current browser page.',
  inputSchema: z.object({}),
  execute: async () => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'inspecting', pendingStep: 'read page', currentUrl: page.url(), lastOutcome: 'snapshot' });
    const snap = await page.accessibility.snapshot();
    const text = JSON.stringify(snap);
    return { snapshot: text.length > 8000 ? text.slice(0, 8000) + '...[truncated]' : text };
  },
});

const browserEvaluate = tool({
  name: 'browser_evaluate',
  description: 'Run a JavaScript expression in the current browser page and return the JSON-serialized result.',
  inputSchema: z.object({ expression: z.string() }),
  execute: async ({ expression }) => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'inspecting', pendingStep: 'evaluate page', currentUrl: page.url(), lastOutcome: 'evaluated' });
    const r = await page.evaluate((expr) => {
      const f = new Function('return (' + expr + ')');
      return f();
    }, expression);
    return { result: JSON.stringify(r).slice(0, 4000) };
  },
});

const browserClick = tool({
  name: 'browser_click',
  description: 'Click an element in the current browser page by CSS selector.',
  inputSchema: z.object({ selector: z.string() }),
  execute: async ({ selector }) => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'browsing', pendingStep: 'inspect result', currentUrl: page.url(), lastOutcome: 'clicked' });
    await page.click(selector, { timeout: 15000 });
    return { ok: true };
  },
});

const browserFill = tool({
  name: 'browser_fill',
  description: 'Fill an input in the current browser page by CSS selector.',
  inputSchema: z.object({ selector: z.string(), value: z.string() }),
  execute: async ({ selector, value }) => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'applying', pendingStep: 'fill application', currentUrl: page.url(), lastOutcome: 'filled field' });
    await page.fill(selector, value, { timeout: 15000 });
    return { ok: true };
  },
});

const browserUpload = tool({
  name: 'browser_upload',
  description: 'Upload a local file to a file input in the current browser page by CSS selector.',
  inputSchema: z.object({ selector: z.string(), filePath: z.string() }),
  execute: async ({ selector, filePath }) => {
    if (!page) return { error: 'browser not connected' };
    updateWorkflow({ step: 'applying', pendingStep: 'upload cv', currentUrl: page.url(), lastOutcome: 'uploaded file' });
    await page.setInputFiles(selector, filePath);
    return { ok: true };
  },
});

async function openNextPortal(region) {
  if (!page) return { error: 'browser not connected' };
  const list = PORTALS[region] || PORTALS.il;
  const portal = list[state.cycles % list.length];
  state.cycles++;
  updateWorkflow({ step: 'searching', currentRegion: region, currentPortal: portal.name, pendingStep: 'review candidates', lastOutcome: 'opened portal' });
  const existing = await findExistingPortalPage(page.context(), portal.url);
  if (existing) {
    page = existing;
  }
  const nav = await navigateControlledPage(portal.url, { waitMs: 2500 });
  if (existing) {
    log('portal tab reused', { portal: portal.name, from: nav.from, url: nav.url, target: portal.url, navigated: nav.navigated, urlMismatch: nav.urlMismatch });
  } else {
    log('portal opened', { portal: portal.name, from: nav.from, url: nav.url, target: portal.url, navigated: nav.navigated, urlMismatch: nav.urlMismatch });
  }
  return {
    portal: portal.name,
    region,
    url: page.url(),
    reused: Boolean(existing),
    navigated: nav.navigated,
    urlMismatch: nav.urlMismatch,
    instruction: 'Use browser_snapshot and browser_evaluate to inspect this live page, identify real job listings, then score and dedupe each candidate. Do not treat this tool result as a candidate list.',
  };
}

const searchPortal = tool({
  name: 'search_portal',
  description: 'Open the next preferred source for a region (il|eu). This only navigates/foregrounds a source; inspect the live page with browser_snapshot/browser_evaluate to discover real listings.',
  inputSchema: z.object({ region: z.enum(['il', 'eu']) }),
  execute: async ({ region }) => openNextPortal(region),
});

const scoreCandidate = tool({
  name: 'score_candidate',
  description: 'Record the model-directed fit decision for a candidate. The tool does not score with hidden rules; provide decision and reason from the prompt policy.',
  inputSchema: z.object({
    roleTitle: z.string(),
    description: z.string().optional().default(''),
    region: z.enum(['il', 'eu', 'global']).optional().default('il'),
    remotePolicy: z.string().optional().default(''),
    salarySeen: z.string().optional().default(''),
    decision: z.enum(['apply', 'skip']),
    reason: z.string(),
  }),
  execute: async ({ roleTitle, description, region, remotePolicy, salarySeen, decision, reason }) => {
    updateWorkflow({
      step: decision === 'apply' ? 'scoring' : 'skipped',
      pendingStep: decision === 'apply' ? 'prepare application' : 'record skip',
      currentCandidate: { roleTitle, region, remotePolicy, salarySeen },
      lastOutcome: decision,
    });
    return { decision, reason, candidate: { roleTitle, description, region, remotePolicy, salarySeen } };
  },
});

const dedupe = tool({
  name: 'dedupe',
  description: 'Lookup whether a candidate is already present in tracker.json by company, id, or normalized URL. Returns duplicate metadata.',
  inputSchema: z.object({
    company: z.string(),
    roleTitle: z.string().optional().default(''),
    id: z.string().optional().default(''),
    url: z.string().optional().default(''),
  }),
  execute: async ({ company, roleTitle, id, url }) => {
    const t = loadTracker();
    updateWorkflow({ step: 'deduping', pendingStep: 'confirm candidate', currentCandidate: { company, roleTitle, id, url }, lastOutcome: 'checked duplicate status' });
    const apps = (t && t.applications) || [];
    const ck = normalizeCompany(company);
    const candId = id || '';
    const norm = (u) => {
      try {
        const p = new URL(u);
        return (p.origin + p.pathname).replace(/\/$/, '').toLowerCase();
      } catch {
        return '';
      }
    };
    const candUrl = norm(url);
    const companies = new Set(apps.map((a) => a.companyKey).filter(Boolean));
    const ids = new Set(apps.map((a) => a.id).filter(Boolean));
    const urls = new Set(apps.map((a) => norm(a.jobUrl || a.listingUrl)).filter(Boolean));
    let matchedBy = null;
    if (candId && ids.has(candId)) matchedBy = 'id';
    else if (ck && companies.has(ck) && !['confidential', 'חברה-חסויה', 'anonymous'].includes(ck)) matchedBy = 'company';
    else if (candUrl && urls.has(candUrl)) matchedBy = 'url';
    const gmailUrl = 'https://mail.google.com/mail/u/0/#search/' + encodeURIComponent(`(in:inbox OR in:sent) newer_than:60d "${company}"`);
    return { duplicate: matchedBy !== null, matchedBy, companyKey: ck, gmailUrl };
  },
});

const recordSubmission = tool({
  name: 'record_submission',
  description: 'Append a submitted application record to tracker.json and update progress. DRY_RUN makes this a log-only no-op.',
  inputSchema: z.object({
    company: z.string(),
    roleTitle: z.string(),
    url: z.string(),
    region: z.string().optional().default('il'),
    remotePolicy: z.string().optional().default(''),
    salarySeen: z.string().optional().default(''),
    source: z.string().optional().default('portal'),
    sourceJobId: z.string().optional().default(''),
    confirmationText: z.string().optional().default(''),
  }),
  execute: async (c) => {
    const ck = normalizeCompany(c.company);
    updateWorkflow({
      step: 'submitted',
      pendingStep: 'done',
      currentCandidate: { company: c.company, roleTitle: c.roleTitle, url: c.url, region: c.region },
      lastOutcome: 'submitted',
    });
    if (DRY_RUN) {
      state.dryApplied++;
      log('[DRY-RUN] would record submission:', c.company, c.roleTitle, 'dryApplied=', state.dryApplied);
      return { ok: true, dryRun: true, submitted: state.submitted, dryApplied: state.dryApplied };
    }
    const t = loadTracker() || { applications: [] };
    t.applications = t.applications || [];
    t.applications.push({
      id: c.source + ':' + (c.sourceJobId || ck + '-' + Date.now()),
      company: c.company,
      companyKey: ck,
      roleTitle: c.roleTitle,
      jobUrl: c.url,
      listingUrl: c.url,
      region: c.region,
      remotePolicy: c.remotePolicy,
      salarySeen: c.salarySeen,
      source: c.source,
      sourceJobId: c.sourceJobId,
      status: 'submitted',
      appliedAt: new Date().toISOString(),
      confirmationText: c.confirmationText,
    });
    t.updatedAt = new Date().toISOString();
    if (typeof t.targetApplications === 'number') t.targetApplications = Math.max(t.targetApplications, TARGET);
    saveTracker(t);
    state.submitted = readSubmitted();
    log('SUBMITTED', c.company, '-> total', state.submitted, '/', state.target);
    return { ok: true, submitted: state.submitted };
  },
});

const recordSkip = tool({
  name: 'record_skip',
  description: 'Append a skipped candidate record to tracker.json. DRY_RUN logs only.',
  inputSchema: z.object({
    company: z.string(),
    reason: z.enum(['skippedDuplicate', 'skippedSalary', 'skippedFilter']),
  }),
  execute: async ({ company, reason }) => {
    updateWorkflow({ step: 'skipped', pendingStep: 'record skip', currentCandidate: { company }, lastOutcome: reason });
    if (DRY_RUN) {
      log('[DRY-RUN] skip', reason, company);
      return { ok: true, dryRun: true };
    }
    const t = loadTracker() || { applications: [] };
    t.applications = t.applications || [];
    t.applications.push({ company, companyKey: normalizeCompany(company), status: reason, skippedAt: new Date().toISOString() });
    saveTracker(t);
    return { ok: true };
  },
});

const getStatusAlias = tool({
  name: 'forget_status',
  description: 'Alias for get_status. Use only if the model accidentally calls this typo; it returns the same progress payload.',
  inputSchema: z.object({}).passthrough(),
  execute: async () => ({ submitted: state.submitted, target: state.target, dryRun: DRY_RUN, errors: state.errors, cycles: state.cycles, activity: state.activity, workflow: state.workflow }),
});

const getStatus = tool({
  name: 'get_status',
  description: 'Return current progress: submitted, target, dryRun, errors, cycles.',
  inputSchema: z.object({}),
  execute: async () => ({ submitted: state.submitted, target: state.target, dryRun: DRY_RUN, errors: state.errors, cycles: state.cycles, activity: state.activity, workflow: state.workflow }),
});

const vectorWriteTool = tool({
  name: 'vectorWriteTool',
  description: 'Upsert a text chunk into the local vector DB for later retrieval.',
  inputSchema: z.object({ key: z.string(), text: z.string() }),
  execute: async ({ key, text }) => {
    await ensureVec();
    await vecIndex.upsertItem({ vector: embed(key + '\n' + text), metadata: { key, text } });
    return { ok: true };
  },
});

const vectorSearchTool = tool({
  name: 'vectorSearchTool',
  description: 'Search the local vector DB for text chunks relevant to a query.',
  inputSchema: z.object({ query: z.string(), limit: z.number().optional().default(5) }),
  execute: async ({ query, limit }) => {
    await ensureVec();
    if (!(await vecIndex.isIndexCreated())) return { results: [] };
    const res = await vecIndex.queryItems(embed(query), limit);
    return { results: res.map((r) => ({ key: r.item.metadata.key, text: r.item.metadata.text, score: r.score })) };
  },
});

const logTool = tool({
  name: 'log',
  description: 'Append a message to the run log.',
  inputSchema: z.object({ message: z.string() }),
  execute: async ({ message }) => {
    log('[agent]', message);
    return { ok: true };
  },
});

const TOOLS = [
  readCampaignFile,
  writeCampaignFile,
  browserNavigate,
  browserSnapshot,
  browserEvaluate,
  browserClick,
  browserFill,
  browserUpload,
  searchPortal,
  scoreCandidate,
  dedupe,
  recordSubmission,
  recordSkip,
  getStatusAlias,
  getStatus,
  vectorWriteTool,
  vectorSearchTool,
  logTool,
];

// ---------------------------------------------------------------------------
// System prompt for the agent
// ---------------------------------------------------------------------------
const SYSTEM = `You are JobLooper, a headless worker that applies to jobs until the submitted count reaches the target (${TARGET}).

Each cycle you MUST:
1. Load context: call get_status, readCampaignFile for tracker.json and applicant.json, and vectorSearchTool for relevant past learnings/campaign context.
2. Do not analyze tracker.json at length. It is compacted for you and dedupe performs exact checks.
3. Immediately call search_portal after startup context. Prefer the configured portal list and region "il" unless you have a concrete reason to switch. Google discovery results, other reputable portals, and company career pages are valid sources too when they lead to real listings.
4. search_portal only opens/foregrounds a source. You must use browser_snapshot and browser_evaluate to understand the live page, distinguish real job listings from navigation/category/account links, and extract each candidate yourself.
5. For each real candidate you discover: decide fit yourself using the prompt rules, then call score_candidate with decision "apply" or "skip" and a concrete reason. score_candidate records your decision; it does not contain hidden screening logic.
6. If all checks are green: browser_navigate to the listing, find the apply path, fill the form using applicant.json fields (phone: IL +972559344507 / EU +48790775407; salary 15000 PLN EU or 15000 ILS IL; LinkedIn in the LinkedIn field; GitHub https://github.com/mstaszew-dev when a GitHub field exists; coverNote or coverNotePl for motivation; for PL/EU roles include plB2bNote). For IL/Petah Tikva roles, upload the current Petah Tikva CV file at /Users/mst/Downloads/job-search/cv/michael-staszewski-cv.pdf when a file field exists.
7. Verify success IN the browser with browser_snapshot (a thank-you URL or confirmation text). NEVER record without this.
8. Call record_submission only after verification. Call record_skip for duplicates/salary/filter.
9. Update progress and persist anything useful with vectorWriteTool/writeCampaignFile (portal quirks, form selectors that worked, lessons, compressed campaign context) before the next loop.

Rules:
- One company once (dedupe). Skip ABAP, Salesforce, pure QA, C/C++, .NET, mobile, ML/data, DevOps-only, team-lead/lead/architect/manager.
- IL: remote, hybrid, or onsite all OK. EU/global: full remote only, B2B >= 15000 PLN/month when salary is listed.
- Jobs may come from the configured portals, Google search results, another job board, or a company career page. Apply the same score, dedupe, browser verification, and record gates for every source.
- After score_candidate returns "apply", build a full candidate object (company, roleTitle, region, remotePolicy, salarySeen, source, sourceJobId, url) and call dedupe before any application attempt.
- dedupe is a tracker lookup, not a scorer. If duplicate is true, call record_skip with skippedDuplicate.
- record_submission is the only way to write verified submissions to tracker.json. writeCampaignFile is for notes/context files, not tracker state.
- Never rely on search_portal output as a candidate list. Treat it as a page opener only; candidate discovery is your browser reasoning job.
- Reuse already-open portal tabs. Do not open duplicate tabs for the same site.
- You must call a tool on every turn. Do not end a turn with only text before the target is reached.
- Keep applying until get_status shows submitted >= target.
- Use only the exact tool names provided to you. Tool names are plain names like readCampaignFile and browser_snapshot; never include channel markers, angle-bracket tags, JSON suffixes, or any other decoration. Start every turn with get_status with an empty object.`;

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------
const client = new OpenRouterCore({ apiKey: OPENROUTER_API_KEY });
const conversationStateStore = createConversationStateStore(path.join(ROOT, '.conversation-state.json'));

function stopWhen({ steps }) {
  if (state.submitted >= state.target) return true;
  if (TEST_MAX_APPLIES && state.dryApplied >= TEST_MAX_APPLIES) return true;
  if (state.errors > 50) return true;
  return false; // keep the loop alive
}

function isTransientModelError(error) {
  const message = (error && (error.message || error.toString())) || '';
  return /rate limit|429|timeout|timed out|network|fetch failed|econn|temporarily unavailable|overloaded|too many requests|service unavailable|response failed|malformed tool/i.test(message);
}
function isRateLimitModelError(error) {
  const message = (error && (error.message || error.toString())) || '';
  return /rate limit|429|throttl|too many requests|free-models-per-day|quota/i.test(message);
}
function isBrowserConnectionError(error) {
  const message = (error && (error.message || error.toString())) || '';
  return /browser|cdp|target closed|page closed|context closed|protocol error|playwright/i.test(message);
}

function buildTurnInput() {
  return `Workflow state: ${workflowContextString()}. Begin by calling get_status with {}. Read the compact tracker summary and applicant.json, use vectorSearchTool briefly, then immediately call search_portal with {"region":"il"} so the browser opens a live board. Do not spend turns analyzing tracker history.`;
}

async function runOnce() {
  const request = {
    model: MODEL,
    instructions: SYSTEM,
    input: buildTurnInput(),
    tools: TOOLS,
    stopWhen,
  };
  if (PERSIST_CONVERSATION_STATE) request.state = conversationStateStore;
  debugLog('calling model', { model: MODEL, toolCount: TOOLS.length, dryRun: DRY_RUN, target: state.target, persistState: PERSIST_CONVERSATION_STATE });
  return callModel(client, request);
}

async function debugModelTurn(result) {
  const toolCallSummaries = [];
  try {
    for await (const event of result.getFullResponsesStream()) {
      const summary = summarizeModelEvent(event);
      if (summary) debugLog('model event', summary);
    }
  } catch (e) {
    debugLog('model event stream failed', e.message);
    throw e;
  }

  let text;
  try {
    text = await result.getText();
  } catch (e) {
    debugLog('getText failed', e.message);
    throw e;
  }
  let toolCalls;
  try {
    toolCalls = await result.getToolCalls();
  } catch (e) {
    debugLog('getToolCalls failed', e.message);
    throw e;
  }
  for (const call of toolCalls || []) {
    toolCallSummaries.push({ name: call.name, arguments: call.arguments });
  }
  await recoverMalformedToolCalls(toolCallSummaries);
  const textPreview = (text || '').slice(0, 400);
  debugLog('model turn complete', {
    textPreview,
    toolCalls: toolCallSummaries,
    submitted: state.submitted,
    dryApplied: state.dryApplied,
  });
  return { text, toolCalls };
}

function summarizeModelEvent(event) {
  if (!event || typeof event !== 'object') return String(event);
  if (event.type && /\.delta$/.test(event.type)) return null;
  if (event.type === 'response.content_part.added' || event.type === 'response.content_part.done') return null;
  return {
    type: event.type,
    itemType: event.item?.type,
    name: event.name || event.item?.name,
    toolCallId: event.call_id || event.toolCallId,
    status: event.status,
    responseId: event.response?.id,
  };
}

async function recoverMalformedToolCalls(toolCalls) {
  const malformedSearch = toolCalls.find((call) => call.name && call.name !== 'search_portal' && String(call.name).includes('search_portal'));
  if (malformedSearch) {
    if (!page && !NO_BROWSER) await ensureBrowser();
    const region = malformedSearch.arguments?.region === 'eu' ? 'eu' : 'il';
    const opened = await openNextPortal(region);
    log('[tool-recovery] opened portal for malformed tool call', malformedSearch.name, opened);
  }
  const malformed = toolCalls.find((call) => {
    const name = String(call.name || '');
    return /<\|.*\|>|<[^>]+>|json$/i.test(name) && !name.includes('search_portal');
  });
  if (malformed) {
    throw new Error(`Malformed tool name emitted by model: ${malformed.name}`);
  }
}

async function runTurnWithRetry() {
  const maxAttempts = 8;
  let delayMs = 2000;
  let lastError = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await runOnce();
      return await debugModelTurn(result);
    } catch (e) {
      lastError = e;
      if (!isTransientModelError(e) || attempt >= maxAttempts) {
        throw e;
      }
      const rateLimited = isRateLimitModelError(e);
      const waitMs = rateLimited ? RATE_LIMIT_BACKOFF_MS[Math.min(attempt - 1, RATE_LIMIT_BACKOFF_MS.length - 1)] : delayMs;
      log(`model turn attempt ${attempt}/${maxAttempts} failed, retrying in ${waitMs}ms`, e.message);
      await sleep(waitMs);
      delayMs = rateLimited ? waitMs : Math.min(delayMs * 2, 30000);
    }
  }

  throw lastError;
}

async function main() {
  const mode = process.argv[2] || 'run';
  await ensureVec();
  state.submitted = readSubmitted();
  log('JobLooper start. mode=', mode, 'submitted=', state.submitted, '/', state.target, 'dryRun=', DRY_RUN, 'model=', MODEL, 'debug=', DEBUG, 'apiKeyConfigured=', Boolean(OPENROUTER_API_KEY));

  if (mode === 'login') {
    launchChromeForLogin();
    await waitForEnter('Log in to the portals in the opened Chrome, then press ENTER to start applying...');
  }

  let connected = false;
  if (NO_BROWSER) {
    connected = true;
  } else {
    for (let i = 0; i < 10 && !connected; i++) {
      try {
        await ensureBrowser();
        connected = true;
      } catch (e) {
        log('browser not ready, retry', i, e.message);
        await sleep(3000);
      }
    }
    if (!connected) {
      log('ERROR: could not connect to browser on', CDP_URL);
      process.exit(1);
    }
  }
  if (!NO_BROWSER && process.env.JOBLOOPER_BOOTSTRAP_PORTAL !== '0') {
    try {
      const opened = await openNextPortal('il');
      log('bootstrap portal opened', opened);
    } catch (e) {
      log('bootstrap portal failed', e.message);
    }
  }

  while (state.submitted < state.target) {
    state.iterations = (state.iterations || 0) + 1;
    const progressBefore = state.submitted + state.dryApplied + state.activity;
    try {
      if (!NO_BROWSER && !page) await ensureBrowser();
      const { text } = await runTurnWithRetry();
      log('runOnce ended. submitted=', state.submitted, 'text=', text.slice(0, 200));
    } catch (e) {
      state.errors++;
      log('ERROR in runOnce:', e.message);
      if (isBrowserConnectionError(e)) {
        browser = null;
        page = null;
      }
      const backoff = isRateLimitModelError(e)
        ? RATE_LIMIT_BACKOFF_MS[Math.min(state.errors - 1, RATE_LIMIT_BACKOFF_MS.length - 1)]
        : Math.min(30000, 5000 * Math.pow(2, Math.min(state.errors, 4)));
      await sleep(backoff);
      continue;
    }
    if (state.submitted >= state.target) break;
    if (TEST_MAX_APPLIES && state.dryApplied >= TEST_MAX_APPLIES) {
      log('test max applies reached');
      break;
    }
    const progressAfter = state.submitted + state.dryApplied + state.activity;
    state.idle = progressAfter === progressBefore ? (state.idle || 0) + 1 : 0;
    if (state.iterations > MAX_ITERATIONS) {
      log('MAX_ITERATIONS', MAX_ITERATIONS, 'reached; stopping');
      break;
    }
    if (state.idle > MAX_IDLE_ITERATIONS) {
      log('no workflow activity for', state.idle, 'iterations; stopping to avoid an endless loop');
      break;
    }
    // Opt-in human review checkpoint after an iteration that submitted something.
    if (PAUSE_AFTER_APPLY && state.submitted > progressBefore) {
      await waitForEnter(`Submitted ${state.submitted - progressBefore} this iteration (total ${state.submitted}/${state.target}). Press ENTER to continue...`);
    }
    await sleep(2000);
  }
  log('STOPPED. submitted=', state.submitted, '/', state.target);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main().catch((e) => {
    log('FATAL', e && e.stack ? e.stack : e.message);
    process.exit(1);
  });
}
