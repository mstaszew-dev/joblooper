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
import { execFile } from 'node:child_process';

// ---------------------------------------------------------------------------
// Config (from .env)
// ---------------------------------------------------------------------------
const ROOT = process.cwd();
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
const OPENROUTER_API_KEY = (process.env.OPENROUTER_API_KEY || '').trim();
const VEC_DIR = path.join(ROOT, '.vectors');
const LOG_FILE = path.join(ROOT, 'joblooper.log');
const TRACKER_FILE = path.join(CAMPAIGN_DIR, 'tracker.json');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

// ---------------------------------------------------------------------------
// Campaign state (read fresh from disk each call so the agent never drifts)
// ---------------------------------------------------------------------------
const state = { submitted: 0, target: TARGET, errors: 0, cycles: 0, dryApplied: 0 };

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
async function ensureBrowser() {
  if (browser) return;
  if (NO_BROWSER) {
    log('browser disabled by JOBLOOPER_NO_BROWSER; skipping CDP connection');
    return;
  }
  try {
    browser = await chromium.connectOverCDP(CDP_URL);
    const ctx = browser.contexts()[0] || (await browser.newContext());
    page = ctx.pages()[0] || (await ctx.newPage());
    log('browser connected via CDP', CDP_URL);
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
    { name: 'JobMaster', url: 'https://www.jobmaster.co.il/%D7%9E%D7%A9%D7%A8%D7%95%D7%AA-%D7%A4%D7%99%D7%AA%D7%95%D7%97-%D7%A8%D7%A7%D7%A2/%D7%AA%D7%95%D7%9B%D7%A0%D7%99%D7%AA/' },
    { name: 'AllJobs', url: 'https://www.alljobs.co.il/SearchResults.aspx?position=java&type=2' },
    { name: 'Drushim', url: 'https://www.drushim.co.il/%D7%9E%D7%A4%D7%A8%D7%90%D7%99%D7%9D-%D7%A4%D7%99%D7%AA%D7%95%D7%97-%D7%A8%D7%A7%D7%A2/' },
    { name: 'JobNet', url: 'https://www.jobnet.co.il/jobs?q=java' },
  ],
  eu: [
    { name: 'NoFluffJobs', url: 'https://nofluffjobs.com/pl/jobs/java' },
    { name: 'JustJoinIT', url: 'https://justjoin.it/all/location/remote' },
    { name: 'Pracuj', url: 'https://www.pracuj.pl/praca/java%20developer' },
  ],
};

// ---------------------------------------------------------------------------
// CV-alignment policy (kept in sync with the campaign rules)
// ---------------------------------------------------------------------------
const SKIP_ROLE = /(abap|salesforce|apex|\bqa\b|quality assurance|automation tester|c\+\+|\.net|asp\.net|c#|mobile|android|ios|ml|machine learning|data scientist|devops|sre|site reliability|team lead|tech lead|technical lead|lead developer|lead engineer|principal|staff|architect|manager|director|head|vp)/i;
const ALLOW_ROLE = /(java|kotlin|spring|php|laravel|symfony|node|nest|react|angular|backend|full.?stack|fullstack|engineer|developer)/i;

// ---------------------------------------------------------------------------
// Tools (the "MCP tool set" the agent calls)
// ---------------------------------------------------------------------------
const readCampaignFile = tool({
  name: 'readCampaignFile',
  description: 'Read a campaign file: tracker.json, applicant.json, PORTALS.md, AGENT_TICK.md, or any notes file. Returns text content.',
  inputSchema: z.object({ path: z.string() }),
  execute: async ({ path: rel }) => {
    const full = safePath(rel, 'read');
    if (!full) return { error: 'path not allowed' };
    try {
      return { content: fs.readFileSync(full, 'utf8') };
    } catch (e) {
      return { error: String(e) };
    }
  },
});

const writeCampaignFile = tool({
  name: 'writeCampaignFile',
  description: 'Write or update a campaign file (e.g. a notes/learnings file). Never overwrite tracker.json or applicant.json; use the record_* tools for those.',
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
  description: 'Navigate the single working browser tab to a URL.',
  inputSchema: z.object({ url: z.string() }),
  execute: async ({ url }) => {
    if (!page) return { error: 'browser not connected' };
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    return { ok: true, url: page.url() };
  },
});

const browserSnapshot = tool({
  name: 'browser_snapshot',
  description: 'Return a trimmed accessibility snapshot of the current page to read text and locate form fields.',
  inputSchema: z.object({}),
  execute: async () => {
    if (!page) return { error: 'browser not connected' };
    const snap = await page.accessibility.snapshot();
    const text = JSON.stringify(snap);
    return { snapshot: text.length > 8000 ? text.slice(0, 8000) + '...[truncated]' : text };
  },
});

const browserEvaluate = tool({
  name: 'browser_evaluate',
  description: 'Run a JS expression in the page and return a JSON string of the result. Use to extract data or inspect the DOM.',
  inputSchema: z.object({ expression: z.string() }),
  execute: async ({ expression }) => {
    if (!page) return { error: 'browser not connected' };
    const r = await page.evaluate((expr) => {
      const f = new Function('return (' + expr + ')');
      return f();
    }, expression);
    return { result: JSON.stringify(r).slice(0, 4000) };
  },
});

const browserClick = tool({
  name: 'browser_click',
  description: 'Click an element by CSS selector.',
  inputSchema: z.object({ selector: z.string() }),
  execute: async ({ selector }) => {
    if (!page) return { error: 'browser not connected' };
    await page.click(selector, { timeout: 15000 });
    return { ok: true };
  },
});

const browserFill = tool({
  name: 'browser_fill',
  description: 'Fill an input by CSS selector with a value.',
  inputSchema: z.object({ selector: z.string(), value: z.string() }),
  execute: async ({ selector, value }) => {
    if (!page) return { error: 'browser not connected' };
    await page.fill(selector, value, { timeout: 15000 });
    return { ok: true };
  },
});

const browserUpload = tool({
  name: 'browser_upload',
  description: 'Upload a file to a file input by CSS selector (used for the CV).',
  inputSchema: z.object({ selector: z.string(), filePath: z.string() }),
  execute: async ({ selector, filePath }) => {
    if (!page) return { error: 'browser not connected' };
    await page.setInputFiles(selector, filePath);
    return { ok: true };
  },
});

const searchPortal = tool({
  name: 'search_portal',
  description: 'Open the next job board for a region (il|eu) and extract listing links (title, company, url). Returns candidate stubs to score and apply to.',
  inputSchema: z.object({ region: z.enum(['il', 'eu']) }),
  execute: async ({ region }) => {
    if (!page) return { error: 'browser not connected' };
    const list = PORTALS[region] || PORTALS.il;
    const portal = list[state.cycles % list.length];
    state.cycles++;
    await page.goto(portal.url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(2500);
    const items = await page.$$eval('a[href]', (as) => {
      const out = [];
      const seen = new Set();
      for (const a of as) {
        const href = a.href || '';
        const text = (a.innerText || a.textContent || '').trim();
        if (!text || text.length < 3) continue;
        if (!/job|career|praca|oferta|mit?|drushim|alljobs|jobmaster/i.test(href) && !/java|php|node|react|developer|engineer|full.?stack/i.test(text)) continue;
        if (seen.has(href)) continue;
        seen.add(href);
        if (out.length >= 12) break;
        out.push({ url: href, title: text.slice(0, 120) });
      }
      return out;
    });
    return { portal: portal.name, region, count: items.length, candidates: items };
  },
});

const scoreCandidate = tool({
  name: 'score_candidate',
  description: 'CV-alignment gate. Pass a candidate {roleTitle, description, region, remotePolicy, salarySeen}. Returns {decision: apply|skip, reason}.',
  inputSchema: z.object({
    roleTitle: z.string(),
    description: z.string().optional().default(''),
    region: z.enum(['il', 'eu', 'global']).optional().default('il'),
    remotePolicy: z.string().optional().default(''),
    salarySeen: z.string().optional().default(''),
  }),
  execute: async ({ roleTitle, description, region, remotePolicy, salarySeen }) => {
    const text = (roleTitle + ' ' + description).toLowerCase();
    if (SKIP_ROLE.test(text)) return { decision: 'skip', reason: 'excluded role/level' };
    if (!ALLOW_ROLE.test(text)) return { decision: 'skip', reason: 'no matching stack' };
    if (region === 'eu' || region === 'global') {
      if (!/remote/i.test(remotePolicy) && !/remote/i.test(text)) return { decision: 'skip', reason: 'EU must be full remote' };
      const m = salarySeen && salarySeen.match(/(\d[\d,]*)\s*(pln|zł|zl)/i);
      if (m) {
        const num = Number(m[1].replace(/,/g, ''));
        if (num > 0 && num < 15000) return { decision: 'skip', reason: 'EU B2B below 15k PLN' };
      }
    }
    return { decision: 'apply', reason: 'matches stack and policy' };
  },
});

const dedupe = tool({
  name: 'dedupe',
  description: 'Check a candidate against tracker.json (one company once) by companyKey, id, and url. Returns {duplicate, matchedBy, companyKey, gmailUrl}.',
  inputSchema: z.object({
    company: z.string(),
    roleTitle: z.string().optional().default(''),
    id: z.string().optional().default(''),
    url: z.string().optional().default(''),
  }),
  execute: async ({ company, roleTitle, id, url }) => {
    const t = loadTracker();
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
  description: 'Record a VERIFIED submission to tracker.json and increment the submitted counter. Only call after confirming success IN the browser. DRY_RUN makes this a log-only no-op.',
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
  description: 'Record a skipped candidate (duplicate/salary/filter) into tracker.json stats. DRY_RUN logs only.',
  inputSchema: z.object({
    company: z.string(),
    reason: z.enum(['skippedDuplicate', 'skippedSalary', 'skippedFilter']),
  }),
  execute: async ({ company, reason }) => {
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

const getStatus = tool({
  name: 'get_status',
  description: 'Return current progress: submitted, target, dryRun, errors, cycles.',
  inputSchema: z.object({}),
  execute: async () => ({ submitted: state.submitted, target: state.target, dryRun: DRY_RUN, errors: state.errors, cycles: state.cycles }),
});

const vectorWriteTool = tool({
  name: 'vectorWriteTool',
  description: 'Embed and store a text chunk (a lesson learned, a portal quirk, campaign notes) into the local vector DB for later retrieval as augmented context.',
  inputSchema: z.object({ key: z.string(), text: z.string() }),
  execute: async ({ key, text }) => {
    await ensureVec();
    await vecIndex.upsertItem({ vector: embed(key + '\n' + text), metadata: { key, text } });
    return { ok: true };
  },
});

const vectorSearchTool = tool({
  name: 'vectorSearchTool',
  description: 'Retrieve the most relevant stored context chunks for a query (past learnings about a portal or role). Returns texts.',
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
1. Load context: call get_status, readCampaignFile for tracker.json and applicant.json, and vectorSearchTool for relevant past learnings.
2. Pick a region (rotate il then eu) and call search_portal to get candidate stubs.
3. For each candidate: call score_candidate. If apply, build a full candidate object (company, roleTitle, region, remotePolicy, salarySeen, source, sourceJobId, url) and call dedupe.
4. If not a duplicate: browser_navigate to the listing, find the apply path, fill the form using applicant.json fields (phone: IL +972559344507 / EU +48790775407; salary 15000 PLN EU or 15000 ILS IL; LinkedIn in the LinkedIn field; GitHub https://github.com/mstaszew-dev when a GitHub field exists; coverNote or coverNotePl for motivation; for PL/EU roles include plB2bNote). Upload the CV file at /Users/mst/Downloads/job-search/cv/michael-staszewski-cv.pdf when a file field exists.
5. Verify success IN the browser with browser_snapshot (a thank-you URL or confirmation text). NEVER record without this.
6. Call record_submission only after verification. Call record_skip for duplicates/salary/filter.
7. Persist anything useful with vectorWriteTool (portal quirks, form selectors that worked, lessons).

Rules:
- One company once (dedupe). Skip ABAP, Salesforce, pure QA, C/C++, .NET, mobile, ML/data, DevOps-only, team-lead/lead/architect/manager.
- IL: remote, hybrid, or onsite all OK. EU/global: full remote only, B2B >= 15000 PLN/month when salary is listed.
- Use only the single browser tab; navigate on it. Do not open extra tabs.
- You must call a tool on every turn. Do not end a turn with only text before the target is reached.
- Keep applying until get_status shows submitted >= target.`;

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------
const client = new OpenRouterCore({ apiKey: OPENROUTER_API_KEY });

function stopWhen({ steps }) {
  if (state.submitted >= state.target) return true;
  if (TEST_MAX_APPLIES && state.dryApplied >= TEST_MAX_APPLIES) return true;
  if (state.errors > 50) return true;
  return false; // keep the loop alive
}

async function runOnce() {
  return callModel(client, {
    model: MODEL,
    instructions: SYSTEM,
    input: 'Begin. Load current campaign context via tools, then start applying. Use vectorSearchTool for relevant past learnings and vectorWriteTool to persist new ones.',
    tools: TOOLS,
    stopWhen,
  });
}

async function main() {
  const mode = process.argv[2] || 'run';
  await ensureVec();
  state.submitted = readSubmitted();
  log('JobLooper start. mode=', mode, 'submitted=', state.submitted, '/', state.target, 'dryRun=', DRY_RUN, 'model=', MODEL);

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

  while (state.submitted < state.target) {
    state.iterations = (state.iterations || 0) + 1;
    const progressBefore = state.submitted + state.dryApplied;
    try {
      const result = await runOnce();
      const text = await result.getText().catch(() => '');
      log('runOnce ended. submitted=', state.submitted, 'text=', text.slice(0, 200));
    } catch (e) {
      state.errors++;
      log('ERROR in runOnce:', e.message);
      browser = null;
      page = null;
      const backoff = Math.min(30000, 5000 * Math.pow(2, Math.min(state.errors, 4)));
      await sleep(backoff);
      continue;
    }
    if (state.submitted >= state.target) break;
    if (TEST_MAX_APPLIES && state.dryApplied >= TEST_MAX_APPLIES) {
      log('test max applies reached');
      break;
    }
    const progressAfter = state.submitted + state.dryApplied;
    state.idle = progressAfter === progressBefore ? (state.idle || 0) + 1 : 0;
    if (state.iterations > MAX_ITERATIONS) {
      log('MAX_ITERATIONS', MAX_ITERATIONS, 'reached; stopping');
      break;
    }
    if (state.idle > MAX_IDLE_ITERATIONS) {
      log('no progress for', state.idle, 'iterations; stopping to avoid an endless loop');
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

main().catch((e) => {
  log('FATAL', e.message);
  process.exit(1);
});
