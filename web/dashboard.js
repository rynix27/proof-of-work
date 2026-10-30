/* Proof of Work dashboard.
 *
 * SECURITY: every piece of dynamic text (task prompts, model answers, agent
 * names, epitaphs, commentary, errors, replay filenames, loan records) is
 * untrusted. It is only ever inserted with textContent / createTextNode via
 * the el() helper below. This file must never use any API that parses a string
 * as HTML or code (see the denylist in tests/test_security.py, which fails the
 * build if one appears). Colors come from a fixed palette, never from API data.
 */
'use strict';

const AGENT_COLORS = {
  Specialist: '#C9A24B', Generalist: '#6E8F7C', LoanShark: '#8A6A2E',
  Hoarder: '#7A8FA6', Gambler: '#B5432E',
};
const FALLBACK_COLORS = ['#C9A24B', '#6E8F7C', '#8A6A2E', '#7A8FA6', '#B5432E', '#9C7A9E'];
const $ = (id) => document.getElementById(id);

let replayMode = false;
let soundOn = false;
let audioCtx = null;
let controlToken = '';        // kept in memory only
let controlsMode = 'open';    // open | token | disabled (from /state)
let busy = false;
let lastSeenTs = 0;
let lastNotableKey = null;
let notableInitialized = false;
let failedPolls = 0;
let currentRunId = null;      // run the user is looking at; sent with control requests (server returns 409 if stale)

// ---------------------------------------------------------------- safe DOM
function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v === undefined || v === null) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = String(v);
      else if (k === 'style') Object.assign(node.style, v);
      else node.setAttribute(k, String(v));
    }
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
  }
  return node;
}
function setChildren(parent, ...children) {
  parent.replaceChildren();
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    parent.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
  }
}
const num = (v, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const str = (v, d = '') => (typeof v === 'string' ? v : v == null ? d : String(v));
const pct = (v) => (typeof v === 'number' && Number.isFinite(v) ? Math.round(v * 100) + '%' : '—');
const fmt = (v, digits = 1) => (typeof v === 'number' && Number.isFinite(v) ? String(Math.round(v * 10 ** digits) / 10 ** digits) : '—');
const colorFor = (name, i) => AGENT_COLORS[name] || FALLBACK_COLORS[i % FALLBACK_COLORS.length];
const obj = (v) => (v && typeof v === 'object' && !Array.isArray(v) ? v : {});
const arr = (v) => (Array.isArray(v) ? v : []);

// ------------------------------------------------------------------- sound
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}
function chime(freqs, durations, type = 'sine', gain = 0.06) {
  if (!soundOn) return;
  const ctx = ensureAudio();
  let t = ctx.currentTime;
  freqs.forEach((f, i) => {
    const osc = ctx.createOscillator(); const g = ctx.createGain();
    osc.type = type; osc.frequency.value = f;
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + durations[i]);
    osc.connect(g).connect(ctx.destination);
    osc.start(t); osc.stop(t + durations[i]);
    t += durations[i] * 0.6;
  });
}
const soundSolved = () => chime([880, 1320], [0.12, 0.15], 'triangle', 0.05);
const soundBankrupt = () => chime([220, 160, 110], [0.25, 0.25, 0.4], 'sawtooth', 0.05);
const soundMonopoly = () => chime([523, 659, 784, 1046], [0.15, 0.15, 0.15, 0.4], 'triangle', 0.06);

function toggleSound() {
  soundOn = !soundOn;
  if (soundOn) ensureAudio();
  const t = $('sound-toggle');
  t.textContent = soundOn ? '♫ sound on' : '♫ sound off';
  t.classList.toggle('on', soundOn);
  t.setAttribute('aria-pressed', String(soundOn));
}
$('sound-toggle').addEventListener('click', toggleSound);
$('sound-toggle').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSound(); }
});

// ---------------------------------------------------------------- requests
async function post(url, body) {
  const headers = { 'Content-Type': 'application/json' };
  if (controlToken) headers['X-Control-Token'] = controlToken;
  if (currentRunId && url.startsWith('/control/') && url !== '/control/reset') headers['X-Run-Id'] = currentRunId;
  const res = await fetch(url, { method: 'POST', headers, body: body ? JSON.stringify(body) : undefined });
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
  return { ok: res.ok, status: res.status, data: obj(data) };
}
function say(msg) { $('control-status').textContent = msg || ''; }

async function control(label, url, body, onOk) {
  if (busy) return;
  busy = true; refreshControlButtons();
  say(label + '…');
  try {
    const r = await post(url, body);
    if (r.ok) { say(''); if (onOk) onOk(r.data); }
    else if (r.status === 401) { controlToken = ''; say('Control token rejected. Use "Unlock controls".'); }
    else if (r.status === 409) say('That run has ended or been replaced; the view has been refreshed.');
    else if (r.status === 429) say('Too many attempts. Wait a moment and try again.');
    else say('Error: ' + str(r.data.error, 'request failed (' + r.status + ')'));
  } catch (_) {
    say('Error reaching the server.');
  } finally {
    busy = false; refreshControlButtons(); poll(true);
  }
}

const CONTROL_IDS = ['btn-pause', 'btn-resume', 'btn-reset', 'speed-select', 'btn-save-replay'];
let lastPaused = false;
function refreshControlButtons() {
  const locked = controlsMode === 'disabled' || (controlsMode === 'token' && !controlToken);
  const off = replayMode || busy || locked;
  CONTROL_IDS.forEach((id) => { $(id).disabled = off; });
  if (!off) { $('btn-pause').disabled = lastPaused; $('btn-resume').disabled = !lastPaused; }
  const unlock = $('btn-unlock');
  unlock.hidden = !(controlsMode === 'token' && !controlToken && !replayMode);
  const why = controlsMode === 'disabled' ? 'Controls are disabled on this public deployment.'
    : locked ? 'Controls need a token.' : '';
  CONTROL_IDS.forEach((id) => { if (why) $(id).title = why; else $(id).removeAttribute('title'); });
}

$('btn-unlock').addEventListener('click', () => {
  const t = prompt('Control token:');
  if (t) { controlToken = t.trim(); say('Token set for this tab.'); refreshControlButtons(); }
});
$('btn-pause').addEventListener('click', () => control('Pausing', '/control/pause'));
$('btn-resume').addEventListener('click', () => control('Resuming', '/control/resume'));
$('btn-reset').addEventListener('click', () => {
  if (!confirm('Reset the current run and start a fresh one?')) return;
  control('Resetting', '/control/reset', undefined, () => {
    $('replay-select').value = ''; replayMode = false;
    lastSeenTs = 0; notableInitialized = false;
  });
});
$('speed-select').addEventListener('change', (e) =>
  control('Changing speed', '/control/speed', { seconds: parseFloat(e.target.value) }));
$('btn-save-replay').addEventListener('click', () => {
  const label = prompt('Name this replay (optional):', 'best-run');
  if (label === null) return;
  control('Saving replay', '/control/save_replay', { label: label || 'manual' },
    (d) => { say('Saved as ' + str(d.filename)); loadReplayList(); });
});

$('btn-ask').addEventListener('click', async () => {
  const promptEl = $('ask-prompt'), answerEl = $('ask-answer'), statusEl = $('ask-status');
  const prompt = promptEl.value.trim();
  if (!prompt) { statusEl.textContent = 'Type a question first.'; return; }
  $('btn-ask').disabled = true;
  try {
    const r = await post('/inject_task', { prompt, answer: answerEl.value.trim() || undefined });
    if (r.ok && r.data.ok) {
      statusEl.textContent = r.data.graded
        ? 'Queued (graded) — will be answered in an upcoming round.'
        : "Queued as an exhibition question (no known answer, so it won't affect balances) — will be answered soon.";
      promptEl.value = ''; answerEl.value = '';
    } else {
      statusEl.textContent = 'Error: ' + str(r.data.error, 'could not queue question');
    }
  } catch (_) {
    statusEl.textContent = 'Error reaching the server.';
  } finally { $('btn-ask').disabled = false; }
});
$('ask-prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btn-ask').click(); });

$('replay-select').addEventListener('change', (e) => {
  replayMode = !!e.target.value;
  refreshControlButtons();
  lastSeenTs = 0;
  poll(true);
  updateExportLinks();
});

function updateExportLinks() {
  const q = replayMode ? '?replay=' + encodeURIComponent($('replay-select').value) : '';
  $('export-summary').href = '/export/summary.json' + q;
  $('export-csv').href = '/export/events.csv' + q;
  $('export-json').href = '/export/events.json' + q;
  $('export-report').href = '/export/report.md' + q;
}

async function loadReplayList() {
  try {
    const res = await fetch('/replays');
    const list = arr(await res.json());
    const sel = $('replay-select');
    const current = sel.value;
    const opts = [el('option', { value: '', text: '— live —' })];
    for (const r of list) {
      const name = str(r.filename);
      if (!name) continue;
      const n = num(r.event_count);
      opts.push(el('option', { value: name, text: `${name} (${n} events${r.summary ? ', ' + str(r.summary) + ' won' : ''})` }));
    }
    setChildren(sel, opts);
    sel.value = current;
  } catch (err) { console.error('replay list failed', err); }
}

// --------------------------------------------------------------- rendering
function renderMode(d) {
  const b = $('mode-banner');
  if (d.replay) {
    b.className = 'mode-banner replay';
    b.textContent = 'REPLAY';
  } else if (d.mode === 'mock') {
    b.className = 'mode-banner mock';
    b.textContent = 'DEMO';
  } else if (d.mode === 'live') {
    b.className = 'mode-banner live';
    b.textContent = 'LIVE';
  } else {
    b.className = 'mode-banner replay';
    b.textContent = 'READY';
  }
}

function renderLedger(d) {
  const balances = obj(d.balances), bankrupt = arr(d.bankrupt), won = obj(d.tasks_won);
  const start = num(d.starting_balance, 100);
  const names = Object.keys(balances).sort((a, b) => num(balances[b]) - num(balances[a]));
  if (!names.length) { setChildren($('ledger'), el('div', { class: 'empty-note', text: 'Waiting for the run to start…' })); return; }
  const maxBalance = Math.max(start, ...names.map((n) => num(balances[n])));
  setChildren($('ledger'), names.map((name, idx) => {
    const bal = num(balances[name]);
    const bust = bankrupt.includes(name);
    const w = Math.max(0, Math.min(100, (bal / maxBalance) * 100));
    const fill = el('div', { class: 'bar-fill' + (bust ? ' bankrupt' : '') });
    fill.style.width = w + '%';
    if (!bust) fill.style.background = colorFor(name, idx);
    return el('div', { class: 'agent-row' },
      el('div', { class: 'agent-name' + (bust ? ' bankrupt' : ''), text: name }),
      el('div', { class: 'bar-track', role: 'img', 'aria-label': `${name} balance ${bust ? 'bankrupt' : bal}` }, fill),
      el('div', { class: 'agent-balance' + (bust ? ' bankrupt' : ''), text: bust ? 'bust' : bal }),
      el('div', { class: 'agent-won', text: `${num(won[name])} solved` }));
  }));
}

function renderChart(history, startBalance) {
  const canvas = $('history-chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 860, h = 220;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const hist = arr(history).filter((p) => p && typeof p === 'object' && p.balances && typeof p.balances === 'object');
  if (!hist.length) {
    ctx.fillStyle = '#9CB3A8'; ctx.font = '13px IBM Plex Mono, monospace';
    ctx.fillText('Waiting for history to accumulate…', 10, h / 2);
    setChildren($('legend'));
    return;
  }
  const names = Object.keys(hist[hist.length - 1].balances);
  let maxVal = num(startBalance, 100);
  hist.forEach((p) => names.forEach((n) => { maxVal = Math.max(maxVal, num(p.balances[n])); }));
  const pad = { left: 34, right: 10, top: 10, bottom: 20 };
  const pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;
  ctx.strokeStyle = 'rgba(239,231,216,0.08)'; ctx.fillStyle = '#9CB3A8'; ctx.font = '10px IBM Plex Mono, monospace';
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph * i / 4);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    ctx.fillText(Math.round(maxVal - (maxVal * i / 4)), 2, y + 3);
  }
  names.forEach((name, idx) => {
    ctx.strokeStyle = colorFor(name, idx); ctx.lineWidth = 2; ctx.beginPath();
    hist.forEach((p, i) => {
      const x = pad.left + (pw * i / Math.max(1, hist.length - 1));
      const y = pad.top + ph - (num(p.balances[name]) / maxVal) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  setChildren($('legend'), names.map((n, i) =>
    el('span', { class: 'legend-item' }, (() => { const s = el('span', { class: 'legend-swatch' }); s.style.background = colorFor(n, i); return s; })(), n)));
}

function renderStats(d) {
  const s = obj(d.stats);
  const cells = [
    [num(s.tasks_total), 'Tasks'],
    [pct(s.success_rate), 'Accuracy (graded)'],
    [num(s.tasks_unclaimed), 'Unclaimed'],
    [num(s.compute_burned), 'Compute consumed'],
    [num(s.rewards_total), 'Rewards paid'],
    [fmt(s.compute_per_correct, 2), 'Compute / correct'],
    [pct(s.escalation_rate), 'Escalation rate'],
    [num(s.ultra_calls) + ' / ' + num(s.nano_calls), 'Ultra / Nano calls'],
    [fmt(d.gini, 2), 'Inequality (Gini)'],
    [num(s.loan_volume), 'Loan volume'],
    [num(s.outstanding_debt), 'Outstanding debt'],
    [pct(s.repayment_rate), 'Repayment rate'],
    [num(s.loans_defaulted), 'Loan defaults'],
    [pct(s.invalid_decision_rate), 'Invalid decisions'],
  ];
  setChildren($('stats'), cells.map(([v, l]) =>
    el('div', { class: 'stat-cell' }, el('div', { class: 'stat-value', text: v }), el('div', { class: 'stat-label', text: l }))));
  const q = num(d.pending_questions);
  $('stats-note').textContent = (d.mode === 'live'
    ? 'Compute figures are simulated costs; real provider billing is not tracked.'
    : 'Mock mode: all figures describe the fake model.') + (q ? ` ${q} question(s) queued.` : '');
}

function table(target, headers, rows, emptyText) {
  if (!rows.length) { setChildren(target, el('tr', {}, el('td', { class: 'empty-note', text: emptyText }))); return; }
  setChildren(target,
    el('thead', {}, el('tr', {}, headers.map((h) => el('th', { scope: 'col', text: h })))),
    el('tbody', {}, rows.map((r) => el('tr', {}, r.map((c) => el('td', { text: c }))))));
}

function renderAgents(d) {
  const a = obj(d.agents);
  const rows = Object.keys(a).map((name) => {
    const x = obj(a[name]);
    return [name + (x.bankrupt ? ' (bust)' : ''), str(x.strategy, '—'), num(x.balance),
      (num(x.net_pl) >= 0 ? '+' : '') + num(x.net_pl), num(x.economic_profit), pct(x.accuracy), num(x.attempts),
      num(x.compute_spent), fmt(x.efficiency, 2), pct(x.escalation_rate), num(x.loans_issued), num(x.outstanding_debt)];
  });
  table($('agent-table'), ['Agent', 'Strategy', 'Balance', 'P/L', 'Econ. profit', 'Accuracy', 'Solves', 'Compute',
    'Reward/compute', 'Escalation rate', 'Loans issued', 'Debt'], rows, 'No data yet.');
}

function renderLoans(d) {
  const s = obj(d.stats);
  $('loan-summary').textContent = `(${num(s.loans_made)} made, ${num(s.loans_repaid)} repaid, ${num(s.loans_defaulted)} defaulted)`;
  const loans = arr(d.loans).slice().reverse().map((l) => {
    l = obj(l);
    return ['#' + str(l.id), `${str(l.lender)} → ${str(l.borrower)}`, num(l.principal), num(l.total_owed),
      num(l.repaid), num(l.outstanding), str(l.status), l.due_round == null ? '—' : num(l.due_round)];
  });
  table($('loan-table'), ['Loan', 'Lender → Borrower', 'Principal', 'Owed', 'Repaid', 'Outstanding', 'Status', 'Due rd'],
    loans, 'No loans yet.');
  const sc = obj(d.sim_config);
  $('loan-note').textContent = sc.interest_rate != null
    ? `Simple interest ${Math.round(num(sc.interest_rate) * 100)}% per loan, due ${num(sc.loan_term_rounds)} rounds after creation; unpaid balance at the due round defaults.`
    : '';
}

async function renderLeaderboard() {
  try {
    const sort = $('leaderboard-sort').value;
    const res = await fetch('/leaderboard?sort=' + encodeURIComponent(sort));
    const data = obj(await res.json());
    const n = num(data.total_runs);
    $('leaderboard-runs').textContent = `(${n} run${n === 1 ? '' : 's'} completed)`;
    const rows = arr(data.standings).map((r) => {
      r = obj(r);
      return [str(r.agent), num(r.wins), num(r.runs), pct(r.win_rate), fmt(r.avg_final_balance),
        pct(r.avg_accuracy), r.avg_net_pl == null ? '—' : fmt(r.avg_net_pl), fmt(r.efficiency, 2), fmt(r.cost_per_correct, 2)];
    });
    table($('leaderboard-table'), ['Agent', 'Wins', 'Runs', 'Win rate', 'Avg balance', 'Accuracy', 'Avg P/L', 'Reward/compute', 'Cost/correct'],
      rows, 'No completed runs yet — let one finish, or lower the round cap for faster results.');
  } catch (err) { console.error('leaderboard fetch failed', err); }
}

function renderCommentary(c) {
  const e = $('commentary');
  if (!c) { e.textContent = 'Commentary will appear as the run develops…'; e.classList.add('empty'); return; }
  e.textContent = str(c.text); e.classList.remove('empty');
}

function renderTask(t) {
  const e = $('current-task');
  if (!t) { e.textContent = 'Waiting for the first task to arrive…'; return; }
  const exhibition = t.injected && num(t.bounty) === 0;
  setChildren(e,
    t.injected ? el('span', { class: 'badge-injected', text: 'ASKED BY YOU' }) : null,
    el('span', { class: 'prompt', text: '“' + str(t.prompt) + '”' }),
    exhibition ? el('span', { class: 'badge-exhibition', text: 'exhibition — ungraded' }) : null,
    el('br'),
    `difficulty: ${str(t.difficulty)} — `, el('span', { class: 'bounty', text: 'bounty ' + num(t.bounty) }));
}

function tag(cls, text) { return el('span', { class: 'tag ' + cls, text }); }
function eventLine(e) {
  e = obj(e);
  const who = str(e.agent);
  switch (e.type) {
    case 'decision':
      return e.action === 'bid' ? [tag('bid', 'BID'), ` ${who} bids in round ${num(e.round)} (confidence ${fmt(e.confidence, 2)})`] : null;
    case 'bid_rejected':
      return [tag('error', 'REJECTED'), ` ${who}'s bid rejected: ${str(e.reason)}`];
    case 'decision_invalid':
      return [tag('error', 'INVALID'), ` ${who} gave an unusable decision (${str(e.code)}) — treated as skip`];
    case 'escalation':
      return [tag('loan', 'ESCALATE'), ` ${who} escalates to Ultra (confidence ${fmt(e.confidence, 2)} < ${fmt(e.cutoff, 2)}, cost ${num(e.cost)})`];
    case 'task_resolved':
      if (e.exhibition) return [tag('exhibition', 'ANSWERED'), ` ${who} answered your question: “${str(e.answer)}” (unverified, no bounty)`];
      return [tag(e.correct ? 'resolved-correct' : 'resolved-wrong', e.correct ? 'SOLVED' : 'FAILED'),
        ` ${who} ${e.correct ? 'solved it, +' + num(e.bounty) : 'got it wrong'} (${str(e.tier)} tier, −${num(e.cost, 0)}) → balance ${num(e.new_balance)}`];
    case 'task_unclaimed':
      return [tag('exhibition', 'UNCLAIMED'), ` round ${num(e.round)}: ${str(e.reason, 'no one took the task')}`];
    case 'loan_made':
      return [tag('loan', 'LOAN'), ` ${str(e.lender)} lends ${num(e.amount)} to ${str(e.borrower)}` +
        (e.total_owed ? ` (owes ${num(e.total_owed)} by round ${num(e.due_round)})` : '')];
    case 'loan_repayment':
      return [tag('loan', 'REPAID'), ` ${str(e.borrower)} pays ${num(e.amount)} to ${str(e.lender)}` + (e.status === 'repaid' ? ' — loan closed' : ` (${num(e.outstanding)} left)`)];
    case 'loan_default':
      return [tag('bankrupt', 'DEFAULT'), ` ${str(e.borrower)} defaults, ${str(e.lender)} writes off ${num(e.written_off)}`];
    case 'loan_rejected':
      return [tag('error', 'LOAN DENIED'), ` ${str(e.code)}`];
    case 'bankruptcy':
      return [tag('bankrupt', 'BANKRUPT'), ` ${who} is out of compute`, e.epitaph ? el('em', { text: ` — “${str(e.epitaph)}”` }) : null];
    case 'monopoly':
      return [tag('bankrupt', 'MONOPOLY'), ` ${str(e.winner)} is the last solvent agent`];
    case 'call_error':
      return [tag('error', 'API ERROR'), ` ${who} (${str(e.purpose)}${e.kind ? ', ' + str(e.kind) : ''}) — ${str(e.error)}` +
        (num(e.charged) ? ` — charged ${num(e.charged)}` : ' — not charged') + ' — round continued'];
    case 'answer_invalid':
      return [tag('error', 'BAD ANSWER'), ` ${who}'s answer was unusable (${str(e.code)}) — graded wrong`];
    case 'commentary':
      return [tag('commentary', '📣'), el('em', { text: ' ' + str(e.text) })];
    case 'round_error':
      return [tag('error', 'ROUND ERROR'), ` round ${num(e.round)}: ${str(e.error)} — simulation continued`];
    default:
      return null;
  }
}

function renderTicker(evts) {
  const lines = arr(evts).map(eventLine).filter(Boolean).reverse();
  setChildren($('ticker'), lines.length
    ? lines.map((parts) => el('div', { class: 'ticker-line' }, parts))
    : el('div', { class: 'empty-note', text: 'No transactions yet.' }));
}

function showBanner(n) {
  const b = $('banner');
  if (n.kind === 'bankruptcy') {
    b.textContent = n.epitaph ? `${str(n.agent)} is bankrupt: “${str(n.epitaph)}”` : `${str(n.agent)} just went bankrupt`;
    b.className = 'banner bankruptcy show'; soundBankrupt();
  } else {
    b.textContent = `${str(n.agent)} now controls the entire economy`;
    b.className = 'banner monopoly show'; soundMonopoly();
  }
  setTimeout(() => b.classList.remove('show'), 3200);
}

function renderStatus(d) {
  lastPaused = !!d.paused;
  const label = d.status === 'ended' ? 'ended' : d.status === 'replay' ? 'replay' : d.paused ? 'paused' : d.status === 'idle' ? 'idle' : 'live';
  $('pulse-dot').classList.toggle('paused', label !== 'live');
  $('pulse-label').textContent = label;
  controlsMode = str(d.controls, 'open');
  const sel = $('speed-select');
  if (document.activeElement !== sel) {
    const want = String(num(d.round_delay, 8));
    if ([...sel.options].some((o) => o.value === want)) sel.value = want;
  }
  refreshControlButtons();
}

// ------------------------------------------------------------ round-2 panels
function drawSeries(canvasId, points, lines, opts) {
  const canvas = $(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = canvas.clientHeight || 140;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px IBM Plex Mono, monospace';
  const pts = arr(points).filter((p) => p && typeof p === 'object');
  if (pts.length < 2) {
    ctx.fillStyle = '#9CB3A8'; ctx.fillText('Collecting data…', 8, h / 2);
    return;
  }
  const pad = { l: 34, r: 8, t: 8, b: 16 };
  let hi = opts.max != null ? opts.max : 0;
  if (opts.max == null) pts.forEach((p) => lines.forEach((l) => { hi = Math.max(hi, num(p[l.key])); }));
  hi = hi || 1;
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  ctx.strokeStyle = 'rgba(239,231,216,0.08)'; ctx.fillStyle = '#9CB3A8';
  for (let i = 0; i <= 2; i++) {
    const y = pad.t + ph * i / 2;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const v = hi - hi * i / 2;
    ctx.fillText(opts.pct ? Math.round(v * 100) + '%' : Math.round(v), 2, y + 3);
  }
  lines.forEach((l) => {
    ctx.strokeStyle = l.color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    pts.forEach((p, i) => {
      if (typeof p[l.key] !== 'number' || !Number.isFinite(p[l.key])) return;   // gaps, never fake zeros
      const x = pad.l + pw * i / (pts.length - 1), y = pad.t + ph - (p[l.key] / hi) * ph;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

function renderTrends(d) {
  const pts = arr(d.series);
  drawSeries('chart-accuracy', pts, [{ key: 'accuracy', color: '#C9A24B' }], { max: 1, pct: true });
  drawSeries('chart-econ', pts, [{ key: 'compute', color: '#B5432E' }, { key: 'rewards', color: '#6E8F7C' }], {});
  drawSeries('chart-loans', pts, [{ key: 'loan_volume', color: '#7A8FA6' }], {});
  const sw = (c) => { const x = el('span', { class: 'legend-swatch' }); x.style.background = c; return x; };
  setChildren($('legend-econ'), el('span', { class: 'legend-item' }, sw('#B5432E'), 'compute spent'),
    el('span', { class: 'legend-item' }, sw('#6E8F7C'), 'rewards paid'));
}

function renderAuction(d) {
  const a = d.last_auction ? obj(d.last_auction) : null;
  const rows = a ? arr(a.bids).map((b) => {
    b = obj(b);
    return [str(b.agent) + (b.agent === a.winner ? '  ◀ won' : ''), str(b.strategy, '—'), fmt(b.confidence, 2),
      fmt(b.p_est, 2), b.ev == null ? '—' : (num(b.ev) >= 0 ? '+' : '') + fmt(b.ev, 2)];
  }) : [];
  table($('auction-table'), ['Bidder', 'Strategy', 'Stated confidence', 'Est. success p', 'Expected value'],
    rows, 'No auction has resolved yet.');
  [...$('auction-table').querySelectorAll('tbody tr')].forEach((tr, i) => {
    if (a && arr(a.bids)[i] && arr(a.bids)[i].agent === a.winner) tr.classList.add('winner');
  });
  const decs = arr(d.latest_decisions).map((e) => {
    e = obj(e);
    const changed = e.model_action && e.model_action !== e.action;
    const head = el('span', {}, el('span', { class: 'who', text: str(e.agent) }), ` (${str(e.strategy, '—')}): `,
      changed ? el('span', { class: 'override', text: `model proposed ${str(e.model_action)} → strategy chose ${str(e.action)}` })
        : str(e.action), arr(e.policy_reasons).length ? ` [${arr(e.policy_reasons).map(str).join(', ')}]` : '');
    return el('div', { class: 'decision-item' }, head,
      e.explanation ? el('span', { class: 'why', text: str(e.explanation) }) : null,
      e.model_explanation ? el('span', { class: 'said', text: 'model says: “' + str(e.model_explanation) + '”' }) : null);
  });
  setChildren($('decision-list'), decs.length ? decs : el('div', { class: 'empty-note', text: 'No decisions yet.' }));
}

function renderTiers(d) {
  const t = obj(d.tiers);
  const rows = Object.keys(t).sort().map((k) => {
    const r = obj(t[k]);
    return [k, num(r.solves), num(r.graded), pct(r.accuracy), num(r.compute), num(r.rewards), num(r.net_profit),
      fmt(r.profit_per_solve, 2)];
  });
  table($('tier-table'), ['Tier', 'Solves', 'Graded', 'Accuracy', 'Compute', 'Rewards', 'Net profit', 'Profit / solve'],
    rows, 'No solves yet.');
  const smallest = Math.min(...Object.values(t).map((r) => num(obj(r).graded)), Infinity);
  $('tier-note').textContent = Number.isFinite(smallest)
    ? `Smallest tier sample: n=${smallest}. Differences on samples this small are usually noise, and agents escalate ` +
      'precisely when confidence is low, which confounds the comparison. ' +
      (d.mode === 'mock' ? 'In mock mode accuracy does not depend on tier at all.' : '')
    : '';
}

function renderCalibration(d) {
  const c = obj(obj(d.calibration).self_reported);
  const n = num(c.n);
  $('cal-badge').textContent = n ? `(n=${n}, Brier ${fmt(c.brier, 3)}, ECE ${fmt(c.ece, 3)}${c.reliable ? '' : ', indicative only'})` : '';
  const rows = arr(c.bins).filter((b) => num(obj(b).n) > 0).map((b) => {
    b = obj(b);
    return [`${Math.round(num(b.lo) * 100)}–${Math.round(num(b.hi) * 100)}%`, num(b.n), pct(b.mean_predicted), pct(b.observed_accuracy)];
  });
  table($('cal-table'), ['Stated confidence', 'Results', 'Mean stated', 'Observed accuracy'], rows, 'No graded results yet.');
  $('cal-note').textContent = 'Well-calibrated means the observed accuracy matches the stated confidence in every row. ' +
    'Confidence is measured here, never assumed.' + (c.note ? ' ' + str(c.note) + '.' : '');
}

function renderCost(d) {
  const s = obj(d.stats), u = d.provider_usage ? obj(d.provider_usage) : null;
  const lines = [
    el('div', {}, el('strong', { text: 'Simulated compute-seconds: ' }), `${num(s.compute_burned)} spent, ${num(s.rewards_total)} paid out `,
      '(the economy\'s currency — not money).'),
    u ? el('div', {}, el('strong', { text: 'Provider-reported usage: ' }),
      `${num(u.input_tokens)} input / ${num(u.output_tokens)} output tokens (${num(u.calls_with_usage)}/${num(u.calls)} calls reported usage). `,
      u.estimated_cost_usd == null ? 'Estimated cost: unknown (no prices configured).' : 'Estimated cost: $' + num(u.estimated_cost_usd).toFixed(4) + '.')
      : el('div', {}, d.mode === 'mock' ? 'No provider usage: mock mode makes no provider calls.' : 'No provider usage recorded yet.'),
  ];
  setChildren($('cost-note'), lines);
}

// ----------------------------------------------------------------- experiments
async function loadExperiments() {
  try {
    const res = await fetch('/experiments');
    const list = arr(await res.json());
    const sel = $('exp-select'), current = sel.value;
    const opts = [el('option', { value: '', text: list.length ? '— choose an experiment —' : '— none found (run python -m training.experiment) —' })];
    list.forEach((e) => {
      e = obj(e);
      opts.push(el('option', { value: str(e.name), text: `${str(e.name)} — ${str(e.kind)} ${str(e.param)} (${num(e.trials)} trials, ${str(e.mode)})` }));
    });
    setChildren(sel, opts); sel.value = current;
  } catch (err) { console.error('experiments list failed', err); }
}

function ciText(m) {
  return m && m.ci95 ? `${fmt(m.mean, 2)} (${fmt(m.ci95[0], 1)} to ${fmt(m.ci95[1], 1)})` : (m ? fmt(m.mean, 2) + ' (n=' + num(m.n) + ')' : '—');
}

async function showExperiment(name) {
  if (!name) { setChildren($('exp-table')); setChildren($('exp-analysis')); return; }
  try {
    const res = await fetch('/experiments/' + encodeURIComponent(name));
    const body = obj(await res.json());
    if (!res.ok) throw new Error(str(body.error, 'HTTP ' + res.status));
    const sm = obj(body.summary);
    const group = sm.kind === 'tournament' ? obj(sm.strategies) : obj(sm.conditions);
    const rows = Object.keys(group).map((k) => {
      const g = obj(group[k]);
      const m = sm.kind === 'tournament' ? g : obj(g.metrics);
      return [k, ciText(m.economic_profit), ciText(m.accuracy), ciText(m.compute_spent)];
    });
    table($('exp-table'), [sm.kind === 'tournament' ? 'Strategy' : str(sm.param), 'Econ. profit mean (95% CI)', 'Accuracy mean (95% CI)', 'Compute mean (95% CI)'],
      rows, 'No data.');
    setChildren($('exp-analysis'), str(body.analysis).split('\n').filter((l) => l.trim()).map((l) => el('p', { text: l })));
  } catch (err) {
    setChildren($('exp-analysis'), el('p', { text: 'Could not load experiment: ' + str(err.message) }));
  }
}
$('exp-select').addEventListener('change', (e) => showExperiment(e.target.value));

// ----------------------------------------------------------------- polling
let pollTimer = null;
async function poll(immediate) {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  try {
    const url = replayMode ? '/state?replay=' + encodeURIComponent($('replay-select').value) : '/state';
    const res = await fetch(url);
    const d = obj(await res.json());
    if (!res.ok) throw new Error(str(d.error, 'HTTP ' + res.status));
    failedPolls = 0; $('conn-status').textContent = '';
    renderMode(d); renderStatus(d);
    $('round-num').textContent = num(d.round_count);
    renderLedger(d); renderChart(d.history, d.starting_balance); renderStats(d);
    renderAgents(d); renderLoans(d); renderCommentary(d.latest_commentary);
    renderTrends(d); renderAuction(d); renderTiers(d); renderCalibration(d); renderCost(d);
    currentRunId = replayMode ? null : (typeof d.run_id === 'string' ? d.run_id : null);
    renderTask(d.latest_task); renderTicker(d.recent_events);

    if (!replayMode) {
      const evts = arr(d.recent_events);
      if (lastSeenTs !== 0) evts.filter((e) => num(e.ts) > lastSeenTs)
        .forEach((e) => { if (e.type === 'task_resolved' && e.correct) soundSolved(); });
      if (evts.length) lastSeenTs = Math.max(...evts.map((e) => num(e.ts)));
      if (d.last_notable) {
        const n = obj(d.last_notable);
        const key = str(n.kind) + str(n.agent) + str(n.round);
        if (!notableInitialized) { lastNotableKey = key; notableInitialized = true; }
        else if (key !== lastNotableKey) { lastNotableKey = key; showBanner(n); }
      }
    }
  } catch (err) {
    failedPolls += 1;
    $('conn-status').textContent = failedPolls > 1 ? 'Cannot reach the server or read its state — retrying…' : '';
    console.error('poll failed', err);
  }
  pollTimer = setTimeout(poll, 1500);
}

$('leaderboard-sort').addEventListener('change', renderLeaderboard);
window.addEventListener('resize', () => poll());
loadReplayList(); loadExperiments(); renderLeaderboard(); updateExportLinks(); poll();
setInterval(loadReplayList, 10000);
setInterval(loadExperiments, 30000);
setInterval(renderLeaderboard, 8000);
