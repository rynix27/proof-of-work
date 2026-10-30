// Loads the REAL dashboard.html + dashboard.js in jsdom (scripts enabled) and feeds it hostile API data.
// Exits non-zero if any payload executes or is parsed into markup.
const { JSDOM } = require(process.env.JSDOM_PATH || "jsdom");
const fs = require("fs"), path = require("path");
const web = process.env.WEB_DIR || path.resolve(__dirname, "../../web");
const html = fs.readFileSync(path.join(web, "dashboard.html"), "utf8");
const js = fs.readFileSync(path.join(web, "dashboard.js"), "utf8");

const P = (tag) => `<img src=x onerror="window.__pwned='${tag}'"><script>window.__pwned='${tag}-s'</script>`;
const state = {
  replay: false, mode: "live", status: "running", paused: false, controls: "open", round_delay: 8,
  models: { nano: P("nano"), ultra: P("ultra") }, starting_balance: 100, round_count: 3,
  balances: { [P("agent")]: 120, Normal: 80 }, bankrupt: [], tasks_won: { [P("agent")]: 2 },
  agents: { [P("agent")]: { balance: 120, net_pl: 20, accuracy: 0.5, attempts: 2, compute_spent: 3, rewards: 8,
    efficiency: 1, escalations: 0, invalid_decisions: 0, bankrupt: false, strategy: P("agent-strategy"),
    economic_profit: 5, escalation_rate: 0.1, loans_issued: 1, outstanding_debt: 2 } },
  history: [{ round: 1, balances: { [P("agent")]: 100 } }],
  latest_task: { prompt: P("task"), difficulty: P("diff"), bounty: 8, injected: true },
  latest_commentary: { text: P("commentary") },
  loans: [{ id: P("loanid"), lender: P("lender"), borrower: P("borrower"), principal: 5, total_owed: 6, repaid: 0,
    outstanding: 6, status: P("status"), due_round: 4 }],
  run_id: P("run-id"),
  series: [{ round: 1, accuracy: 0.5, compute: 3, rewards: 8, loan_volume: 5 }, { round: 2, accuracy: null, compute: 6, rewards: 8, loan_volume: 9 }],
  last_auction: { winner: P("winner"), bids: [{ agent: P("winner"), strategy: P("strat"), confidence: 0.8, p_est: 0.7, ev: 4.6 },
                                              { agent: P("loser"), strategy: P("strat2"), confidence: 0.4, p_est: 0.3, ev: null }] },
  latest_decisions: [{ agent: P("dec-agent"), strategy: P("dec-strat"), action: P("act"), model_action: P("model-act"),
                       policy_reasons: [P("reason")], explanation: P("explanation"), model_explanation: P("model-says") }],
  tiers: { [P("tier")]: { solves: 3, graded: 3, accuracy: 0.66, compute: 3, rewards: 16, net_profit: 13, profit_per_solve: 4.33 } },
  calibration: { self_reported: { n: 3, brier: 0.2, ece: 0.1, reliable: false, note: P("cal-note"),
                                  bins: [{ lo: 0.8, hi: 1.0, n: 3, mean_predicted: 0.9, observed_accuracy: 0.66 }] } },
  provider_usage: { calls: 4, calls_with_usage: 3, input_tokens: 100, output_tokens: 20, estimated_cost_usd: null },
  sim_config: { interest_rate: 0.15, loan_term_rounds: 5 }, gini: 0.1,
  stats: { tasks_total: 3 }, pending_questions: 0, last_notable: null,
  recent_events: [
    { type: "task_resolved", agent: P("ev-agent"), correct: true, tier: P("tier"), bounty: 8, cost: 1, new_balance: 9, ts: 1 },
    { type: "task_resolved", agent: "a", exhibition: true, answer: P("answer"), ts: 2 },
    { type: "bankruptcy", agent: P("b"), epitaph: P("epitaph"), ts: 3 },
    { type: "call_error", agent: "a", purpose: "solve", kind: P("kind"), error: P("error"), charged: 1, ts: 4 },
    { type: "commentary", text: P("c2"), ts: 5 },
    { type: "loan_made", lender: P("l"), borrower: P("b"), amount: 5, ts: 6 },
    { type: "round_error", round: 1, error: P("round-error"), ts: 7 },
  ],
};
const leaderboard = { total_runs: 1, standings: [{ agent: P("lb"), wins: 1, runs: 1, win_rate: 1, avg_final_balance: 100 }] };
const experiments = [{ name: P("exp-name"), kind: P("exp-kind"), param: P("exp-param"), trials: 5, mode: P("exp-mode") }];
const experimentDetail = { summary: { kind: "sweep", param: P("sweep-param"), conditions: { [P("cond")]: { n: 5, metrics: {
  economic_profit: { n: 5, mean: 10, ci95: [1, 19] }, accuracy: { n: 5, mean: 0.6, ci95: null }, compute_spent: null } } } },
  analysis: "line one " + P("analysis") + "\nline two" };
const replays = [{ filename: P("replay-file"), event_count: 3, summary: P("summary") }];

const dom = new JSDOM(html.replace(/<script src="\/dashboard.js" defer><\/script>/, ""), {
  runScripts: "dangerously", url: "http://localhost/", pretendToBeVisual: true,
  beforeParse(w) {
    w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: () => () => {}, set: () => true });
    w.fetch = async (u) => {
      const body = u.startsWith("/state") ? state : u.startsWith("/leaderboard") ? leaderboard
        : u.startsWith("/experiments/") ? experimentDetail : u.startsWith("/experiments") ? experiments : replays;
      return { ok: true, status: 200, json: async () => body };
    };
    w.confirm = () => true; w.prompt = () => "x";
  },
});
const w = dom.window;
w.eval(js);
setTimeout(() => {
  const sel = w.document.getElementById("exp-select");
  sel.value = sel.options[1] ? sel.options[1].value : "";
  sel.dispatchEvent(new w.Event("change"));
}, 250);
setTimeout(() => {
  const doc = w.document;
  const injected = [...doc.querySelectorAll("img, script, iframe, object, embed, svg, [onerror], [onclick]")]
    .filter((n) => !(n.tagName === "SCRIPT" && n.getAttribute("src") === "/dashboard.js"));
  const text = doc.body.textContent;
  const checks = {
    "no payload executed": w.__pwned === undefined,
    "no injected elements": injected.length === 0,
    "ledger rendered": doc.querySelectorAll("#ledger .agent-row").length >= 1,
    "stats rendered": doc.querySelectorAll("#stats .stat-cell").length > 0,
    "payload shown as inert text": text.includes("<img src=x onerror="),
    "ticker rendered": doc.querySelectorAll("#ticker .ticker-line").length >= 5,
    "loans rendered": doc.querySelectorAll("#loan-table tbody tr").length === 1,
    "leaderboard rendered": doc.querySelectorAll("#leaderboard-table tbody tr").length === 1,
    "replay option rendered": doc.querySelectorAll("#replay-select option").length === 2,
    "auction rendered (2 bids)": doc.querySelectorAll("#auction-table tbody tr").length === 2,
    "winner row highlighted": doc.querySelectorAll("#auction-table tr.winner").length >= 1,
    "decision explanation rendered": doc.querySelectorAll("#decision-list .decision-item").length === 1,
    "model words labelled as the model's": doc.querySelector("#decision-list .said").textContent.startsWith("model says:"),
    "tier table rendered": doc.querySelectorAll("#tier-table tbody tr").length === 1,
    "calibration rendered": doc.querySelectorAll("#cal-table tbody tr").length === 1,
    "cost panel rendered": doc.querySelectorAll("#cost-note > div").length === 2,
    "unknown cost shown as unknown, not $0": doc.getElementById("cost-note").textContent.includes("unknown") && !doc.getElementById("cost-note").textContent.includes("$0"),
    "experiments option rendered": doc.querySelectorAll("#exp-select option").length === 2,
    "experiment table rendered": doc.querySelectorAll("#exp-table tbody tr").length === 1,
    "experiment analysis rendered": doc.querySelectorAll("#exp-analysis p").length === 2,
    "agent strategy column rendered": doc.querySelector("#agent-table thead").textContent.includes("Strategy"),
    "charts present": ["chart-accuracy", "chart-econ", "chart-loans"].every((id) => doc.getElementById(id)),
  };
  let bad = 0;
  for (const [k, v] of Object.entries(checks)) { console.log((v ? "PASS " : "FAIL ") + k); if (!v) bad++; }
  if (injected.length) console.log("injected:", injected.map((n) => n.outerHTML.slice(0, 80)));
  process.exit(bad ? 1 : 0);
}, 400);
