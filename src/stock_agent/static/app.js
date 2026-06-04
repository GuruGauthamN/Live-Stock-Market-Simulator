const el = (id) => document.getElementById(id);

const state = {
  selected: null,
  symbols: [],
  ticks: [],
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed: ${res.status}`);
  return data;
}

function fmtNum(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function pnlColor(value) {
  if (value > 0) return "var(--up)";
  if (value < 0) return "var(--down)";
  return "var(--warn)";
}

function setHealth(health) {
  el("health-pill").textContent = `Status: ${health.status}`;
  el("model-pill").textContent = `Model: ${health.model_loaded ? "loaded" : "not trained"}`;
}

function setSymbols(symbols) {
  const select = el("symbol-select");
  const previous = state.selected;
  select.innerHTML = "";
  symbols.forEach((symbol) => {
    const opt = document.createElement("option");
    opt.value = symbol;
    opt.textContent = symbol;
    select.appendChild(opt);
  });
  state.symbols = symbols;
  state.selected = previous && symbols.includes(previous) ? previous : symbols[0] || null;
  if (state.selected) select.value = state.selected;
}

function renderChart(ticks) {
  const svg = el("live-chart");
  if (!ticks || ticks.length < 2) {
    svg.innerHTML = "";
    el("chart-meta").textContent = "Waiting for ticks...";
    return;
  }

  const width = 1000;
  const height = 360;
  const pad = 20;
  const prices = ticks.map((t) => Number(t.price));
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const span = Math.max(max - min, 0.000001);

  const points = prices.map((price, i) => {
    const x = (i / (prices.length - 1)) * (width - pad * 2) + pad;
    const y = height - pad - ((price - min) / span) * (height - pad * 2);
    return [x, y];
  });

  const line = points.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
  const area = `${pad},${height - pad} ${line} ${width - pad},${height - pad}`;
  const start = prices[0];
  const end = prices[prices.length - 1];
  const up = end >= start;
  const stroke = up ? "#41d38a" : "#ff6f7f";
  const fill = up ? "rgba(65, 211, 138, 0.16)" : "rgba(255, 111, 127, 0.14)";

  const yLines = [0.2, 0.4, 0.6, 0.8]
    .map((r) => {
      const y = (height - pad * 2) * r + pad;
      return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
    })
    .join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="rgba(9,14,22,0.98)" />
    ${yLines}
    <polygon points="${area}" fill="${fill}" />
    <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="3" />
    <circle cx="${points[points.length - 1][0]}" cy="${points[points.length - 1][1]}" r="4.5" fill="${stroke}" />
  `;

  const change = end - start;
  const pct = (change / start) * 100;
  el("last-price").textContent = fmtNum(end, 4);
  el("last-price").style.color = pnlColor(change);
  el("chart-meta").textContent = `${state.selected} | ${ticks.length} ticks | range ${fmtNum(min, 4)} - ${fmtNum(max, 4)} | change ${fmtNum(change, 4)} (${fmtNum(pct, 2)}%)`;
}

function renderSignal(signal) {
  const action = (signal.action || "hold").toLowerCase();
  const card = el("signal-card");
  card.className = `signal ${action}`;
  el("signal-action").textContent = action.toUpperCase();
  el("signal-reason").textContent = signal.reason || "-";
  el("signal-meta").textContent = `confidence: ${fmtNum(signal.confidence ?? 0, 4)} | expected_return: ${fmtNum(signal.expected_return ?? 0, 6)}`;
}

function renderPortfolio(p) {
  el("p-cash").textContent = fmtNum(p.cash);
  el("p-equity").textContent = fmtNum(p.equity);
  el("p-realized").textContent = fmtNum(p.realized_pnl);
  el("p-unrealized").textContent = fmtNum(p.unrealized_pnl);
  el("p-total").textContent = fmtNum(p.total_pnl);
  el("p-daily").textContent = fmtNum(p.daily_pnl);

  ["p-realized", "p-unrealized", "p-total", "p-daily"].forEach((id) => {
    const value = Number(el(id).textContent);
    el(id).style.color = pnlColor(value);
  });

  el("positions").textContent = JSON.stringify(p.positions || {}, null, 2);
}

function renderBacktest(bt) {
  el("m-accuracy").textContent = `${fmtNum(bt.accuracy * 100, 2)}%`;
  el("m-pnl").textContent = fmtNum(bt.total_pnl, 4);
  el("m-pnl").style.color = pnlColor(Number(bt.total_pnl));
  el("m-trades").textContent = `${bt.trades}`;
  el("m-winloss").textContent = `${bt.winning_trades}/${bt.losing_trades}`;
  el("backtest-log").textContent = JSON.stringify(bt, null, 2);
}

function setError(target, err) {
  el(target).textContent = `Error: ${err.message || err}`;
}

async function refreshTicks() {
  if (!state.selected) return;
  const data = await api(`/ticks/${state.selected}?seconds=300`);
  state.ticks = data.ticks || [];
  renderChart(state.ticks);
}

async function refreshNews() {
  if (!state.selected) return;
  const news = await api(`/news/${state.selected}?minutes=180&limit=20`);
  el("n-sentiment").textContent = fmtNum(news.sentiment ?? 0, 4);
  el("n-sentiment").style.color = pnlColor(Number(news.sentiment || 0));
  el("news-log").textContent = JSON.stringify(news.items || [], null, 2);
}

async function refreshLlmStatus() {
  const status = await api("/llm/agent/status");
  el("llm-enabled").textContent = status.enabled ? "YES" : "NO";
  el("llm-auto").textContent = status.auto_run ? "YES" : "NO";
  el("llm-model").textContent = status.model || "-";
  el("llm-pnl").textContent = fmtNum(status.portfolio_total_pnl || 0, 2);
  el("llm-pnl").style.color = pnlColor(Number(status.portfolio_total_pnl || 0));
}

async function refreshPortfolio() {
  const p = await api("/portfolio");
  renderPortfolio(p);
}

async function refreshSignal() {
  if (!state.selected) return;
  const signal = await api(`/signal/${state.selected}`);
  renderSignal(signal);
}

async function runBacktest() {
  const ratio = Number(el("test-ratio").value || 0.2);
  el("backtest-log").textContent = "Running backtest...";
  const bt = await api(`/backtest?test_ratio=${ratio}`, { method: "POST" });
  renderBacktest(bt);
}

async function runDecision() {
  if (!state.selected) return;
  const horizon = Number(el("decision-horizon").value || 30);
  const decision = await api(`/decision/${state.selected}?horizon_minutes=${horizon}`, {
    method: "POST",
  });
  el("d-invest").textContent = decision.invest ? "YES" : "NO";
  el("d-invest").style.color = decision.invest ? "var(--up)" : "var(--down)";
  el("d-confidence").textContent = fmtNum(decision.confidence, 4);
  el("d-exp").textContent = fmtNum(decision.expected_return, 6);
  el("trade-log").textContent = JSON.stringify(decision, null, 2);
  await refreshPortfolio();
}

async function runLlmAgent() {
  const horizon = Number(el("decision-horizon").value || 30);
  const result = await api(`/llm/agent/run?horizon_minutes=${horizon}`, {
    method: "POST",
  });
  el("news-log").textContent = JSON.stringify(result, null, 2);
  await Promise.all([refreshPortfolio(), refreshLlmStatus(), evaluateDecisions()]);
}

async function evaluateDecisions() {
  const perf = await api("/decision/performance");
  el("perf-resolved").textContent = `${perf.resolved}/${perf.total}`;
  el("perf-accuracy").textContent = `${fmtNum((perf.accuracy || 0) * 100, 2)}%`;
  el("perf-avgret").textContent = `${fmtNum((perf.avg_return_pct || 0) * 100, 3)}%`;
  el("perf-pnl").textContent = fmtNum(perf.total_hypothetical_pnl || 0, 2);
  el("perf-pnl").style.color = pnlColor(Number(perf.total_hypothetical_pnl || 0));
  el("perf-updated").textContent = `${perf.updated}`;
  el("perf-log").textContent = JSON.stringify(perf.recent || [], null, 2);
}

async function placeTrade(action) {
  if (!state.selected) return;
  const qty = Number(el("trade-qty").value || 1);
  const trade = await api(`/trade/${state.selected}?action=${action}&qty=${qty}`, { method: "POST" });
  el("trade-log").textContent = JSON.stringify(trade, null, 2);
  await refreshPortfolio();
}

async function fullRefresh() {
  try {
    const [health, symbolsRes] = await Promise.all([api("/health"), api("/symbols")]);
    setHealth(health);
    setSymbols(symbolsRes.symbols || []);
    await Promise.all([
      refreshTicks(),
      refreshSignal(),
      refreshPortfolio(),
      refreshNews(),
      refreshLlmStatus(),
    ]);
  } catch (err) {
    setError("trade-log", err);
  }
}

el("symbol-select").addEventListener("change", async (event) => {
  state.selected = event.target.value;
  try {
    await Promise.all([refreshTicks(), refreshSignal(), refreshNews()]);
  } catch (err) {
    setError("chart-meta", err);
  }
});

el("refresh-symbols").addEventListener("click", async () => {
  try {
    const symbolsRes = await api("/symbols");
    setSymbols(symbolsRes.symbols || []);
    await Promise.all([refreshTicks(), refreshSignal(), refreshNews()]);
  } catch (err) {
    setError("chart-meta", err);
  }
});

el("btn-news").addEventListener("click", () => refreshNews().catch((err) => setError("news-log", err)));
el("btn-decision").addEventListener("click", () => runDecision().catch((err) => setError("trade-log", err)));
el("btn-llm-run").addEventListener("click", () => runLlmAgent().catch((err) => setError("news-log", err)));
el("btn-performance").addEventListener("click", () => evaluateDecisions().catch((err) => setError("perf-log", err)));
el("btn-signal").addEventListener("click", () => refreshSignal().catch((err) => setError("signal-reason", err)));
el("btn-buy").addEventListener("click", () => placeTrade("buy").catch((err) => setError("trade-log", err)));
el("btn-sell").addEventListener("click", () => placeTrade("sell").catch((err) => setError("trade-log", err)));
el("btn-backtest").addEventListener("click", () => runBacktest().catch((err) => setError("backtest-log", err)));

fullRefresh();

setInterval(() => {
  refreshTicks().catch((err) => setError("chart-meta", err));
  refreshPortfolio().catch((err) => setError("positions", err));
}, 2000);

setInterval(() => {
  refreshSignal().catch(() => null);
  refreshNews().catch(() => null);
  refreshLlmStatus().catch(() => null);
  evaluateDecisions().catch(() => null);
}, 10000);
