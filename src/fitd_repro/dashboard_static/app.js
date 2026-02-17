const CONDITION_DEFS = [
  { label: "Standard", attack: "standard", defense: "none" },
  { label: "FITD", attack: "fitd", defense: "none" },
  { label: "FITD + Vigilant", attack: "fitd", defense: "vigilant" },
];

const pollMs = 1800;
const traceLimit = 100;

const els = {
  serverTime: document.getElementById("serverTime"),
  activeBatch: document.getElementById("activeBatch"),
  modelsSeen: document.getElementById("modelsSeen"),
  overallStatus: document.getElementById("overallStatus"),
  overallBar: document.getElementById("overallBar"),
  runList: document.getElementById("runList"),
  matrixGrid: document.getElementById("matrixGrid"),
  effectCards: document.getElementById("effectCards"),
  effectTurns: document.getElementById("effectTurns"),
  transitionWrap: document.getElementById("transitionWrap"),
  transitionBody: document.getElementById("transitionBody"),
  activityLog: document.getElementById("activityLog"),
  modelStats: document.getElementById("modelStats"),
  form: document.getElementById("runForm"),
  startButton: document.getElementById("startButton"),
  formMessage: document.getElementById("formMessage"),
  backendSelect: document.getElementById("backend"),
  backendPolicy: document.getElementById("backendPolicy"),
  fitdVariant: document.getElementById("fitdVariant"),
  authorPromptTrack: document.getElementById("authorPromptTrack"),
  authorPromptFile: document.getElementById("authorPromptFile"),
  authorMaxWarmups: document.getElementById("authorMaxWarmups"),
  traceRun: document.getElementById("traceRun"),
  traceFilter: document.getElementById("traceFilter"),
  traceReveal: document.getElementById("traceReveal"),
  traceMeta: document.getElementById("traceMeta"),
  traceList: document.getElementById("traceList"),
};

const traceState = {
  selectedRunId: "",
  includeResponse: false,
  filterMode: "all",
};

let tickInFlight = false;

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fileName(pathValue) {
  if (!pathValue) return "-";
  const normalized = String(pathValue).replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function formatProgressValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value || "0");
  if (Math.abs(numeric - Math.round(numeric)) < 0.001) return String(Math.round(numeric));
  return numeric.toFixed(2);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function isoToLocal(isoText) {
  if (!isoText) return "-";
  const d = new Date(isoText);
  if (Number.isNaN(d.getTime())) return isoText;
  return d.toLocaleTimeString();
}

function setFormMessage(text, isError = false) {
  els.formMessage.textContent = text;
  els.formMessage.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function updateAuthorInputs() {
  const authorMode = els.fitdVariant.value === "author";
  els.authorPromptTrack.disabled = !authorMode;
  els.authorPromptFile.disabled = !authorMode;
  els.authorMaxWarmups.disabled = !authorMode;

  const datasetInput = document.getElementById("datasetPath");
  const datasetPath = datasetInput.value.trim().toLowerCase();
  const hasAuthorDatasetHint =
    datasetPath.includes("jailbreakbench") || datasetPath.includes("harmbench");

  if (authorMode) {
    if (!hasAuthorDatasetHint || datasetPath.endsWith("sample_prompts.csv")) {
      datasetInput.value = "data/author_fitd/jailbreakbench.csv";
    }
    if (!els.authorPromptFile.value.trim()) {
      els.authorPromptFile.value = "data/author_fitd/prompt_jailbreakbench.json";
    }
    if (!els.authorMaxWarmups.value.trim()) {
      els.authorMaxWarmups.value = "4";
    }
  }
}

function findLatestBatch(snapshot) {
  if (!snapshot || !Array.isArray(snapshot.batches) || snapshot.batches.length === 0) return null;
  const active = snapshot.batches.find((batch) => batch.batch_id === snapshot.active_batch_id);
  return active || snapshot.batches[0];
}

function findRecentSummary(snapshot, condition) {
  if (!snapshot || !Array.isArray(snapshot.recent_summaries)) return null;
  return snapshot.recent_summaries.find(
    (row) => row.attack === condition.attack && row.defense === condition.defense
  );
}

function renderCapabilities(snapshot) {
  const openaiAllowed = Boolean(snapshot?.capabilities?.openai_backend_allowed);
  const openaiOption = els.backendSelect.querySelector("option[value='openai']");
  if (openaiOption) {
    openaiOption.disabled = !openaiAllowed;
  }

  if (!openaiAllowed) {
    if (els.backendSelect.value === "openai") {
      els.backendSelect.value = "hf";
    }
    els.backendPolicy.textContent =
      "OpenAI backend locked by default. Use hf/mock. To opt in, relaunch with FITD_ALLOW_OPENAI=1.";
    return;
  }

  els.backendPolicy.textContent =
    "OpenAI backend enabled. Use only approved policy-compliant prompts for this project.";
}

function renderProgress(batch) {
  if (!batch) {
    els.overallStatus.textContent = "No batch has started yet.";
    els.overallBar.style.width = "0%";
    els.runList.innerHTML = "<div class='run-card'>Start a matrix run to see progress.</div>";
    return;
  }

  const totalRuns = batch.runs.length;
  const completedRuns = batch.runs.filter((run) => run.status === "completed").length;
  const runningRun = batch.runs.find((run) => run.status === "running");

  let overall = totalRuns ? completedRuns / totalRuns : 0;
  if (runningRun && runningRun.progress_total > 0) {
    const runFraction = runningRun.progress_completed / runningRun.progress_total;
    overall = (completedRuns + runFraction) / totalRuns;
  }

  els.overallBar.style.width = `${Math.max(0, Math.min(1, overall)) * 100}%`;
  els.overallStatus.textContent =
    `Batch ${batch.batch_id} | ${batch.status.toUpperCase()} | ${completedRuns}/${totalRuns} conditions complete`;

  els.runList.innerHTML = batch.runs
    .map((run) => {
      const ratio =
        run.progress_total > 0 ? Math.max(0, Math.min(1, run.progress_completed / run.progress_total)) : 0;
      const phase = run.phase || run.status;
      const latest = run.latest_turn || null;
      let latestLine = "";
      if (latest && latest.turn_kind) {
        const state =
          latest.turn_status === "generating"
            ? "generating..."
            : latest.assistant_refusal === true
              ? "refusal"
              : latest.assistant_refusal === false
                ? "non-refusal"
                : "pending";
        latestLine = `<div class="run-meta">Latest: ${escapeHtml(latest.turn_kind)}-${escapeHtml(latest.turn_index)} (${state})</div>`;
      }

      const capLabel =
        run.author_max_warmup_turns && Number(run.author_max_warmup_turns) > 0
          ? String(run.author_max_warmup_turns)
          : "full";
      const variantDetail =
        run.fitd_variant === "author"
          ? `${run.author_prompt_track || "prompts1"}, cap=${capLabel}`
          : `${run.author_prompt_track || "prompts1"}`;

      return `
        <article class="run-card">
          <div class="run-head">
            <strong>${run.label}</strong>
            <span class="state ${run.status}">${run.status}</span>
          </div>
          <div class="run-meta">${run.backend}:${run.model}</div>
          <div class="run-meta">FITD variant: ${run.fitd_variant || "scaffold"} (${variantDetail})</div>
          <div class="run-meta">Phase: ${phase} | ${formatProgressValue(run.progress_completed)}/${run.progress_total || "?"}</div>
          ${latestLine}
          <div class="run-meta">Log: ${fileName(run.log_path)}</div>
          <div class="progress-bar mini-progress">
            <div class="progress-fill" style="width:${ratio * 100}%"></div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderMatrix(snapshot, batch) {
  const rows = CONDITION_DEFS.map((condition) => {
    const run = batch
      ? batch.runs.find((entry) => entry.attack === condition.attack && entry.defense === condition.defense)
      : null;
    const recent = !run ? findRecentSummary(snapshot, condition) : null;

    const source = run || recent;
    const status = run ? run.status : recent ? "history" : "pending";
    const asr = source && source.asr !== null && source.asr !== undefined ? source.asr : null;
    const refusal = source && source.refusal_rate !== null && source.refusal_rate !== undefined ? source.refusal_rate : null;
    const total = source ? source.total_examples || source.max_examples || "-" : "-";

    let className = "matrix-card";
    if (status === "failed") className += " failed";
    if (status === "completed" || status === "history") className += " success";

    const statusLabel = run ? `${run.status.toUpperCase()} (${run.phase})` : recent ? "HISTORY" : "PENDING";
    const modelLabel = source ? `${source.backend}:${source.model}` : "-";
    const variantLabel = source ? `${source.fitd_variant || "scaffold"}:${source.author_prompt_track || "prompts1"}` : "-";

    return `
      <article class="${className}">
        <h3 class="matrix-title">${condition.label}</h3>
        <p class="matrix-condition">${condition.attack} / ${condition.defense}</p>
        <div class="metric"><span>Status</span><strong>${statusLabel}</strong></div>
        <div class="metric"><span>ASR</span><strong>${pct(asr)}</strong></div>
        <div class="metric"><span>Refusal</span><strong>${pct(refusal)}</strong></div>
        <div class="metric"><span>Examples</span><strong>${total}</strong></div>
        <div class="metric"><span>Model</span><strong>${modelLabel}</strong></div>
        <div class="metric"><span>FITD Variant</span><strong>${variantLabel}</strong></div>
      </article>
    `;
  });

  els.matrixGrid.innerHTML = rows.join("");
}

function renderEffect(batch) {
  const effect = batch && batch.effect ? batch.effect : null;
  if (!effect || !effect.available) {
    els.transitionWrap.classList.remove("limit-rows");
    els.effectCards.innerHTML =
      "<article class='effect-card'><h3>Awaiting Data</h3><p>Run the three conditions to compute FITD conversion metrics.</p></article>";
    els.effectTurns.innerHTML = "";
    els.transitionBody.innerHTML =
      "<tr><td colspan='7'>No transition rows yet.</td></tr>";
    return;
  }

  const doorOpened = Number(effect.door_opened_count || 0);
  const standardRefusals = Number(effect.standard_refusals || 0);
  const defenseRecovered = Number(effect.defense_recovered_count || 0);
  const stillCompromised = Number(effect.still_compromised_count || 0);

  els.effectCards.innerHTML = `
    <article class="effect-card accent warn">
      <h3>Door Opened</h3>
      <strong>${doorOpened}</strong>
      <p>Std refusal -> FITD success</p>
    </article>
    <article class="effect-card">
      <h3>Baseline Refusals</h3>
      <strong>${standardRefusals}</strong>
      <p>Compared Standard/FITD set</p>
    </article>
    <article class="effect-card ok">
      <h3>Recovered By Vigilant</h3>
      <strong>${defenseRecovered}</strong>
      <p>FITD success -> Vigilant refusal</p>
    </article>
    <article class="effect-card danger">
      <h3>Still Compromised</h3>
      <strong>${stillCompromised}</strong>
      <p>FITD success persists with defense</p>
    </article>
    <article class="effect-card">
      <h3>Door Open Rate</h3>
      <strong>${pct(effect.door_opened_rate_over_standard_refusals)}</strong>
      <p>Over standard refusals</p>
    </article>
    <article class="effect-card">
      <h3>Defense Recovery Rate</h3>
      <strong>${pct(effect.defense_recovered_rate_over_fitd_successes)}</strong>
      <p>Over FITD successes</p>
    </article>
    <article class="effect-card">
      <h3>Std vs FITD Overlap</h3>
      <strong>${effect.comparisons_standard_fitd || 0}</strong>
      <p>Examples compared directly</p>
    </article>
    <article class="effect-card">
      <h3>FITD vs Vigilant Overlap</h3>
      <strong>${effect.comparisons_fitd_vigilant || 0}</strong>
      <p>Examples compared directly</p>
    </article>
  `;

  const turnRows = Array.isArray(effect.first_non_refusal_histogram)
    ? effect.first_non_refusal_histogram
    : [];
  if (!turnRows.length) {
    els.effectTurns.innerHTML = "<span class='turn-pill'>No non-refusal turns observed yet.</span>";
  } else {
    els.effectTurns.innerHTML = turnRows
      .map(
        (row) =>
          `<span class="turn-pill"><strong>${escapeHtml(row.label)}</strong> : ${Number(row.count || 0)}</span>`
      )
      .join("");
  }

  const transitions = Array.isArray(effect.transitions) ? effect.transitions : [];
  if (!transitions.length) {
    els.transitionWrap.classList.remove("limit-rows");
    els.transitionBody.innerHTML = "<tr><td colspan='7'>No transition rows yet.</td></tr>";
    return;
  }

  if (transitions.length > 8) {
    els.transitionWrap.classList.add("limit-rows");
  } else {
    els.transitionWrap.classList.remove("limit-rows");
  }

  els.transitionBody.innerHTML = transitions
    .slice(0, 80)
    .map((row) => {
      const standardStatus = (row.standard || "NA").toLowerCase();
      const fitdStatus = (row.fitd || "NA").toLowerCase();
      const vigilantStatus = (row.vigilant || "NA").toLowerCase();

      let transition = "<span class='transition-badge pending'>No shift</span>";
      if (row.door_opened && row.recovered_by_defense) {
        transition = "<span class='transition-badge recovered'>Opened + Recovered</span>";
      } else if (row.door_opened) {
        transition = "<span class='transition-badge opened'>Door Opened</span>";
      } else if (row.recovered_by_defense) {
        transition = "<span class='transition-badge recovered'>Recovered</span>";
      }

      return `
        <tr>
          <td>${escapeHtml(row.example_id)}</td>
          <td class="goal" title="${escapeHtml(row.goal || "")}">${escapeHtml(row.goal || "-")}</td>
          <td><span class="status-badge ${standardStatus}">${escapeHtml(row.standard || "NA")}</span></td>
          <td><span class="status-badge ${fitdStatus}">${escapeHtml(row.fitd || "NA")}</span></td>
          <td><span class="status-badge ${vigilantStatus}">${escapeHtml(row.vigilant || "NA")}</span></td>
          <td>${escapeHtml(row.first_non_refusal_turn || "none")}</td>
          <td>${transition}</td>
        </tr>
      `;
    })
    .join("");
}

function renderActivity(snapshot) {
  const entries = Array.isArray(snapshot.activity) ? snapshot.activity.slice(-35).reverse() : [];
  if (!entries.length) {
    els.activityLog.innerHTML = "<div class='activity-item'><p>No events yet.</p></div>";
    return;
  }

  els.activityLog.innerHTML = entries
    .map(
      (entry) => `
        <article class="activity-item">
          <time>${isoToLocal(entry.timestamp_utc)}</time>
          <p>${entry.message}</p>
        </article>
      `
    )
    .join("");
}

function renderModelStats(snapshot) {
  const rows = Array.isArray(snapshot.recent_summaries) ? snapshot.recent_summaries : [];
  const groups = new Map();

  for (const row of rows) {
    const key = `${row.backend}:${row.model}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        backend: row.backend,
        model: row.model,
        runs: 0,
        asrSum: 0,
        asrCount: 0,
      });
    }
    const target = groups.get(key);
    target.runs += 1;
    if (typeof row.asr === "number") {
      target.asrSum += row.asr;
      target.asrCount += 1;
    }
  }

  const sorted = Array.from(groups.values()).sort((a, b) => b.runs - a.runs);
  if (!sorted.length) {
    els.modelStats.innerHTML = "No summaries yet.";
    return;
  }

  els.modelStats.innerHTML = `
    <table class="stats-table">
      <thead>
        <tr>
          <th>Backend</th>
          <th>Model</th>
          <th>Runs</th>
          <th>Avg ASR</th>
        </tr>
      </thead>
      <tbody>
        ${sorted
          .map((entry) => {
            const avgAsr = entry.asrCount ? entry.asrSum / entry.asrCount : null;
            return `
              <tr>
                <td>${entry.backend}</td>
                <td>${entry.model}</td>
                <td>${entry.runs}</td>
                <td>${pct(avgAsr)}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function updateTraceRunOptions(batch) {
  const runs = batch && Array.isArray(batch.runs) ? batch.runs : [];
  if (!runs.length) {
    traceState.selectedRunId = "";
    els.traceRun.innerHTML = "";
    els.traceRun.disabled = true;
    return [];
  }

  const options = runs.map(
    (run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.label)} | ${escapeHtml(run.attack)}/${escapeHtml(run.defense)} | ${escapeHtml(run.status)}</option>`
  );
  els.traceRun.innerHTML = options.join("");
  els.traceRun.disabled = false;

  const runIds = new Set(runs.map((run) => run.run_id));
  if (!traceState.selectedRunId || !runIds.has(traceState.selectedRunId)) {
    const running = runs.find((run) => run.status === "running");
    traceState.selectedRunId = (running || runs[0]).run_id;
  }

  els.traceRun.value = traceState.selectedRunId;
  return runs;
}

function renderTraceEmpty(text) {
  els.traceMeta.textContent = text;
  els.traceList.innerHTML = `<div class="trace-empty">${escapeHtml(text)}</div>`;
}

async function fetchRunTrace(runId, includeResponse) {
  const q = new URLSearchParams({
    run_id: runId,
    limit: String(traceLimit),
    include_response: includeResponse ? "1" : "0",
  });
  const response = await fetch(`/api/run-trace?${q.toString()}`, { cache: "no-store" });
  const raw = await response.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    if (response.status === 404) {
      throw new Error("Trace endpoint not found (404). Restart dashboard server to load latest backend.");
    }
    throw new Error("Trace endpoint returned non-JSON payload. Restart dashboard server.");
  }
  if (!response.ok) {
    throw new Error(data?.error || `trace request failed (${response.status})`);
  }
  return data;
}

function renderTraceRows(payload) {
  const events = Array.isArray(payload?.events) ? payload.events : [];
  const run = payload?.run || null;
  const filterMode = traceState.filterMode || "all";

  if (!run) {
    renderTraceEmpty("No run selected.");
    return;
  }

  const filteredEvents = events.filter((event) => {
    if (filterMode === "refusal") return Boolean(event.assistant_refusal);
    if (filterMode === "success") return !Boolean(event.assistant_refusal);
    return true;
  });

  els.traceMeta.textContent =
    `${run.label} | ${run.backend}:${run.model} | ${run.status.toUpperCase()} | ` +
    `Showing ${filteredEvents.length}/${events.length} loaded turns (filter: ${filterMode})`;

  if (!filteredEvents.length) {
    els.traceList.innerHTML = "<div class='trace-empty'>No turns match the current filter.</div>";
    return;
  }

  const rows = filteredEvents
    .slice()
    .reverse()
    .map((event) => {
      const refusalClass = event.assistant_refusal ? "refusal" : "nonrefusal";
      const refusalText = event.assistant_refusal ? "Refusal" : "Non-Refusal";
      const responseClass = event.assistant_refusal ? "response-refusal" : "response-success";
      const turnLabel = `${event.turn_kind || "turn"}-${event.turn_index || 0}`;
      const goal = event.goal || "-";
      const errorBlock = event.error
        ? `<div class="trace-block"><strong>Error</strong><pre>${escapeHtml(event.error)}</pre></div>`
        : "";

      return `
        <article class="trace-item">
          <div class="trace-item-head">
            <div class="trace-item-meta">
              <span class="trace-badge ${refusalClass}">${refusalText}</span>
              <span class="trace-badge">${escapeHtml(turnLabel)}</span>
              <span class="trace-badge">Example ${escapeHtml(event.example_id || "-")}</span>
            </div>
            <time class="trace-time">${isoToLocal(event.timestamp_utc)}</time>
          </div>
          <div class="trace-goal">Goal: ${escapeHtml(goal)}</div>
          <div class="trace-block">
            <strong>User Prompt</strong>
            <pre>${escapeHtml(event.user_prompt || "")}</pre>
          </div>
          <div class="trace-block response-block ${responseClass}">
            <strong>Assistant Response</strong>
            <pre>${escapeHtml(event.assistant_response || "")}</pre>
          </div>
          ${errorBlock}
        </article>
      `;
    });

  els.traceList.innerHTML = rows.join("");
}

async function updateTrace(batch) {
  const runs = updateTraceRunOptions(batch);
  if (!runs.length || !traceState.selectedRunId) {
    renderTraceEmpty("Start a batch to inspect prompt/response progression.");
    return;
  }

  traceState.includeResponse = Boolean(els.traceReveal.checked);
  traceState.filterMode = els.traceFilter.value || "all";

  try {
    const payload = await fetchRunTrace(traceState.selectedRunId, traceState.includeResponse);
    renderTraceRows(payload);
  } catch (error) {
    renderTraceEmpty(`Trace load error: ${String(error.message || error)}`);
  }
}

function render(snapshot) {
  const batch = findLatestBatch(snapshot);

  els.serverTime.textContent = isoToLocal(snapshot.server_time_utc);
  els.activeBatch.textContent = snapshot.active_batch_id || "None";
  els.modelsSeen.textContent = Array.isArray(snapshot.models_seen) ? snapshot.models_seen.length : "0";

  renderCapabilities(snapshot);
  renderProgress(batch);
  renderMatrix(snapshot, batch);
  renderEffect(batch);
  renderActivity(snapshot);
  renderModelStats(snapshot);
  return batch;
}

async function fetchState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  const raw = await response.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error("State endpoint returned non-JSON payload. Restart dashboard server.");
  }
  if (!response.ok) throw new Error(data?.error || `state request failed (${response.status})`);
  return data;
}

async function startBatch(event) {
  event.preventDefault();
  els.startButton.disabled = true;
  setFormMessage("Launching matrix run...");

  const payload = {
    backend: els.backendSelect.value,
    model: document.getElementById("model").value.trim(),
    dataset_path: document.getElementById("datasetPath").value.trim(),
    fitd_variant: els.fitdVariant.value,
    author_prompt_track: els.authorPromptTrack.value,
    author_prompt_file: els.authorPromptFile.value.trim(),
    author_max_warmup_turns:
      els.fitdVariant.value === "author" ? Number(els.authorMaxWarmups.value || 0) : 0,
    max_examples: Number(document.getElementById("maxExamples").value),
    max_tokens: Number(document.getElementById("maxTokens").value),
    temperature: Number(document.getElementById("temperature").value),
    sleep_seconds: Number(document.getElementById("sleepSeconds").value),
  };

  try {
    const response = await fetch("/api/start-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const raw = await response.text();
    let data = null;
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      throw new Error("Start endpoint returned non-JSON payload. Restart dashboard server.");
    }
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `request failed (${response.status})`);
    }
    traceState.selectedRunId = "";
    setFormMessage(`Started ${data.batch_id}`);
    await tick();
  } catch (error) {
    setFormMessage(String(error.message || error), true);
  } finally {
    els.startButton.disabled = false;
  }
}

async function tick() {
  if (tickInFlight) return;
  tickInFlight = true;

  try {
    const snapshot = await fetchState();
    const batch = render(snapshot);
    await updateTrace(batch);
  } catch (error) {
    setFormMessage(`State poll error: ${String(error.message || error)}`, true);
    renderTraceEmpty("Unable to load state from dashboard server.");
  } finally {
    tickInFlight = false;
  }
}

function startMatrixRain() {
  const canvas = document.getElementById("matrixRain");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const fontSize = 12;
  const lineHeight = 15;
  const gutter = 16;
  const topPad = 12;
  const targetFps = 24;
  const frameMs = 1000 / targetFps;
  const typeMs = 46;
  const charWidthPx = 7.2;
  const history = [];
  let maxVisible = 0;
  let maxChars = 120;
  let outputCountdown = 0;
  let lastFrameTime = 0;
  let typingAccumulator = 0;
  let activeLine = {
    full: "",
    shown: "",
    type: "command",
  };

  const models = ["qwen2.5-3b", "llama3-8b", "mistral-7b", "gemma2-9b"];
  const attacks = ["standard", "fitd"];
  const defenses = ["none", "vigilant"];
  const phases = ["loading_model", "loading_dataset", "running", "example_complete"];

  function randInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function pick(values) {
    return values[randInt(0, values.length - 1)];
  }

  function timestamp() {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const ss = String(now.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
  }

  function buildCommandLine() {
    const phase = pick(phases);
    if (phase === "running") {
      return `[${timestamp()}] student@fitd-lab:~$ python -m fitd_repro --backend hf --model ${pick(models)} --attack ${pick(attacks)} --defense ${pick(defenses)}`;
    }
    return `[${timestamp()}] student@fitd-lab:~$ ./scripts/run_dashboard.sh --phase ${phase}`;
  }

  function buildOutputLine() {
    const mode = randInt(0, 5);
    if (mode === 0) {
      return `INFO  model=${pick(models)} tokenizer loaded in ${randInt(240, 1400)}ms`;
    }
    if (mode === 1) {
      return `TRACE example=${randInt(1, 220)} turn=${pick(["warmup-1", "warmup-2", "warmup-3", "final-1"])} refusal=${pick(["true", "false"])} success=${pick(["true", "false"])}`;
    }
    if (mode === 2) {
      return `METRIC asr=${(Math.random() * 0.9 + 0.05).toFixed(2)} refusal_rate=${(Math.random() * 0.9).toFixed(2)} door_opened=${randInt(0, 48)}`;
    }
    if (mode === 3) {
      return `WARN  policy_guard triggered; switching defense=${pick(defenses)}`;
    }
    if (mode === 4) {
      return `PERF  tok/s=${(Math.random() * 72 + 18).toFixed(1)} latency=${randInt(180, 2300)}ms gpu_mem=${randInt(5, 21)}GB`;
    }
    return `INFO  write summary -> results/batch_${Math.random().toString(16).slice(2, 10)}/summary.json`;
  }

  function nextLine() {
    if (outputCountdown > 0) {
      outputCountdown -= 1;
      return { full: buildOutputLine(), shown: "", type: "output" };
    }
    outputCountdown = randInt(1, 4);
    return { full: buildCommandLine(), shown: "", type: "command" };
  }

  function trimToWidth(text) {
    if (text.length <= maxChars) return text;
    return `${text.slice(0, Math.max(4, maxChars - 3))}...`;
  }

  function pushHistory(entry) {
    history.unshift(entry);
    if (history.length > maxVisible + 20) {
      history.length = maxVisible + 20;
    }
  }

  function stepTyping() {
    if (!activeLine.full) {
      activeLine = nextLine();
      return;
    }
    const remaining = activeLine.full.length - activeLine.shown.length;
    if (remaining <= 0) {
      pushHistory(activeLine);
      activeLine = nextLine();
      return;
    }
    const burst = randInt(1, 3);
    activeLine.shown = activeLine.full.slice(0, activeLine.shown.length + burst);
  }

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    maxVisible = Math.max(8, Math.floor((canvas.height - topPad * 2) / lineHeight) - 2);
    maxChars = Math.max(24, Math.floor((canvas.width - gutter * 2) / charWidthPx));
  }

  function drawLine(text, y, alpha, type) {
    const clipped = trimToWidth(text);
    const color =
      type === "command"
        ? `rgba(244, 224, 255, ${alpha})`
        : `rgba(217, 167, 255, ${Math.max(0.16, alpha - 0.06)})`;
    ctx.fillStyle = color;
    ctx.fillText(clipped, gutter, y);
  }

  function draw(now = 0) {
    if (document.hidden) {
      lastFrameTime = now;
      window.requestAnimationFrame(draw);
      return;
    }

    if (lastFrameTime === 0) {
      lastFrameTime = now;
    }
    const delta = now - lastFrameTime;
    if (delta < frameMs) {
      window.requestAnimationFrame(draw);
      return;
    }
    lastFrameTime = now;

    typingAccumulator += delta;
    while (typingAccumulator >= typeMs) {
      stepTyping();
      typingAccumulator -= typeMs;
    }

    ctx.fillStyle = "rgba(0, 0, 2, 0.68)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.font = `${fontSize}px "IBM Plex Mono", monospace`;
    ctx.textBaseline = "top";
    ctx.shadowColor = "rgba(217, 167, 255, 0.58)";
    ctx.shadowBlur = 7;

    const cursorOn = Math.floor(now / 450) % 2 === 0;
    const activeText = `${activeLine.shown}${cursorOn ? "_" : " "}`;
    drawLine(activeText, topPad, 0.98, "command");

    const visibleHistory = history.slice(0, maxVisible);
    for (let i = 0; i < visibleHistory.length; i += 1) {
      const entry = visibleHistory[i];
      const y = topPad + lineHeight * (i + 1);
      const fade = Math.max(0.18, 0.92 - i * 0.032);
      drawLine(entry.full, y, fade, entry.type);
    }

    window.requestAnimationFrame(draw);
  }

  resize();
  for (let i = 0; i < 18; i += 1) {
    history.unshift({ full: buildOutputLine(), shown: "", type: "output" });
  }
  activeLine = nextLine();
  window.addEventListener("resize", resize);
  window.requestAnimationFrame(draw);
}

els.form.addEventListener("submit", startBatch);
els.fitdVariant.addEventListener("change", () => {
  updateAuthorInputs();
});
els.traceRun.addEventListener("change", () => {
  traceState.selectedRunId = els.traceRun.value;
  void tick();
});
els.traceFilter.addEventListener("change", () => {
  traceState.filterMode = els.traceFilter.value || "all";
  void tick();
});
els.traceReveal.addEventListener("change", () => {
  traceState.includeResponse = Boolean(els.traceReveal.checked);
  void tick();
});

updateAuthorInputs();
startMatrixRain();
renderTraceEmpty("Start a batch to inspect prompt/response progression.");
void tick();
window.setInterval(() => {
  void tick();
}, pollMs);
