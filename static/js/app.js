/**
 * DataIntel Autonomous Agent - Modern SPA Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowse = document.getElementById('btnBrowse');
  const btnExportPdf = document.getElementById('btnExportPdf');
  const btnToggleDrawer = document.getElementById('btnToggleDrawer');
  const btnClearChat = document.getElementById('btnClearChat');
  const btnResetSession = document.getElementById('btnResetSession');
  
  const modelDrawer = document.getElementById('modelDrawer');
  const radarStatusBadge = document.getElementById('radarStatusBadge');
  const specModel = document.getElementById('specModel');
  const radarLatency = document.getElementById('radarLatency');
  const radarRetries = document.getElementById('radarRetries');
  const radarFigures = document.getElementById('radarFigures');
  const pipelineStatusText = document.getElementById('pipelineStatusText');
  const pipelineListTitle = document.getElementById('pipelineListTitle');
  const consoleTerminal = document.getElementById('consoleTerminal');
  const btnClearLogs = document.getElementById('btnClearLogs');
  const llmStreamBox = document.getElementById('llmStreamBox');
  const llmPhaseBadge = document.getElementById('llmPhaseBadge');
  
  const ollamaStatus = document.getElementById('ollamaStatus');
  const ollamaStatusText = document.getElementById('ollamaStatusText');
  
  const kpiTables = document.getElementById('kpiTables');
  const kpiRows = document.getElementById('kpiRows');
  const kpiHealth = document.getElementById('kpiHealth');
  const kpiJoins = document.getElementById('kpiJoins');

  const tablesContainer = document.getElementById('tablesContainer');
  const tableList = document.getElementById('tableList');
  const tableCountBadge = document.getElementById('tableCountBadge');
  
  const schemaContainer = document.getElementById('schemaContainer');
  const relationshipList = document.getElementById('relationshipList');

  const welcomeHero = document.getElementById('welcomeHero');
  const analysisContainer = document.getElementById('analysisContainer');
  const briefingHeadline = document.getElementById('briefingHeadline');
  const briefingNote = document.getElementById('briefingNote');
  const findingsGrid = document.getElementById('findingsGrid');
  const suggestionGrid = document.getElementById('suggestionGrid');
  const chatThread = document.getElementById('chatThread');
  
  const chatForm = document.getElementById('chatForm');
  const queryInput = document.getElementById('queryInput');
  const btnSend = document.getElementById('btnSend');

  const tableModal = document.getElementById('tableModal');
  const modalTableName = document.getElementById('modalTableName');
  const modalTableMeta = document.getElementById('modalTableMeta');
  const modalTableWrapper = document.getElementById('modalTableWrapper');
  const btnModalClose = document.getElementById('btnModalClose');
  const btnModalExport = document.getElementById('btnModalExport');
  
  let currentModalTable = null;

  // 1. Initialize Status & State
  initApp();

  async function initApp() {
    await checkStatus();
    await fetchCurrentState();
  }

  async function checkStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.status === 'online') {
        ollamaStatus.className = 'status-pill';
        ollamaStatusText.innerHTML = `● Ollama: <b>${data.model}</b>`;
      } else {
        ollamaStatus.className = 'status-pill status-offline';
        ollamaStatusText.textContent = 'Ollama Offline';
      }
    } catch (e) {
      ollamaStatusText.textContent = 'Ollama: Not Reachable';
    }
  }

  async function fetchCurrentState() {
    try {
      const res = await fetch('/api/state');
      const data = await res.json();
      if (data.has_data) {
        renderDashboard(data);
      }
    } catch (e) {
      console.error("State restore failed:", e);
    }
  }

  // 2. Drag & Drop File Upload
  btnBrowse.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadFiles(fileInput.files);
    }
  });

  async function streamOnboarding(url, body, isFormData = false) {
    setPipelineMode('onboarding');
    resetPipeline();
    radarStatusBadge.className = 'radar-badge running';
    radarStatusBadge.textContent = 'PROFILING';
    pipelineStatusText.textContent = 'Profiling dataset...';
    radarLatency.textContent = '...';

    logToConsole('SYSTEM', 'Initiating dataset ingestion & automated profiling...', 'INIT');
    resetLlmStream();

    const fetchOpts = {
      method: 'POST',
      body: isFormData ? body : JSON.stringify(body)
    };
    if (!isFormData) {
      fetchOpts.headers = { 'Content-Type': 'application/json' };
    }

    let finalData = null;

    try {
      const response = await fetch(url, fetchOpts);
      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop();

        for (const part of parts) {
          const cleanPart = part.trim();
          if (cleanPart.startsWith('data: ')) {
            const jsonStr = cleanPart.substring(6).trim();
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'onboarding_step') {
                updateStepStatus(event.step_idx, event.status, event.detail);
              } else if (event.type === 'log') {
                logToConsole(event.level, event.msg, event.level);
              } else if (event.type === 'llm_token') {
                handleLlmToken(event.phase, event.token);
              } else if (event.type === 'llm_done') {
                finishLlmPhase(event.phase);
              } else if (event.type === 'onboarding_result') {
                finalData = event.payload;
              } else if (event.type === 'error') {
                throw new Error(event.detail || "Profiling error");
              }
            } catch (err) {
              console.error("Stream parse error:", err);
            }
          }
        }
      }

      if (finalData) {
        renderDashboard(finalData);
        radarStatusBadge.className = 'radar-badge completed';
        radarStatusBadge.textContent = 'READY';
        pipelineStatusText.textContent = 'Profiling Complete';
        if (finalData.duration) {
          radarLatency.textContent = `${finalData.duration}s`;
        }
      } else {
        await fetchCurrentState();
      }

    } catch (e) {
      alert("Profiling error: " + e.message);
      radarStatusBadge.className = 'radar-badge error';
      radarStatusBadge.textContent = 'ERROR';
      logToConsole('ERROR', e.message, 'AUDIT');
    }
  }

  async function uploadFiles(files) {
    const formData = new FormData();
    for (let f of files) {
      formData.append('files', f);
    }

    setDropzoneLoading(true, "Ingesting & profiling tables with local AI...");

    try {
      await streamOnboarding('/api/upload-stream', formData, true);
    } catch (e) {
      alert("Error uploading files: " + e.message);
    } finally {
      setDropzoneLoading(false);
      fileInput.value = '';
    }
  }

  function setDropzoneLoading(isLoading, text = "Uploading...") {
    if (isLoading) {
      dropzone.innerHTML = `<div class="spinner"></div><p class="dropzone-title" style="margin-top:8px;">${text}</p>`;
    } else {
      dropzone.innerHTML = `
        <div class="dropzone-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <p class="dropzone-title">Drag & drop CSV / Excel files</p>
        <p class="dropzone-sub">Upload 1 or multiple related tables</p>
        <button class="btn btn-sm btn-primary" id="btnBrowseInner">Browse Files</button>
      `;
      document.getElementById('btnBrowseInner').addEventListener('click', () => fileInput.click());
    }
  }

  // 4. Render Dashboard Components
  function renderDashboard(data) {
    welcomeHero.style.display = 'none';
    analysisContainer.style.display = 'flex';
    btnExportPdf.disabled = false;
    btnClearChat.disabled = false;
    btnResetSession.disabled = false;

    const p = data.profile;
    const isMulti = p.is_multi_table;

    // Update KPI Bar
    if (isMulti) {
      kpiTables.textContent = p.table_count;
      kpiRows.textContent = p.total_rows.toLocaleString();
      kpiHealth.textContent = `${p.quality_score}/100`;
      kpiJoins.textContent = (p.relationships || []).length;
    } else {
      kpiTables.textContent = "1";
      kpiRows.textContent = p.shape.rows.toLocaleString();
      kpiHealth.textContent = `${p.quality_score}/100`;
      kpiJoins.textContent = "N/A";
    }

    // Render Sidebar Tables
    tablesContainer.style.display = 'block';
    tableCountBadge.textContent = data.tables.length;
    tableList.innerHTML = '';
    
    data.tables.forEach(tName => {
      const item = document.createElement('div');
      item.className = 'table-item';
      
      let rowCountStr = '';
      if (isMulti && p.table_profiles[tName]) {
        rowCountStr = `${p.table_profiles[tName].shape.rows.toLocaleString()} rows`;
      } else if (!isMulti && p.shape) {
        rowCountStr = `${p.shape.rows.toLocaleString()} rows`;
      }

      item.innerHTML = `
        <div class="table-item-info">
          <span class="table-item-name">🗄️ ${tName}</span>
          <span class="table-item-meta">${rowCountStr}</span>
        </div>
        <button class="btn btn-sm btn-subtle btn-inspect" data-table="${tName}">👁️</button>
      `;
      tableList.appendChild(item);
    });

    document.querySelectorAll('.btn-inspect').forEach(btn => {
      btn.addEventListener('click', (e) => {
        openTableInspector(e.currentTarget.dataset.table);
      });
    });

    // Render Relationships (if multi-table)
    if (isMulti && p.relationships && p.relationships.length > 0) {
      schemaContainer.style.display = 'block';
      relationshipList.innerHTML = '';
      p.relationships.forEach(r => {
        const rItem = document.createElement('div');
        rItem.className = 'rel-item';
        rItem.innerHTML = `
          <div class="rel-text">${r.parent_table}.${r.key_column} ➔ ${r.child_table}.${r.key_column}</div>
          <span class="rel-badge">${r.cardinality} • ${r.shared_values_count} keys</span>
        `;
        relationshipList.appendChild(rItem);
      });
    } else {
      schemaContainer.style.display = 'none';
    }

    // Render Proactive Briefing
    if (data.briefing) {
      const b = data.briefing;
      briefingHeadline.textContent = b.executive_headline || "Proactive Onboarding Analysis Complete";
      briefingNote.innerHTML = `🛡️ <b>Data Health Note:</b> ${b.data_health_note || "Schema audited and ready."}`;
      
      findingsGrid.innerHTML = '';
      (b.key_findings || []).forEach(f => {
        const fEl = document.createElement('div');
        fEl.className = 'finding-item';
        fEl.innerHTML = `<span class="finding-icon">📌</span><div>${f}</div>`;
        findingsGrid.appendChild(fEl);
      });

      // Render Suggestions
      suggestionGrid.innerHTML = '';
      (b.suggested_questions || []).slice(0, 4).forEach(q => {
        const sBtn = document.createElement('div');
        sBtn.className = 'suggestion-card';
        sBtn.innerHTML = `<span>🔍</span><div>${q}</div>`;
        sBtn.addEventListener('click', () => {
          submitQuery(q);
        });
        suggestionGrid.appendChild(sBtn);
      });
    }

    // Render Chat History
    chatThread.innerHTML = '';
    (data.chat_history || []).forEach(chat => {
      appendChatMessage('user', chat.query);
      appendAssistantMessage(chat);
    });

    if (data.chat_history && data.chat_history.length > 0) {
      const lastChat = data.chat_history[data.chat_history.length - 1];
      if (lastChat.telemetry) {
        loadTelemetryToDrawer(lastChat.telemetry);
        radarStatusBadge.className = 'radar-badge completed';
        radarStatusBadge.textContent = 'RESTORED';
        pipelineStatusText.textContent = 'Session Restored';
      }
    }
  }

  // 5. Chat Query Execution
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (query) {
      submitQuery(query);
    }
  });

  // --- Model Radar & Telemetry Helpers ---
  const PIPELINE_MODES = {
    onboarding: {
      title: "ONBOARDING & PROFILING PIPELINE",
      steps: [
        { label: "1. Ingestion & File Parsing", sub: "Reading tables & raw records" },
        { label: "2. Automated Data Cleaning", sub: "Standardizing headers & formats" },
        { label: "3. Statistical Profiling", sub: "Calculating nulls, stats & health" },
        { label: "4. Relational Key Discovery", sub: "Discovering joins & DuckDB mount" },
        { label: "5. Local LLM Executive Briefing", sub: "Local Qwen 2.5-Coder:3B schema audit" }
      ]
    },
    react: {
      title: "REACT PIPELINE EXECUTION",
      steps: [
        { label: "1. Intent & Schema Mapping", sub: "Table resolution & real data sample injection" },
        { label: "2. Analytical Planning", sub: "LLM formulates strategy & flags pitfalls" },
        { label: "3. Code Generation", sub: "Plan-informed Python & DuckDB script" },
        { label: "4. Sandbox & Self-Healing", sub: "Isolated execution & auto error correction" },
        { label: "5. Business Synthesis", sub: "Executive findings & ROI recommendations" }
      ]
    }
  };

  function setPipelineMode(mode) {
    const config = PIPELINE_MODES[mode] || PIPELINE_MODES.react;
    if (pipelineListTitle) pipelineListTitle.textContent = config.title;

    config.steps.forEach((st, idx) => {
      const stepIdx = idx + 1;
      const stepEl = document.getElementById(`step${stepIdx}`);
      const labelEl = stepEl ? stepEl.querySelector('.step-label') : null;
      const subEl = document.getElementById(`subStep${stepIdx}`);
      if (labelEl) labelEl.textContent = st.label;
      if (subEl) subEl.textContent = st.sub;
    });
  }

  function logToConsole(tag, message, level = 'INFO') {
    const row = document.createElement('div');
    row.className = 'log-row';
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    row.innerHTML = `<span class="log-time">[${timeStr}]</span> <span class="log-tag log-tag-${escapeHtml(level)}">[${escapeHtml(tag)}]</span> ${escapeHtml(message)}`;
    consoleTerminal.appendChild(row);
    consoleTerminal.scrollTop = consoleTerminal.scrollHeight;
  }

  // --- Live LLM Token Streaming ---
  let _llmCursor = null;

  const PHASE_LABELS = {
    planning:   '🟠 PLANNING',
    codegen:    '🟣 CODE-GEN',
    fix:        '🔴 SELF-HEAL',
    synthesis:  '🔵 SYNTHESIS',
    validation: '🟢 VALIDATION'
  };

  function handleLlmToken(phase, token) {
    // Switch phase badge if needed
    const phaseText = PHASE_LABELS[phase] || phase.toUpperCase();
    if (llmPhaseBadge.textContent !== phaseText) {
      llmPhaseBadge.textContent = phaseText;
      llmPhaseBadge.className = `llm-phase-badge ${phase}`;
      // Clear box when phase changes
      llmStreamBox.innerHTML = '';
      _llmCursor = null;
    }
    // Remove cursor before appending token
    if (_llmCursor) _llmCursor.remove();
    // Append token as text node
    llmStreamBox.appendChild(document.createTextNode(token));
    // Re-add blinking cursor at end
    _llmCursor = document.createElement('span');
    _llmCursor.className = 'llm-cursor';
    llmStreamBox.appendChild(_llmCursor);
    // Auto-scroll
    llmStreamBox.scrollTop = llmStreamBox.scrollHeight;
  }

  function finishLlmPhase(phase) {
    // Remove cursor and mark done
    if (_llmCursor) { _llmCursor.remove(); _llmCursor = null; }
    llmPhaseBadge.textContent = `✓ ${(PHASE_LABELS[phase] || phase).replace(/^.+ /, '')} done`;
  }

  function resetLlmStream() {
    llmStreamBox.innerHTML = '<span class="llm-placeholder">Waiting for LLM call...</span>';
    llmPhaseBadge.textContent = 'IDLE';
    llmPhaseBadge.className = 'llm-phase-badge idle';
    _llmCursor = null;
  }

  function resetPipeline() {
    for (let i = 1; i <= 5; i++) {
      const el = document.getElementById(`step${i}`);
      const badge = document.getElementById(`badgeStep${i}`);
      if (el) el.className = 'stepper-item';
      if (badge) {
        badge.textContent = 'Idle';
        badge.className = 'step-badge';
      }
    }
    radarStatusBadge.className = 'radar-badge';
    radarStatusBadge.textContent = 'IDLE';
    pipelineStatusText.textContent = 'Ready';
  }

  function updateStepStatus(stepIdx, status, detail) {
    const stepEl = document.getElementById(`step${stepIdx}`);
    const badgeEl = document.getElementById(`badgeStep${stepIdx}`);
    const subEl = document.getElementById(`subStep${stepIdx}`);
    if (!stepEl) return;

    stepEl.className = `stepper-item ${status}`;
    if (badgeEl) {
      badgeEl.textContent = status === 'running' ? 'Active' : (status === 'completed' ? 'Done' : (status === 'warning' ? 'Retrying' : status));
      badgeEl.className = `step-badge ${status}`;
    }
    if (subEl && detail) {
      subEl.textContent = detail;
    }

    if (status === 'running') {
      radarStatusBadge.className = 'radar-badge running';
      radarStatusBadge.textContent = 'RUNNING';
      pipelineStatusText.textContent = `Stage ${stepIdx} / 5`;
    }
  }

  function loadTelemetryToDrawer(telemetry) {
    if (!telemetry) return;
    radarLatency.textContent = telemetry.total_duration_seconds ? `${telemetry.total_duration_seconds}s` : '--';
    radarRetries.textContent = telemetry.retries !== undefined ? telemetry.retries : 0;
    radarFigures.textContent = telemetry.charts_count !== undefined ? telemetry.charts_count : 0;
    if (telemetry.model) {
      specModel.textContent = telemetry.model;
    }
    // Mark completed pipeline steps
    if (telemetry.steps && telemetry.steps.length > 0) {
      telemetry.steps.forEach((st, idx) => {
        const stepNum = idx + 1;
        if (stepNum <= 5) {
          updateStepStatus(stepNum, st.status || 'completed', st.detail);
        }
      });
    }
  }

  // Action Buttons
  btnClearChat.addEventListener('click', async () => {
    if (!confirm("Are you sure you want to clear the conversation? Data tables will remain active.")) return;
    try {
      const res = await fetch('/api/chat/clear', { method: 'POST' });
      if (res.ok) {
        chatThread.innerHTML = '';
        resetPipeline();
        radarLatency.textContent = '--';
        radarRetries.textContent = '0';
        radarFigures.textContent = '0';
        logToConsole('SYSTEM', 'Chat history cleared. Active tables remain ready.', 'INIT');
      }
    } catch (e) {
      alert("Error clearing chat: " + e.message);
    }
  });

  btnResetSession.addEventListener('click', async () => {
    if (!confirm("Reset entire session? This clears all loaded tables and conversation history.")) return;
    try {
      const res = await fetch('/api/session/reset', { method: 'POST' });
      if (res.ok) {
        window.location.reload();
      }
    } catch (e) {
      alert("Error resetting session: " + e.message);
    }
  });

  btnToggleDrawer.addEventListener('click', () => {
    modelDrawer.classList.toggle('collapsed');
  });

  btnClearLogs.addEventListener('click', () => {
    consoleTerminal.innerHTML = `<div class="log-row sys"><span class="log-time">[${new Date().toTimeString().split(' ')[0]}]</span> <span class="log-tag log-tag-INIT">[INIT]</span> Console cleared.</div>`;
  });

  async function submitQuery(query) {
    appendChatMessage('user', query);
    queryInput.value = '';

    btnSend.disabled = true;
    btnClearChat.disabled = true;
    btnResetSession.disabled = true;

    setPipelineMode('react');
    resetPipeline();
    resetLlmStream();
    radarStatusBadge.className = 'radar-badge running';
    radarStatusBadge.textContent = 'RUNNING';
    pipelineStatusText.textContent = 'Initializing...';
    radarLatency.textContent = '...';

    logToConsole('USER', `Dispatched query: "${query}"`, 'INFO');

    const thinkingId = 'thinking_' + Date.now();
    appendThinkingMessage(thinkingId);

    let finalPayload = null;

    try {
      const response = await fetch('/api/query-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop(); // keep trailing incomplete segment

        for (const part of parts) {
          const cleanPart = part.trim();
          if (cleanPart.startsWith('data: ')) {
            const jsonStr = cleanPart.substring(6).trim();
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'step') {
                updateStepStatus(event.step_idx, event.status, event.detail);
              } else if (event.type === 'log') {
                logToConsole(event.level, event.msg, event.level);
              } else if (event.type === 'llm_token') {
                handleLlmToken(event.phase, event.token);
              } else if (event.type === 'llm_done') {
                finishLlmPhase(event.phase);
              } else if (event.type === 'init') {
                if (event.model) specModel.textContent = event.model;
              } else if (event.type === 'result') {
                finalPayload = event.payload;
              } else if (event.type === 'error') {
                throw new Error(event.detail || "Agent analysis error");
              }
            } catch (err) {
              console.error("Stream parse error:", err, cleanPart);
            }
          }
        }
      }

      removeMessage(thinkingId);

      if (finalPayload) {
        appendAssistantMessage(finalPayload);
        loadTelemetryToDrawer(finalPayload.telemetry);
        radarStatusBadge.className = finalPayload.success ? 'radar-badge completed' : 'radar-badge error';
        radarStatusBadge.textContent = finalPayload.success ? 'COMPLETE' : 'FAILED';
        pipelineStatusText.textContent = finalPayload.success ? 'Completed' : 'Audit Failed';
      } else {
        appendErrorMessage("No analysis response returned from agent.");
        radarStatusBadge.className = 'radar-badge error';
        radarStatusBadge.textContent = 'NO_DATA';
      }

    } catch (e) {
      removeMessage(thinkingId);
      appendErrorMessage(e.message);
      radarStatusBadge.className = 'radar-badge error';
      radarStatusBadge.textContent = 'ERROR';
      logToConsole('ERROR', e.message, 'AUDIT');
    } finally {
      btnSend.disabled = false;
      btnClearChat.disabled = false;
      btnResetSession.disabled = false;
      queryInput.focus();
    }
  }

  function appendChatMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `chat-msg ${role}`;
    msg.innerHTML = `
      <div class="msg-avatar">${role === 'user' ? '👤' : '🤖'}</div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    `;
    chatThread.appendChild(msg);
    scrollChat();
  }

  function appendThinkingMessage(id) {
    const msg = document.createElement('div');
    msg.className = 'chat-msg assistant';
    msg.id = id;
    msg.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble" style="display:flex;align-items:center;gap:10px;">
        <div class="spinner"></div>
        <span style="color:var(--text-muted);font-size:13px;">Qwen 2.5-Coder:3B is planning relational analysis & running DuckDB code...</span>
      </div>
    `;
    chatThread.appendChild(msg);
    scrollChat();
  }

  function appendAssistantMessage(data) {
    const msg = document.createElement('div');
    msg.className = 'chat-msg assistant';

    let codeAccordionHtml = '';
    if (data.code) {
      codeAccordionHtml = `
        <details class="code-accordion">
          <summary class="code-accordion-header">
            <span>⚡ View Python / DuckDB SQL Execution</span>
            <span class="code-accordion-chevron">▼</span>
          </summary>
          <div class="code-accordion-body">
            <div class="code-block">${escapeHtml(data.code)}</div>
            ${data.stdout ? `<div class="stdout-block"><strong>Metrics / Output:</strong><br>${escapeHtml(data.stdout)}</div>` : ''}
          </div>
        </details>
      `;
    }

    let chartsHtml = '';
    if (data.charts && data.charts.length > 0) {
      data.charts.forEach(cUrl => {
        chartsHtml += `
          <div class="chart-container">
            <img class="chart-img" src="${cUrl}" alt="Generated Chart">
          </div>
        `;
      });
    }

    let telemetryBtnHtml = '';
    if (data.telemetry) {
      const qs = data.telemetry.quality_score;
      const qsColor = qs >= 8 ? '#4ade80' : qs >= 7 ? '#fbbf24' : '#f87171';
      const qsLabel = qs ? `<span style="font-size:11px;padding:2px 8px;border-radius:10px;background:${qsColor}22;color:${qsColor};font-weight:700;border:1px solid ${qsColor}44;">✦ Quality ${qs}/10</span>` : '';
      telemetryBtnHtml = `
        <div style="margin-top:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <button class="btn btn-sm btn-subtle btn-inspect-radar">🧠 Inspect Model Thoughts (${data.telemetry.total_duration_seconds || '--'}s)</button>
          ${qsLabel}
        </div>
      `;
    }

    msg.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble">
        ${codeAccordionHtml}
        ${chartsHtml}
        <div class="assistant-narrative">${formatMarkdown(data.answer)}</div>
        ${telemetryBtnHtml}
      </div>
    `;

    const btnRadar = msg.querySelector('.btn-inspect-radar');
    if (btnRadar && data.telemetry) {
      btnRadar.addEventListener('click', () => {
        loadTelemetryToDrawer(data.telemetry);
        modelDrawer.classList.remove('collapsed');
      });
    }

    chatThread.appendChild(msg);
    scrollChat();
  }

  function appendErrorMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'chat-msg assistant';
    msg.innerHTML = `
      <div class="msg-avatar">⚠️</div>
      <div class="msg-bubble" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.3);color:#fca5a5;">
        <strong>Analysis Paused:</strong> ${escapeHtml(text)}
      </div>
    `;
    chatThread.appendChild(msg);
    scrollChat();
  }

  function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollChat() {
    analysisContainer.scrollTop = analysisContainer.scrollHeight;
  }

  // 6. Table Inspector Modal
  async function openTableInspector(tableName) {
    currentModalTable = tableName;
    modalTableName.textContent = `🗄️ ${tableName}`;
    modalTableMeta.textContent = "Loading preview...";
    modalTableWrapper.innerHTML = '<div class="spinner" style="margin:20px auto;display:block;"></div>';
    tableModal.classList.add('active');

    try {
      const res = await fetch(`/api/table/${tableName}/preview`);
      if (!res.ok) throw new Error("Could not load preview");
      const data = await res.json();
      
      modalTableMeta.textContent = `${data.total_rows.toLocaleString()} rows • ${data.total_columns} columns`;

      let tableHtml = '<table><thead><tr>';
      data.columns.forEach(col => {
        tableHtml += `<th>${escapeHtml(col)}</th>`;
      });
      tableHtml += '</tr></thead><tbody>';

      data.records.forEach(row => {
        tableHtml += '<tr>';
        data.columns.forEach(col => {
          const val = row[col] !== null && row[col] !== undefined ? row[col] : '<span style="color:#64748b;">null</span>';
          tableHtml += `<td>${val}</td>`;
        });
        tableHtml += '</tr>';
      });

      tableHtml += '</tbody></table>';
      modalTableWrapper.innerHTML = tableHtml;
    } catch (e) {
      modalTableWrapper.innerHTML = `<p style="color:#ef4444;">Error loading table: ${e.message}</p>`;
    }
  }

  btnModalClose.addEventListener('click', () => tableModal.classList.remove('active'));
  tableModal.addEventListener('click', (e) => {
    if (e.target === tableModal) tableModal.classList.remove('active');
  });

  btnModalExport.addEventListener('click', async () => {
    if (!currentModalTable) return;
    const res = await fetch(`/api/table/${currentModalTable}/export`);
    const data = await res.json();
    downloadBlob(data.csv, data.filename, 'text/csv');
  });

  // 7. Executive PDF Export
  btnExportPdf.addEventListener('click', async () => {
    btnExportPdf.disabled = true;
    btnExportPdf.innerHTML = '<div class="spinner"></div> Generating...';
    try {
      const res = await fetch('/api/export-pdf', { method: 'POST' });
      if (!res.ok) throw new Error("Failed to generate PDF");
      const blob = await res.blob();
      downloadBlob(blob, 'executive_data_report.pdf', 'application/pdf');
    } catch (e) {
      alert("PDF Generation Error: " + e.message);
    } finally {
      btnExportPdf.disabled = false;
      btnExportPdf.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Export PDF
      `;
    }
  });

  // Helpers
  function downloadBlob(content, filename, mimeType) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let out = text;
    // Replace headers
    out = out.replace(/^### (.*$)/gim, '<h4 style="margin:12px 0 6px;color:#f8fafc;font-size:15px;">$1</h4>');
    out = out.replace(/^## (.*$)/gim, '<h3 style="margin:14px 0 8px;color:#38bdf8;font-size:16px;">$1</h3>');
    out = out.replace(/^# (.*$)/gim, '<h2 style="margin:16px 0 8px;color:#f8fafc;font-size:18px;">$1</h2>');
    // Bold
    out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Bullet points
    out = out.replace(/^\* (.*$)/gim, '<li style="margin-left:20px;margin-bottom:4px;">$1</li>');
    out = out.replace(/^- (.*$)/gim, '<li style="margin-left:20px;margin-bottom:4px;">$1</li>');
    // Newlines to break
    out = out.replace(/\n\n/g, '<br><br>');
    return out;
  }
});
