document.addEventListener('DOMContentLoaded', () => {
  // UI Elements
  const chatInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const newChatBtn = document.getElementById('new-chat-btn');
  const chatHistory = document.getElementById('chat-history');
  const memoryQueueList = document.getElementById('memory-queue-list');
  const historyListContainer = document.getElementById('history-list');

  // State Elements
  const stateComplexity = document.getElementById('state-complexity');

  // Cost Elements
  const costLlm = document.getElementById('cost-llm');
  const totalCostLlm = document.getElementById('total-cost-llm');
  const costSearch = document.getElementById('cost-search');
  const totalSearch = document.getElementById('total-search');
  const costCode = document.getElementById('cost-code');
  const totalCostCode = document.getElementById('total-cost-code');
  const costTime = document.getElementById('cost-time');

  // Config Elements
  const agentType = document.getElementById('agent-type');
  const memorySizeInput = document.getElementById('memory-size');
  const thresholdT1 = document.getElementById('threshold-t1');
  const thresholdT2 = document.getElementById('threshold-t2');
  const toggleRetrieval = document.getElementById('toggle-retrieval');
  const toggleMemory = document.getElementById('toggle-memory');
  const toggleFailure = document.getElementById('toggle-failure');
  const thresholdSim = document.getElementById('threshold-sim');

  // View Switching Elements
  const chatView = document.getElementById('chat-view');
  const libView = document.getElementById('library-view');
  const showChatBtn = document.getElementById('show-chat-btn');
  const showLibBtn = document.getElementById('show-lib-btn');

  let shortTermMemory = [];
  let totalLlmCost = 0.0;
  let currentSearchCount = 0;
  let totalSearchCount = 0;
  let currentCodeCost = 0.0;
  let totalCodeCost = 0.0;
  let currentChatId = null;
  let ws = null;
  let currentMemoryHit = false;
  let currentMsgId = null;
  let isProcessing = false;
  let reconnectTimer = null;
  let isUnloading = false;

  const API_BASE = 'http://127.0.0.1:8011';
  const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/agent';

  // --- Helper: Generate UUID ---
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  function createTextDiv(className, text) {
    const div = document.createElement('div');
    div.className = className;
    div.textContent = text;
    return div;
  }

  function appendTextWithBreaks(container, text) {
    const parts = String(text ?? '').split('\n');
    parts.forEach((part, index) => {
      if (index > 0) container.appendChild(document.createElement('br'));
      if (part) container.appendChild(document.createTextNode(part));
    });
  }

  function clampNumber(value, fallback, min, max) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  }

  function sanitizeHistoryHTML(html) {
    const template = document.createElement('template');
    template.innerHTML = html || '';
    template.content.querySelectorAll('script, style, iframe, object, embed, link, .system-msg, .thinking-indicator').forEach(node => node.remove());
    template.content.querySelectorAll('*').forEach(node => {
      [...node.attributes].forEach(attr => {
        const name = attr.name.toLowerCase();
        const value = String(attr.value || '').trim().toLowerCase();
        if (name.startsWith('on') || name === 'srcdoc' || value.startsWith('javascript:')) {
          node.removeAttribute(attr.name);
        }
      });
    });
    return template.innerHTML;
  }

  function persistentHistoryHTML() {
    return sanitizeHistoryHTML(chatHistory.innerHTML);
  }

  function readStat(stats, snakeName, camelName) {
    const value = stats?.[snakeName] ?? stats?.[camelName];
    return Number(value) || 0;
  }

  function buildChatState() {
    if (!currentChatId) return null;
    return {
      id: currentChatId,
      title: chatHistory.querySelector('.user-msg')?.textContent?.substring(0, 30) || 'New Conversation',
      timestamp: new Date().toISOString(),
      historyHTML: persistentHistoryHTML(),
      memoryItems: shortTermMemory,
      agentType: agentType.value,
      stats: {
        total_llm_cost: totalLlmCost,
        total_search_count: totalSearchCount,
        total_code_cost: totalCodeCost
      }
    };
  }

  function currentConfig() {
    const config = {
      agent_type: agentType.value,
      memory_size: Math.round(clampNumber(memorySizeInput.value, 5, 1, 20)),
      innovations: {
        retrieval: toggleRetrieval.checked,
        memory: toggleMemory.checked,
        failure: toggleFailure.checked
      },
      thresholds: {
        t1: clampNumber(thresholdT1.value, 0.9, 0, 1),
        t2: clampNumber(thresholdT2.value, 0.6, 0, 1),
        sim: clampNumber(thresholdSim.value, 0.9, 0, 1)
      }
    };
    memorySizeInput.value = config.memory_size;
    thresholdT1.value = config.thresholds.t1;
    thresholdT2.value = config.thresholds.t2;
    thresholdSim.value = config.thresholds.sim;
    return config;
  }

  function syncLoadedConversation() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !currentChatId) return;
    ws.send(JSON.stringify({
      type: 'load_conversation',
      data: {
        memory: shortTermMemory,
        stats: {
          total_llm_cost: totalLlmCost,
          total_search_count: totalSearchCount,
          total_code_cost: totalCodeCost
        },
        agent_type: agentType.value || 'react'
      }
    }));
  }

  function setProcessing(nextValue) {
    isProcessing = nextValue;
    sendBtn.disabled = nextValue;
    chatInput.disabled = nextValue;
  }

  // --- Persistence Logic (Backend API) ---
  async function saveChatState() {
    const state = buildChatState();
    if (!state) return;

    try {
      await fetch(`${API_BASE}/api/history/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state)
      });
      renderHistory();
    } catch (e) {
      console.error("Failed to save state to backend", e);
    }
  }

  function saveChatStateBeforeUnload() {
    const state = buildChatState();
    if (!state) return;
    const payload = JSON.stringify(state);
    if (navigator.sendBeacon) {
      navigator.sendBeacon(`${API_BASE}/api/history/save`, new Blob([payload], { type: 'application/json' }));
    } else {
      fetch(`${API_BASE}/api/history/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: true
      });
    }
  }

  async function renderHistory() {
    try {
      const response = await fetch(`${API_BASE}/api/history`);
      if (!response.ok) throw new Error(`History API returned ${response.status}`);
      const historyList = await response.json();

      historyListContainer.innerHTML = '';
      if (historyList.length === 0) {
        historyListContainer.appendChild(createTextDiv('empty-history', 'No history yet'));
        return;
      }

      historyList.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        if (item.id === currentChatId) div.classList.add('active');

        const info = document.createElement('div');
        info.className = 'history-info';
        info.title = item.title || 'Unknown Chat';

        const time = document.createElement('div');
        time.style.fontSize = '0.7rem';
        time.style.color = 'var(--text-muted)';
        time.textContent = item.timestamp ? new Date(item.timestamp).toLocaleString() : '';

        const title = document.createElement('div');
        title.textContent = item.title || 'Unknown Chat';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.type = 'button';
        deleteBtn.innerHTML = '<i class="ph ph-trash"></i>';
        deleteBtn.addEventListener('click', event => {
          event.stopPropagation();
          window.deleteChatMessage(item.id);
        });

        info.appendChild(time);
        info.appendChild(title);
        div.appendChild(info);
        div.appendChild(deleteBtn);
        div.onclick = () => loadConversation(item.id);
        historyListContainer.appendChild(div);
      });
    } catch (e) {
      console.error("Failed to fetch history", e);
      historyListContainer.innerHTML = '';
      historyListContainer.appendChild(createTextDiv('empty-history', 'Failed to load history'));
    }
  }

  window.deleteChatMessage = async (id) => {
    if (!confirm('Delete this conversation?')) return;
    try {
      await fetch(`${API_BASE}/api/history/${id}`, { method: 'DELETE' });
      if (id === currentChatId) initNewChat();
      else renderHistory();
    } catch (e) {
      console.error("Failed to delete history", e);
    }
  };

  async function loadConversation(id) {
    try {
      const response = await fetch(`${API_BASE}/api/history/${id}`);
      if (!response.ok) throw new Error(`Conversation API returned ${response.status}`);
      const item = await response.json();

      currentChatId = item.id;
      chatHistory.innerHTML = sanitizeHistoryHTML(item.historyHTML);

      const stats = item.stats || {};
      totalLlmCost = readStat(stats, 'total_llm_cost', 'totalLlmCost');
      totalSearchCount = readStat(stats, 'total_search_count', 'totalSearchCount');
      totalCodeCost = readStat(stats, 'total_code_cost', 'totalCodeCost');

      shortTermMemory = Array.isArray(item.memoryItems) ? item.memoryItems : [];
      updateMemoryQueue(shortTermMemory);
      if (item.agentType) agentType.value = item.agentType;

      // Restore stats to UI
      totalCostLlm.textContent = `¥${totalLlmCost.toFixed(5)}`;
      totalSearch.textContent = totalSearchCount;
      totalCostCode.textContent = `¥${totalCodeCost.toFixed(5)}`;

      // Sync with backend runner
      if (ws && ws.readyState === WebSocket.OPEN) {
        syncLoadedConversation();
        applyConfig();
      }

      renderHistory();
      showChatBtn.click();
      addSystemMessage('Conversation restored.');
    } catch (e) {
      console.error("Failed to load conversation", e);
    }
  }

  function initNewChat() {
    currentChatId = generateUUID();
    chatHistory.innerHTML = '<div class="message system-msg">New session started.</div>';
    shortTermMemory = [];
    activeAgentMsgDiv = null;
    currentMsgId = null;
    currentMemoryHit = false;
    setProcessing(false);
    totalLlmCost = 0;
    totalSearchCount = 0;
    totalCodeCost = 0;

    updateCost({
      llm_cost: 0, total_llm_cost: 0,
      search_count: 0, total_search_count: 0,
      code_cost: 0, total_code_cost: 0,
      time_spent: 0
    });

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'clear_memory' }));
      applyConfig(); // Ensure backend is synced with current UI config
    }

    updateMemoryQueue([]);
    renderHistory();
  }

  // Auto-Apply Config Function
  function applyConfig() {
    const config = currentConfig();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'config', data: config }));
      console.log('Config auto-applied:', config);
    }
  }

  // Attach auto-apply listeners
  [agentType, memorySizeInput, thresholdT1, thresholdT2, thresholdSim, toggleRetrieval, toggleMemory, toggleFailure].forEach(el => {
    el.addEventListener('change', applyConfig);
  });

  // Threshold refinement listeners
  thresholdT1.addEventListener('input', () => {
    if (thresholdT1.value > 1) thresholdT1.value = 1;
    if (thresholdT1.value < 0) thresholdT1.value = 0;
  });
  thresholdT2.addEventListener('input', () => {
    if (thresholdT2.value > 1) thresholdT2.value = 1;
    if (thresholdT2.value < 0) thresholdT2.value = 0;
  });
  thresholdSim.addEventListener('input', () => {
    if (thresholdSim.value > 1) thresholdSim.value = 1;
    if (thresholdSim.value < 0) thresholdSim.value = 0;
  });
  memorySizeInput.addEventListener('input', () => {
    if (memorySizeInput.value > 20) memorySizeInput.value = 20;
    if (memorySizeInput.value < 1) memorySizeInput.value = 1;
  });

  // --- View Switching ---
  showChatBtn.addEventListener('click', () => {
    chatView.classList.remove('hidden');
    libView.classList.add('hidden');
    showChatBtn.classList.add('active');
    showLibBtn.classList.remove('active');
  });

  showLibBtn.addEventListener('click', () => {
    chatView.classList.add('hidden');
    libView.classList.remove('hidden');
    showChatBtn.classList.remove('active');
    showLibBtn.classList.add('active');
    renderLibrary();
  });

  async function renderLibrary() {
    const edgeToolList = document.getElementById('edge-tool-list');
    const cloudToolList = document.getElementById('cloud-tool-list');
    const expDbList = document.getElementById('exp-db-list');

    edgeToolList.innerHTML = 'Loading...';
    cloudToolList.innerHTML = 'Loading...';
    expDbList.innerHTML = 'Loading...';

    try {
      const response = await fetch(`${API_BASE}/api/library?agent_type=${encodeURIComponent(agentType.value || 'react')}`);
      if (!response.ok) throw new Error(`Library API returned ${response.status}`);
      const data = await response.json();
      renderLibItems(data.edge_tool, edgeToolList, 'No Edge Tool Records', true);
      renderLibItems(data.cloud_tool, cloudToolList, 'No Cloud Tool Records', true);
      renderLibItems(data.exp_db, expDbList, 'No Experience Records', false);
    } catch (error) {
      [edgeToolList, cloudToolList, expDbList].forEach(container => {
        container.innerHTML = '';
        container.appendChild(createTextDiv('empty-queue', 'Error loading library.'));
      });
    }
  }

  function renderLibItems(recordsObj, container, emptyMsg, isTool = true) {
    container.innerHTML = '';
    const keys = Object.keys(recordsObj || {});
    if (keys.length === 0) {
      container.appendChild(createTextDiv('empty-queue', emptyMsg));
      return;
    }
    const displayKeys = keys.slice(0, 50);
    displayKeys.forEach(key => {
      const entry = recordsObj[key];
      const div = document.createElement('div');
      div.className = 'record-card';
      if (isTool) {
        const records = Array.isArray(entry) ? entry : [entry];
        records.slice(0, 3).forEach(rec => {
          const itemDiv = document.createElement('div');
          const name = document.createElement('div');
          name.style.color = 'var(--primary)';
          name.style.fontWeight = 'bold';
          name.textContent = `[${key}]`;
          const input = document.createElement('div');
          input.style.color = 'var(--text-muted)';
          input.textContent = rec.tool_input || rec.input || '';
          const output = document.createElement('div');
          output.style.fontStyle = 'italic';
          output.textContent = `${String(rec.tool_output || rec.output || '').substring(0, 100)}...`;
          itemDiv.appendChild(name);
          itemDiv.appendChild(input);
          itemDiv.appendChild(output);
          div.appendChild(itemDiv);
        });
      } else {
        const name = document.createElement('div');
        name.style.color = 'var(--primary)';
        name.style.fontWeight = 'bold';
        name.textContent = key;
        const experience = document.createElement('div');
        experience.style.fontStyle = 'italic';
        experience.textContent = `${String(entry.experience || '').substring(0, 150)}...`;
        div.appendChild(name);
        div.appendChild(experience);
      }
      container.appendChild(div);
    });
  }

  // --- WebSocket Logic ---
  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      addSystemMessage('Connected to Edge-Cloud Agent Backend.');
      // Auto-set defaults
      applyConfig();
      syncLoadedConversation();
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleAgentEvent(data);
      } catch (e) {
        console.error("Invalid WS message:", event.data);
      }
    };
    ws.onclose = () => {
      setProcessing(false);
      if (!isUnloading && !reconnectTimer) {
        addSystemMessage('Disconnected. Reconnecting...');
        reconnectTimer = setTimeout(connectWebSocket, 3000);
      }
    };
  }

  function handleAgentEvent(data) {
    if (data.type === 'msg_id') {
      currentMsgId = data.msg_id;
      if (activeAgentMsgDiv) activeAgentMsgDiv.dataset.id = currentMsgId;
    } else if (data.type === 'thought') {
      updateActiveAgentMessage(data.content, true);
    } else if (data.type === 'response') {
      updateActiveAgentMessage(data.content, false);
      saveChatState();
    } else if (data.type === 'error') {
      addSystemMessage(data.content || 'Backend error.');
      setProcessing(false);
    } else if (data.type === 'state') {
      currentMemoryHit = !!data.memory_hit;
      updateState(data.complexity, data.memory_hit);
    } else if (data.type === 'cost') {
      updateCost(data);
    } else if (data.type === 'memory_queue') {
      shortTermMemory = data.items;
      updateMemoryQueue(data.items);
    } else if (data.type === 'done') {
      if (activeThinkingIndicator) {
        activeThinkingIndicator.remove();
        activeThinkingIndicator = null;
      }
      activeAgentMsgDiv = null;
      currentMsgId = null;
      setProcessing(false);
      saveChatState();
    }
  }

  function updateCost(data) {
    if (data.llm_cost !== undefined && costLlm) costLlm.textContent = `¥${(Number(data.llm_cost) || 0).toFixed(5)}`;
    if (data.total_llm_cost !== undefined && totalCostLlm) {
      totalLlmCost = Number(data.total_llm_cost) || 0;
      totalCostLlm.textContent = `¥${totalLlmCost.toFixed(5)}`;
    }
    if (data.search_count !== undefined && costSearch) {
      currentSearchCount = Number(data.search_count) || 0;
      costSearch.textContent = currentSearchCount;
    }
    if (data.total_search_count !== undefined && totalSearch) {
      totalSearchCount = Number(data.total_search_count) || 0;
      totalSearch.textContent = totalSearchCount;
    }
    if (data.code_cost !== undefined && costCode) {
      currentCodeCost = Number(data.code_cost) || 0;
      costCode.textContent = `¥${currentCodeCost.toFixed(5)}`;
    }
    if (data.total_code_cost !== undefined && totalCostCode) {
      totalCodeCost = Number(data.total_code_cost) || 0;
      totalCostCode.textContent = `¥${totalCodeCost.toFixed(5)}`;
    }
    if (data.time_spent !== undefined && costTime) costTime.textContent = `${(Number(data.time_spent) || 0).toFixed(2)} s`;
  }

  newChatBtn.addEventListener('click', () => {
    if (confirm('Start a new chat? Current state will be saved.')) {
      saveChatState().then(() => initNewChat());
    }
  });

  let activeAgentMsgDiv = null;
  let activeAgentThoughtDiv = null;
  let activeAgentContentDiv = null;
  let activeThinkingIndicator = null;

  function addUserMessage(msg) {
    const isAtBottom = chatHistory.scrollHeight - chatHistory.scrollTop <= chatHistory.clientHeight + 150;
    const div = document.createElement('div');
    div.className = 'message user-msg';
    div.textContent = msg;
    chatHistory.appendChild(div);
    if (isAtBottom) chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function addSystemMessage(msg) {
    const isAtBottom = chatHistory.scrollHeight - chatHistory.scrollTop <= chatHistory.clientHeight + 150;
    const div = document.createElement('div');
    div.className = 'message system-msg';
    div.textContent = msg;
    chatHistory.appendChild(div);
    if (isAtBottom) chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function updateActiveAgentMessage(content, isThought = false) {
    const isAtBottom = chatHistory.scrollHeight - chatHistory.scrollTop <= chatHistory.clientHeight + 150;

    if (!activeAgentMsgDiv) {
      activeAgentMsgDiv = document.createElement('div');
      activeAgentMsgDiv.className = 'message agent-msg';
      activeAgentMsgDiv.dataset.id = currentMsgId || generateUUID();

      activeAgentThoughtDiv = document.createElement('div');
      activeAgentThoughtDiv.className = 'agent-thought';
      activeAgentMsgDiv.appendChild(activeAgentThoughtDiv);

      activeAgentContentDiv = document.createElement('div');
      activeAgentContentDiv.className = 'agent-content';
      activeAgentMsgDiv.appendChild(activeAgentContentDiv);

      activeThinkingIndicator = document.createElement('div');
      activeThinkingIndicator.className = 'thinking-indicator';
      activeThinkingIndicator.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Agent is thinking...';
      activeAgentMsgDiv.appendChild(activeThinkingIndicator);

      chatHistory.appendChild(activeAgentMsgDiv);
    }

    if (isThought) {
      appendTextWithBreaks(activeAgentThoughtDiv, content);
    } else {
      // Remove thinking indicator when response starts arriving
      if (activeThinkingIndicator) {
        activeThinkingIndicator.remove();
        activeThinkingIndicator = null;
      }

      appendTextWithBreaks(activeAgentContentDiv, content);

      // Requirement: Only show for Complex tasks with Memory Hit AND Innovation 3 enabled
      if (!activeAgentMsgDiv.querySelector('.feedback-btns') && currentMemoryHit && toggleFailure.checked) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-btns';
        const feedbackMsgId = activeAgentMsgDiv.dataset.id;
        const satisfiedBtn = document.createElement('button');
        satisfiedBtn.className = 'feedback-btn';
        satisfiedBtn.type = 'button';
        satisfiedBtn.innerHTML = '<i class="ph ph-thumbs-up"></i> Satisfied';
        satisfiedBtn.addEventListener('click', event => window.sendFeedback(feedbackMsgId, 'satisfied', event));
        const unsatisfiedBtn = document.createElement('button');
        unsatisfiedBtn.className = 'feedback-btn';
        unsatisfiedBtn.type = 'button';
        unsatisfiedBtn.innerHTML = '<i class="ph ph-thumbs-down"></i> Unsatisfied';
        unsatisfiedBtn.addEventListener('click', event => window.sendFeedback(feedbackMsgId, 'unsatisfied', event));
        feedbackDiv.appendChild(satisfiedBtn);
        feedbackDiv.appendChild(unsatisfiedBtn);
        activeAgentMsgDiv.appendChild(feedbackDiv);
      }
    }

    if (isAtBottom) chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  window.sendFeedback = async (msgId, status, event) => {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'feedback', msg_id: msgId, status: status }));
      } else {
        await fetch(`${API_BASE}/api/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ msg_id: msgId, status: status })
        });
      }
      const btn = event.currentTarget;
      btn.parentElement.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      addSystemMessage(`Feedback received: ${status}`);
      saveChatState();
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };

  function updateState(complexity, memoryHit) {
    // Simplified to show Device/Cloud based on decision
    if (complexity === 'Device' || complexity === 'Cloud') {
      stateComplexity.textContent = complexity;
      stateComplexity.className = `state-value highlight ${complexity.toLowerCase()}`;
    } else {
      // Legacy support or fallback
      const label = (complexity === 'Simple' || (complexity === 'Complex' && memoryHit)) ? 'Device' : (complexity === 'Complex' ? 'Cloud' : (complexity || 'Idle'));
      stateComplexity.textContent = label;
      stateComplexity.className = `state-value highlight ${label.toLowerCase()}`;
    }
  }

  function updateMemoryQueue(items) {
    memoryQueueList.innerHTML = '';
    if (!items || items.length === 0) {
      memoryQueueList.appendChild(createTextDiv('empty-queue', 'Queue is empty'));
      return;
    }
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'queue-item';

      const isUser = item.role === 'user';
      const roleStr = isUser ? 'User' : 'Agent';
      const roleClass = isUser ? 'role-user' : 'role-assistant';
      const iconClass = isUser ? 'ph ph-user' : 'ph ph-robot';

      const header = document.createElement('div');
      header.className = 'queue-header';
      const icon = document.createElement('i');
      icon.className = `${iconClass} ${roleClass}`;
      const role = document.createElement('span');
      role.className = `queue-role ${roleClass}`;
      role.textContent = roleStr;
      const content = document.createElement('div');
      content.className = 'queue-content';
      const text = String(item.content || '');
      content.textContent = text.length > 80 ? `${text.substring(0, 80)}...` : text;
      header.appendChild(icon);
      header.appendChild(role);
      div.appendChild(header);
      div.appendChild(content);
      memoryQueueList.appendChild(div);
    });
  }

  function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    if (isProcessing) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      addSystemMessage('Backend is not connected. Please wait for reconnection.');
      return;
    }
    addUserMessage(msg);
    chatInput.value = '';
    currentMsgId = null; // Reset for new query
    currentMemoryHit = false; // Reset for new query
    activeAgentMsgDiv = null;
    setProcessing(true);
    ws.send(JSON.stringify({ type: 'query', text: msg }));
  }

  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // --- Initial Init ---
  window.addEventListener('beforeunload', () => {
    isUnloading = true;
    saveChatStateBeforeUnload();
  });

  // Try to load the latest history item if exists, else init new
  fetch(`${API_BASE}/api/history`).then(res => res.json()).then(list => {
    if (list.length > 0) loadConversation(list[0].id);
    else initNewChat();
  }).catch(() => initNewChat());

  connectWebSocket();
});
