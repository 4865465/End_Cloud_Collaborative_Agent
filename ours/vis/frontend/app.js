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
  const stateMemory = document.getElementById('state-memory');

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

  const API_BASE = 'http://127.0.0.1:8002';

  // --- Helper: Generate UUID ---
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // --- Persistence Logic (Backend API) ---
  async function saveChatState() {
    if (!currentChatId) return;

    const state = {
      id: currentChatId,
      title: chatHistory.querySelector('.user-msg')?.textContent?.substring(0, 30) || 'New Conversation',
      timestamp: new Date().toISOString(),
      historyHTML: chatHistory.innerHTML,
      memoryItems: shortTermMemory,
      agentType: agentType.value,
      stats: {
        totalLlmCost,
        totalSearchCount,
        totalCodeCost
      }
    };

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

  async function renderHistory() {
    try {
      const response = await fetch(`${API_BASE}/api/history`);
      const historyList = await response.json();

      historyListContainer.innerHTML = '';
      if (historyList.length === 0) {
        historyListContainer.innerHTML = '<div class="empty-history">No history yet</div>';
        return;
      }

      historyList.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        if (item.id === currentChatId) div.classList.add('active');

        div.innerHTML = `
          <div class="history-info" title="${item.title}">
            <div style="font-size: 0.7rem; color: var(--text-muted);">${new Date(item.timestamp).toLocaleString()}</div>
            <div>${item.title}</div>
          </div>
          <button class="delete-btn" onclick="event.stopPropagation(); window.deleteChatMessage('${item.id}')">
            <i class="ph ph-trash"></i>
          </button>
        `;
        div.onclick = () => loadConversation(item.id);
        historyListContainer.appendChild(div);
      });
    } catch (e) {
      console.error("Failed to fetch history", e);
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
      const item = await response.json();
      if (item.error) return;

      currentChatId = item.id;
      chatHistory.innerHTML = item.historyHTML;

      totalLlmCost = item.stats.totalLlmCost || 0;
      totalSearchCount = item.stats.totalSearchCount || 0;
      totalCodeCost = item.stats.totalCodeCost || 0;

      if (item.memoryItems) {
        shortTermMemory = item.memoryItems;
        updateMemoryQueue(shortTermMemory);
      }

      // Restore stats to UI
      totalCostLlm.textContent = `¥${totalLlmCost.toFixed(5)}`;
      totalSearch.textContent = totalSearchCount;
      totalCostCode.textContent = `¥${totalCodeCost.toFixed(5)}`;

      // Sync with backend runner
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'load_conversation',
          data: {
            memory: shortTermMemory,
            stats: item.stats,
            agent_type: item.agentType || 'react'
          }
        }));
        
        // Also update UI and backend config if needed
        if (item.agentType && agentType.value !== item.agentType) {
          agentType.value = item.agentType;
          applyConfig();
        }
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
    const config = {
      agent_type: agentType.value,
      memory_size: parseInt(memorySizeInput.value) || 5,
      innovations: {
        retrieval: toggleRetrieval.checked,
        memory: toggleMemory.checked,
        failure: toggleFailure.checked
      },
      thresholds: {
        t1: parseFloat(thresholdT1.value) || 0.9,
        t2: parseFloat(thresholdT2.value) || 0.6,
        sim: parseFloat(thresholdSim.value) || 0.9
      }
    };
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
      const response = await fetch(`${API_BASE}/api/library`);
      const data = await response.json();
      renderLibItems(data.edge_tool, edgeToolList, 'No Edge Tool Records', true);
      renderLibItems(data.cloud_tool, cloudToolList, 'No Cloud Tool Records', true);
      renderLibItems(data.exp_db, expDbList, 'No Experience Records', false);
    } catch (error) {
      edgeToolList.innerHTML = 'Error loading library.';
    }
  }

  function renderLibItems(recordsObj, container, emptyMsg, isTool = true) {
    container.innerHTML = '';
    const keys = Object.keys(recordsObj);
    if (keys.length === 0) {
      container.innerHTML = `<div class="empty-queue">${emptyMsg}</div>`;
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
          itemDiv.innerHTML = `<div style="color:var(--primary); font-weight:bold;">[${key}]</div>
                               <div style="color:var(--text-muted);">${rec.tool_input || rec.input || ''}</div>
                               <div style="font-style:italic;">${(rec.tool_output || rec.output || '').substring(0, 100)}...</div>`;
          div.appendChild(itemDiv);
        });
      } else {
        div.innerHTML = `<div style="color:var(--primary); font-weight:bold;">${key}</div>
                         <div style="font-style:italic;">${(entry.experience || '').substring(0, 150)}...</div>`;
      }
      container.appendChild(div);
    });
  }

  // --- WebSocket Logic ---
  function connectWebSocket() {
    ws = new WebSocket('ws://127.0.0.1:8002/ws/agent');
    ws.onopen = () => {
      addSystemMessage('Connected to Edge-Cloud Agent Backend.');
      // Auto-set defaults
      ws.send(JSON.stringify({
        type: 'config',
        data: {
          agent_type: agentType.value,
          memory_size: parseInt(memorySizeInput.value),
          innovations: {
            retrieval: toggleRetrieval.checked,
            memory: toggleMemory.checked,
            failure: toggleFailure.checked
          },
          thresholds: {
            t1: parseFloat(thresholdT1.value),
            t2: parseFloat(thresholdT2.value),
            sim: parseFloat(thresholdSim.value)
          }
        }
      }));
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
      addSystemMessage('Disconnected. Reconnecting...');
      setTimeout(connectWebSocket, 3000);
    };
  }

  function handleAgentEvent(data) {
    if (data.type === 'msg_id') {
      currentMsgId = data.msg_id;
    } else if (data.type === 'thought') {
      updateActiveAgentMessage(data.content, true);
    } else if (data.type === 'response') {
      updateActiveAgentMessage(data.content, false);
      saveChatState();
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
      saveChatState();
    }
  }

  function updateCost(data) {
    if (data.llm_cost !== undefined && costLlm) costLlm.textContent = `¥${data.llm_cost.toFixed(5)}`;
    if (data.total_llm_cost !== undefined && totalCostLlm) {
      totalLlmCost = data.total_llm_cost;
      totalCostLlm.textContent = `¥${totalLlmCost.toFixed(5)}`;
    }
    if (data.search_count !== undefined && costSearch) {
      currentSearchCount = data.search_count;
      costSearch.textContent = currentSearchCount;
    }
    if (data.total_search_count !== undefined && totalSearch) {
      totalSearchCount = data.total_search_count;
      totalSearch.textContent = totalSearchCount;
    }
    if (data.code_cost !== undefined && costCode) {
      currentCodeCost = data.code_cost;
      costCode.textContent = `¥${currentCodeCost.toFixed(5)}`;
    }
    if (data.total_code_cost !== undefined && totalCostCode) {
      totalCodeCost = data.total_code_cost;
      totalCostCode.textContent = `¥${totalCodeCost.toFixed(5)}`;
    }
    if (data.time_spent !== undefined && costTime) costTime.textContent = `${data.time_spent.toFixed(2)} s`;
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
      activeAgentThoughtDiv.innerHTML += content.replace(/\n/g, '<br>');
    } else {
      // Remove thinking indicator when response starts arriving
      if (activeThinkingIndicator) {
        activeThinkingIndicator.remove();
        activeThinkingIndicator = null;
      }

      activeAgentContentDiv.innerHTML += content.replace(/\n/g, '<br>');

      // Requirement: Only show for Complex tasks with Memory Hit AND Innovation 3 enabled
      if (!activeAgentMsgDiv.querySelector('.feedback-btns') && currentMemoryHit && toggleFailure.checked) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-btns';
        feedbackDiv.innerHTML = `
            <button class="feedback-btn" onclick="window.sendFeedback('${activeAgentMsgDiv.dataset.id}', 'satisfied', event)">
                <i class="ph ph-thumbs-up"></i> Satisfied
            </button>
            <button class="feedback-btn" onclick="window.sendFeedback('${activeAgentMsgDiv.dataset.id}', 'unsatisfied', event)">
                <i class="ph ph-thumbs-down"></i> Unsatisfied
            </button>
          `;
        activeAgentMsgDiv.appendChild(feedbackDiv);
      }
    }

    if (isAtBottom) chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  window.sendFeedback = async (msgId, status, event) => {
    try {
      await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ msg_id: msgId, status: status })
      });
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
      memoryQueueList.innerHTML = '<div class="empty-queue">Queue is empty</div>';
      return;
    }
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'queue-item';

      const isUser = item.role === 'user';
      const roleStr = isUser ? 'User' : 'Agent';
      const roleClass = isUser ? 'role-user' : 'role-assistant';
      const iconClass = isUser ? 'ph ph-user' : 'ph ph-robot';

      div.innerHTML = `
        <div class="queue-header">
          <i class="${iconClass} ${roleClass}"></i>
          <span class="queue-role ${roleClass}">${roleStr}</span>
        </div>
        <div class="queue-content">${item.content.length > 80 ? item.content.substring(0, 80) + '...' : item.content}</div>
      `;
      memoryQueueList.appendChild(div);
    });
  }

  function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    addUserMessage(msg);
    chatInput.value = '';
    currentMsgId = null; // Reset for new query
    if (ws && ws.readyState === WebSocket.OPEN) {
      currentMemoryHit = false; // Reset for new query
      ws.send(JSON.stringify({ type: 'query', text: msg }));
      activeAgentMsgDiv = null;
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
  });

  // --- Initial Init ---
  window.addEventListener('beforeunload', saveChatState);

  // Try to load the latest history item if exists, else init new
  fetch(`${API_BASE}/api/history`).then(res => res.json()).then(list => {
    if (list.length > 0) loadConversation(list[0].id);
    else initNewChat();
  }).catch(() => initNewChat());

  connectWebSocket();
});
