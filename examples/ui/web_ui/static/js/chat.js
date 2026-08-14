document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const metricsContent = document.getElementById('metrics-content');
    const dnaStrand = document.getElementById('dna-strand');
    const modelSelect = document.getElementById('model-select');
    const modelDate = document.getElementById('model-date');
    const modelSize = document.getElementById('model-size');
    const modelRam = document.getElementById('model-ram');
    const modelLoadBar = document.getElementById('model-load-bar');
    let modelsData = [];

    // Cargar modelos disponibles
    async function loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            if (data && data.models && data.models.length > 0) {
                modelsData = data.models;
                modelSelect.innerHTML = '';
                modelsData.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    let label = model.name;
                    if (label.endsWith('.flat')) {
                        label = '⚡ ' + label.replace('.gaje.flat', '').replace('.flat', '') + ' (ZERO-COPY FLAT MMAP)';
                    } else {
                        label = label.replace('.gaje', '');
                    }
                    opt.innerText = label.replace(/_/g, ' ').toUpperCase();
                    modelSelect.appendChild(opt);
                });
                updateModelMeta();
                if (modelSelect.value) {
                    preloadModel(modelSelect.value);
                }
            }
        } catch (err) {
            console.log('Usando modelos por defecto pre-configurados.');
        }
    }

    // Detectar entorno real de ejecución (arquitectura, CPU, SIMD, Island)
    async function loadEnvInfo() {
        try {
            const response = await fetch('/api/info');
            const info = await response.json();
            if (!info || info.error) return;
            document.getElementById('sf-val').innerText = info.software || '---';
            document.getElementById('hd-val').innerText = info.hardware || '---';
            if (info.architecture) document.getElementById('arch-val').innerText = info.architecture;
            if (info.simd) document.getElementById('simd-val').innerText = info.simd;
            if (info.cores) document.getElementById('cores-val').innerText = info.cores;
            const status = document.querySelector('.status-text');
            if (status && info.simd) status.innerText = info.simd + ' Optimized';

            // Island Model (.gmem) — valores desde el servidor, no hardcodeados
            if (info.island) {
                const pillsEl = document.getElementById('island-pills');
                if (pillsEl) {
                    pillsEl.innerHTML = (info.island.pills || [])
                        .map(p => `<span class="island-pill">${p}</span>`)
                        .join('');
                }
                if (info.island.memory_type) document.getElementById('island-mem-val').innerText = info.island.memory_type;
                if (info.island.retrieval_latency_ms != null) document.getElementById('island-lat-val').innerText = `${info.island.retrieval_latency_ms} ms`;
                if (info.island.context_budget != null) document.getElementById('island-budget-val').innerText = `${info.island.context_budget} tokens`;
            }
        } catch (err) {
            console.log('No se pudo detectar el entorno de ejecución.');
        }
    }

    function formatBytes(bytes) {
        if (!bytes && bytes !== 0) return '—';
        const gb = bytes / (1024 * 1024 * 1024);
        if (gb >= 1) return gb.toFixed(2) + ' GB';
        const mb = bytes / (1024 * 1024);
        if (mb >= 1) return mb.toFixed(0) + ' MB';
        return Math.round(bytes) + ' B';
    }

    function updateModelMeta() {
        const selected = modelSelect.value;
        const model = modelsData.find(m => m.name === selected);
        if (!model) return;
        if (model.date) modelDate.innerText = `Nacido el: ${model.date}`;
        if (model.size_bytes != null) modelSize.innerText = `Peso HD: ${formatBytes(model.size_bytes)}`;
        modelRam.innerText = `RAM: ${(model.ram_mb || 0) > 0 ? model.ram_mb.toFixed(1) + ' MB' : '—'}`;
    }

    async function refreshModelMeta(modelName) {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            if (data && data.models && data.models.length > 0) {
                modelsData = data.models;
                updateModelMeta();
            }
        } catch (err) {
            console.log('No se pudo refrescar los metadatos del modelo.');
        }
    }

    function setModelLoading(active) {
        if (!modelLoadBar) return;
        if (active) {
            modelLoadBar.hidden = false;
            modelLoadBar.setAttribute('aria-valuetext', 'cargando');
            modelLoadBar.setAttribute('aria-valuenow', '');
        } else {
            modelLoadBar.hidden = true;
            modelLoadBar.setAttribute('aria-valuetext', 'inactivo');
        }
        if (chatWindow) chatWindow.setAttribute('aria-busy', active ? 'true' : 'false');
    }

    async function preloadModel(modelName) {
        if (!modelName) return;

        modelSelect.disabled = true;
        userInput.disabled = true;
        sendBtn.disabled = true;

        updateModelMeta();
        setModelLoading(true);
        addMessage(`🧬 Cargando organismo genómico [${modelName}] en el servidor... Por favor espera.`, 'system');

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelName })
            });

            const data = await response.json();
            if (data.status === 'ok') {
                addMessage(`✅ Organismo [${modelName}] cargado y listo en memoria.`, 'system');
                await refreshModelMeta(modelName); // refrescar RAM real tras la carga
            } else {
                addMessage(`❌ Error cargando el modelo: ${data.error}`, 'bot');
            }
        } catch (err) {
            addMessage(`❌ Error de conexión al cargar [${modelName}].`, 'bot');
            console.error(err);
        } finally {
            setModelLoading(false);
            modelSelect.disabled = false;
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            // Actualizar el entorno real de ejecución al terminar de cargar el modelo
            loadEnvInfo();
        }
    }

    modelSelect.addEventListener('change', () => {
        preloadModel(modelSelect.value);
    });
    loadModels();
    loadEnvInfo();

    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            clearHistory();
            const messages = chatWindow.querySelectorAll('.message');
            messages.forEach(m => { if (!m.classList.contains('system')) m.remove(); });
        });
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    function addMessage(text, type, meta = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        if (type === 'bot' && /^❌|^Error/.test(text)) {
            msgDiv.classList.add('error');
        }

        let html = `<p>${escapeHtml(text)}</p>`;
        if (type === 'bot' && meta) {
            let islandBadge = '';
            if (meta.island) {
                islandBadge = `<span class="meta-badge meta-island">🏝️ Island .gmem: ${escapeHtml(meta.island.retrieval_ms)} ms | +${escapeHtml(meta.island.budget_tokens)} tok (CosSim ${escapeHtml(meta.island.cossim)})</span>`;
            }
            html += `
                <div class="message-meta">
                    <span class="meta-badge">⏱️ ${escapeHtml(meta.latency_ms)} ms (${escapeHtml(meta.tokens_sec || 0)} tok/s)</span>
                    <span class="meta-badge">🔢 ${escapeHtml(meta.tokens_count || 0)} tokens</span>
                    ${islandBadge}
                </div>
            `;
        }

        msgDiv.innerHTML = html;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateMetrics(metrics) {
        const sizeLabel = metrics.bit_depth === 4 ? "Compressed:" : "DNA Size:";
        metricsContent.innerHTML = `
            <div class="metric-row"><span>Dims:</span> <span class="metric-val">${metrics.dims}</span></div>
            <div class="metric-row"><span>Original:</span> <span class="metric-val">${metrics.original_size}B</span></div>
            <div class="metric-row"><span>${sizeLabel}</span> <span class="metric-val">${metrics.dna_size}B (${metrics.bit_depth || 4}-bit)</span></div>
            <div class="metric-row"><span>Ratio:</span> <span class="metric-val">${metrics.ratio.toFixed(1)}x</span></div>
            <div class="metric-row"><span>Ahorro:</span> <span class="metric-val">${metrics.saved.toFixed(2)}%</span></div>
            <div class="progress-bar-container"><div class="progress-bar-fill" style="width: ${metrics.saved}%"></div></div>
            <div class="metric-row"><span>Tokens Usados:</span> <span class="metric-val">${metrics.tokens_count || 0} tok</span></div>
            <div class="metric-row"><span>Tiempo Resp:</span> <span class="metric-val">${metrics.latency_ms || 0} ms</span></div>
        `;

        if (metrics.sf_info) {
            document.getElementById('sf-val').innerText = metrics.sf_info;
        }
        if (metrics.hd_info) {
            document.getElementById('hd-val').innerText = metrics.hd_info;
        }
        if (metrics.latency_ms) {
            document.getElementById('latency-val').innerText = `${metrics.latency_ms} ms (${metrics.tokens_sec || 0} tok/s)`;
        }
    }

    function updateDNA(strand) {
        dnaStrand.innerHTML = '';
        strand.split('').forEach(base => {
            const span = document.createElement('span');
            span.className = `dna-char-${base}`;
            span.innerText = base;
            dnaStrand.appendChild(span);
        });
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        const modelSelect = document.getElementById('model-select');
        const modelValue = modelSelect.value;

        if (!text) return;
        if (!modelValue || modelValue === 'none' || modelValue === '') {
            addMessage('Por favor, selecciona un modelo válido.', 'bot');
            return;
        }

        addMessage(text, 'user');
        pushHistory({ role: 'user', content: text });
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Vínculo contextual (Fase 1): resaltar el flujo de inferencia en el diagrama
        if (window.ArchView && window.ArchView.isLoaded()) {
            window.ArchView.setFlow('inference');
        }

        const ok = await streamChat(text, modelValue);
        if (!ok) {
            await fallbackChat(text, modelValue);
        }

        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }

    // Fallback no-streaming con métricas (si el stream falla)
    async function fallbackChat(text, modelName) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, model: modelName })
            });
            const data = await response.json();
            if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else {
                addMessage(data.response, 'bot', data.metrics);
                pushHistory({ role: 'assistant', content: data.response });
                updateMetrics(data.metrics);
                updateDNA(data.dna);
            }
        } catch (err) {
            addMessage('Error de conexión con el núcleo GAJE.', 'bot');
            console.error(err);
        }
    }

    // ===== Streaming SSE (Fase 2.2) =====
    let abortController = null;

    function streamChat(message, modelName) {
        const botMsg = createBotMessage();
        botMsg.classList.add('streaming');
        const statusEl = document.createElement('span');
        statusEl.className = 'stream-status';
        statusEl.textContent = 'Generando';
        const statusAnchor = document.createElement('div');
        statusAnchor.className = 'stream-status-row';
        statusAnchor.appendChild(statusEl);
        botMsg.appendChild(statusAnchor);

        const contentEl = document.createElement('p');
        contentEl.className = 'stream-text';
        botMsg.appendChild(contentEl);
        chatWindow.appendChild(botMsg);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        abortController = new AbortController();
        const stopBtn = document.getElementById('stop-btn');
        stopBtn.hidden = false;
        if (chatWindow) chatWindow.setAttribute('aria-busy', 'true');

        let fullText = '';
        let started = Date.now();
        let done = false;

        const finish = (aborted) => {
            if (done) return;
            done = true;
            abortController = null;
            stopBtn.hidden = true;
            if (chatWindow) chatWindow.setAttribute('aria-busy', 'false');
            botMsg.classList.remove('streaming');
            const elapsed = Date.now() - started;
            if (aborted && fullText) {
                addMetaTo(botMsg, elapsed, '⏹️ detenido');
            } else if (!aborted) {
                addMetaTo(botMsg, elapsed);
            }
            if (fullText) pushHistory({ role: 'assistant', content: fullText });
            chatWindow.scrollTop = chatWindow.scrollHeight;
        };

        const onStop = () => {
            if (abortController) abortController.abort();
        };
        stopBtn.onclick = onStop;

        return fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, model: modelName }),
            signal: abortController.signal
        }).then(async (response) => {
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                botMsg.remove();
                addMessage(`Error: ${data.error || 'Fallo en el stream'}`, 'bot');
                finish(true);
                return false;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done: rdDone } = await reader.read();
                if (rdDone) break;
                buffer += decoder.decode(value, { stream: true });
                // parse SSE lines: data: ...
                let idx;
                while ((idx = buffer.indexOf('\n')) !== -1) {
                    const line = buffer.slice(0, idx).trim();
                    buffer = buffer.slice(idx + 1);
                    if (line.startsWith('data: ')) {
                        const payload = line.slice(6);
                        if (payload === '[DONE]') {
                            reader.releaseLock();
                            finish(false);
                            return true;
                        }
                        try {
                            const token = JSON.parse(payload);
                            if (token && typeof token === 'object' && token.error) {
                                throw new Error(token.error);
                            }
                            fullText += token;
                            contentEl.textContent = fullText;
                            chatWindow.scrollTop = chatWindow.scrollHeight;
                        } catch (e) {
                            // payload no JSON o error
                            if (e.message) {
                                botMsg.remove();
                                addMessage(`Error: ${e.message}`, 'bot');
                                finish(true);
                                return false;
                            }
                        }
                    }
                }
            }
            finish(false);
            return true;
        }).catch((err) => {
            if (err && err.name === 'AbortError') {
                finish(true);
                return true;
            }
            botMsg.remove();
            addMessage('Error de conexión con el núcleo GAJE (streaming).', 'bot');
            finish(true);
            return false;
        });
    }

    function createBotMessage() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot';
        return msgDiv;
    }

    function addMetaTo(msgEl, elapsed, prefix = '') {
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        meta.innerHTML = `<span class="meta-badge">⏱️ ${elapsed} ms ${prefix ? '(' + escapeHtml(prefix) + ')' : ''}</span>`;
        msgEl.appendChild(meta);
    }

    // ===== Historial local (Fase 2.3) =====
    const HISTORY_KEY = 'gaje_chat_history';

    function loadHistory() {
        try {
            const raw = localStorage.getItem(HISTORY_KEY);
            if (!raw) return [];
            const arr = JSON.parse(raw);
            return Array.isArray(arr) ? arr : [];
        } catch (e) {
            return [];
        }
    }

    function pushHistory(entry) {
        const arr = loadHistory();
        arr.push(entry);
        if (arr.length > 100) arr.splice(0, arr.length - 100);
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(arr));
        } catch (e) { /* almacenamiento lleno */ }
    }

    function clearHistory() {
        try {
            localStorage.removeItem(HISTORY_KEY);
        } catch (e) { /* ignore */ }
    }

    function renderHistory() {
        const arr = loadHistory();
        if (arr.length === 0) return;
        arr.forEach(entry => {
            if (entry.role === 'user') addMessage(entry.content, 'user');
            else if (entry.role === 'assistant') addMessage(entry.content, 'bot');
            else if (entry.role === 'system') addMessage(entry.content, 'system');
        });
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    renderHistory();
});

