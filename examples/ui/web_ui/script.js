document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const metricsContent = document.getElementById('metrics-content');
    const dnaStrand = document.getElementById('dna-strand');
    const modelSelect = document.getElementById('model-select');
    const modelDate = document.getElementById('model-date');
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
                updateModelDate();
                if (modelSelect.value) {
                    preloadModel(modelSelect.value);
                }
            }
        } catch (err) {
            console.log('Usando modelos por defecto pre-configurados.');
        }
    }

    function updateModelDate() {
        const selected = modelSelect.value;
        const model = modelsData.find(m => m.name === selected);
        if (model) {
            modelDate.innerText = `Nacido el: ${model.date}`;
        }
    }

    async function preloadModel(modelName) {
        if (!modelName) return;

        modelSelect.disabled = true;
        userInput.disabled = true;
        sendBtn.disabled = true;

        updateModelDate();
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
            } else {
                addMessage(`❌ Error cargando el modelo: ${data.error}`, 'bot');
            }
        } catch (err) {
            addMessage(`❌ Error de conexión al cargar [${modelName}].`, 'bot');
            console.error(err);
        } finally {
            modelSelect.disabled = false;
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    modelSelect.addEventListener('change', () => {
        preloadModel(modelSelect.value);
    });
    loadModels();

    function addMessage(text, type, meta = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;

        let html = `<p>${text}</p>`;
        if (type === 'bot' && meta) {
            let islandBadge = '';
            if (meta.island) {
                islandBadge = `<span class="meta-badge meta-island">🏝️ Island .gmem: ${meta.island.retrieval_ms} ms | +${meta.island.budget_tokens} tok (CosSim ${meta.island.cossim})</span>`;
            }
            html += `
                <div class="message-meta">
                    <span class="meta-badge">⏱️ ${meta.latency_ms} ms (${meta.tokens_sec || 0} tok/s)</span>
                    <span class="meta-badge">🔢 ${meta.tokens_count || 0} tokens</span>
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
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    model: modelSelect.value
                })
            });

            const data = await response.json();

            if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else {
                addMessage(data.response, 'bot', data.metrics);
                updateMetrics(data.metrics);
                updateDNA(data.dna);
            }
        } catch (err) {
            addMessage('Error de conexión con el núcleo GAJE.', 'bot');
            console.error(err);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});
