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
            modelsData = data.models;
            modelSelect.innerHTML = '';

            if (modelsData.length === 0) {
                const opt = document.createElement('option');
                opt.value = 'none';
                opt.innerText = 'No se encontraron modelos .gaje';
                modelSelect.appendChild(opt);
            } else {
                modelsData.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    opt.innerText = model.name.replace('.gaje', '').replace(/_/g, ' ').toUpperCase();
                    modelSelect.appendChild(opt);
                });
                updateModelDate();
            }
        } catch (err) {
            console.error('Error cargando modelos:', err);
        }
    }

    function updateModelDate() {
        const selected = modelSelect.value;
        const model = modelsData.find(m => m.name === selected);
        if (model) {
            modelDate.innerText = `Nacido el: ${model.date}`;
        }
    }

    modelSelect.addEventListener('change', updateModelDate);
    loadModels();

    function addMessage(text, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        msgDiv.innerHTML = `<p>${text}</p>`;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateMetrics(metrics) {
        metricsContent.innerHTML = `
            <div class="metric-row"><span>Dims:</span> <span class="metric-val">${metrics.dims}</span></div>
            <div class="metric-row"><span>Original:</span> <span class="metric-val">${metrics.original_size}B</span></div>
            <div class="metric-row"><span>DNA Size:</span> <span class="metric-val">${metrics.dna_size}B</span></div>
            <div class="metric-row"><span>Ratio:</span> <span class="metric-val">${metrics.ratio.toFixed(1)}x</span></div>
            <div class="metric-row"><span>Ahorro:</span> <span class="metric-val">${metrics.saved.toFixed(2)}%</span></div>
        `;
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
                addMessage(data.response, 'bot');
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
