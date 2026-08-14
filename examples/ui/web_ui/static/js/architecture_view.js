/* ============================================================
   GAJE Architecture View (embebible)
   Renderiza architecture_graph.json como diagrama SVG interactivo
   dentro del portal (pestaña "Arquitectura").
   Fuente de datos única: architecture_graph.json
   ============================================================ */
(function (global) {
    'use strict';

    const NS = 'http://www.w3.org/2000/svg';
    const NODE_W = 132, NODE_H = 46, GAP = 16;
    const LAYER_Y = { ui: 56, python: 180, bridge: 304, rust_core: 418, kernels: 548, io: 668, data: 760 };

    let graph = null;          // datos cargados
    let pos = {};
    let graphW = 0, graphH = 0;
    let nodeEls = {}, edgeEls = {};
    let activeFlow = null, activeColor = '#8b5cf6';
    let scale = 1;

    function el(tag, attrs) {
        const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        return e;
    }

    function layout(container) {
        const rows = {};
        graph.nodes.forEach(n => { (rows[n.layer] = rows[n.layer] || []).push(n.id); });
        let maxW = 0;
        Object.keys(rows).forEach(layer => {
            const ids = rows[layer];
            maxW = Math.max(maxW, ids.length * NODE_W + (ids.length - 1) * GAP);
        });
        const centerX = maxW / 2;
        Object.keys(rows).forEach(layer => {
            const ids = rows[layer];
            const total = ids.length * NODE_W + (ids.length - 1) * GAP;
            let x = centerX - total / 2;
            ids.forEach(id => {
                pos[id] = { x: x, y: LAYER_Y[layer] - NODE_H / 2 };
                x += NODE_W + GAP;
            });
        });
        const maxY = Math.max.apply(null, Object.keys(LAYER_Y).map(k => LAYER_Y[k])) + NODE_H / 2 + 40;
        graphW = maxW + 40;
        graphH = maxY;
    }

    function renderBands(svg, layers) {
        Object.keys(LAYER_Y).forEach(k => {
            const y = LAYER_Y[k];
            const rect = el('rect', { x: -10, y: y - NODE_H / 2 - 16, width: graphW + 20, height: NODE_H + 32, rx: 14, class: 'arch-band', fill: 'currentColor', opacity: 0.012 });
            svg.appendChild(rect);
            const label = el('text', { x: 10, y: y - NODE_H / 2 - 24, class: 'arch-layer-label', 'text-anchor': 'start' });
            const L = layers.find(l => l.id === k);
            label.textContent = (L ? L.label : k).toUpperCase();
            svg.appendChild(label);
        });
    }

    function renderEdges(svg) {
        edgeEls = {};
        graph.edges.forEach(e => {
            const a = pos[e.from], b = pos[e.to];
            if (!a || !b) return;
            const x1 = a.x + NODE_W / 2, y1 = a.y + NODE_H / 2;
            const x2 = b.x + NODE_W / 2, y2 = b.y + NODE_H / 2;
            const mid = y1 + (y2 - y1) / 2;
            const path = el('path', { d: `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`, class: 'arch-edge', fill: 'none' });
            svg.appendChild(path);
            edgeEls[`${e.from}->${e.to}`] = path;
            const tan_ang = Math.atan2(y2 - mid, x2 - (x1 + x2) / 2);
            const ah = el('path', { d: 'M0 0 L7 3 L0 6 Z', class: 'arch-edge-arrow', transform: `translate(${x2},${y2}) rotate(${(tan_ang * 180) / Math.PI})` });
            svg.appendChild(ah);
        });
    }

    function renderNodes(svg, tooltip, pane) {
        nodeEls = {};
        graph.nodes.forEach(n => {
            const p = pos[n.id];
            const g = el('g', { class: 'arch-node', transform: `translate(${p.x},${p.y})` });
            g.dataset.id = n.id;
            const body = el('rect', { class: 'arch-node-body', width: NODE_W, height: NODE_H, rx: 10 });
            const icon = el('text', { class: 'arch-node-icon', x: 20, y: NODE_H / 2 + 6 });
            icon.textContent = n.icon || '•';
            const label = el('text', { class: 'arch-node-label', x: NODE_W / 2 + 6, y: NODE_H / 2 - 3 });
            label.textContent = n.label;
            const kind = el('text', { class: 'arch-node-kind', x: NODE_W / 2 + 6, y: NODE_H / 2 + 15 });
            kind.textContent = n.kind;
            g.appendChild(body); g.appendChild(icon); g.appendChild(label); g.appendChild(kind);
            g.addEventListener('mouseenter', () => showTooltip(n, g, tooltip, pane));
            g.addEventListener('mouseleave', hideTooltip);
            svg.appendChild(g);
            nodeEls[n.id] = g;
        });
    }

    function showTooltip(node, g, tooltip, pane) {
        const layerName = (graph.meta.layers.find(L => L.id === node.layer) || {}).label || node.layer;
        tooltip.innerHTML = `
            <div class="arch-t-title"><span>${node.icon || ''}</span> ${node.label}</div>
            <div class="arch-t-path">${node.fullname}</div>
            <div class="arch-t-desc">${node.description}</div>
            <span class="arch-t-layer">${layerName}</span>`;
        tooltip.classList.add('show');
        const paneRect = pane.getBoundingClientRect();
        const r = g.getBoundingClientRect();
        let left = r.left - paneRect.left + r.width + 12;
        if (left + 300 > paneRect.width) left = r.left - paneRect.left - 312;
        tooltip.style.left = left + 'px';
        tooltip.style.top = (r.top - paneRect.top) + 'px';
    }

    function hideTooltip() {
        const t = document.querySelector('.arch-tooltip');
        if (t) t.classList.remove('show');
    }

    function setFlow(flowId) {
        activeFlow = flowId;
        const flow = graph.flows.find(f => f.id === flowId);
        activeColor = flow ? flow.color : '#8b5cf6';
        const root = document.querySelector('#arch-view');
        if (root) root.style.setProperty('--flow', activeColor);

        Object.values(nodeEls).forEach(g => g.classList.remove('flow-on', 'dim'));
        Object.keys(edgeEls).forEach(k => edgeEls[k].classList.remove('flow-on', 'dim'));
        document.querySelectorAll('.arch-edge-arrow').forEach(a => a.classList.remove('flow-on', 'dim'));

        const status = document.querySelector('#arch-status-text');
        if (flowId) {
            const set = new Set(flow.steps);
            const edgeSet = new Set();
            for (let i = 0; i < flow.steps.length - 1; i++) edgeSet.add(`${flow.steps[i]}->${flow.steps[i + 1]}`);
            graph.nodes.forEach(n => {
                if (set.has(n.id)) nodeEls[n.id].classList.add('flow-on');
                else nodeEls[n.id].classList.add('dim');
            });
            Object.keys(edgeEls).forEach(k => {
                if (edgeSet.has(k)) edgeEls[k].classList.add('flow-on');
                else edgeEls[k].classList.add('dim');
            });
            if (status) status.textContent = 'Flujo: ' + flow.name;
        } else {
            if (status) status.textContent = 'Diagrama cargado';
        }
        renderFlowList();
    }

    function flowIcon(id) {
        if (id && /^i-/.test(id)) {
            return `<svg class="arch-f-icon y2k-icon" aria-hidden="true"><use href="static/icons/y2k/sprite.svg#${id}"/></svg>`;
        }
        return `<span class="arch-f-icon">${id || ''}</span>`;
    }

    function renderFlowList() {
        const list = document.querySelector('#arch-flow-list');
        if (!list) return;
        list.innerHTML = '';
        graph.flows.forEach(f => {
            const item = document.createElement('div');
            item.className = 'arch-flow-item' + (f.id === activeFlow ? ' active' : '');
            item.dataset.id = f.id;
            item.style.setProperty('--flow', f.color);
            const chips = f.steps.map(s => `<span class="arch-f-step">${s}</span>`).join('');
            item.innerHTML = `
                <div class="arch-f-head">
                    ${flowIcon(f.icon)}
                    <span class="arch-f-name">${f.name}</span>
                    <span class="arch-f-count">${f.steps.length} nodos</span>
                </div>
                <div class="arch-f-desc">${f.description}</div>
                <div class="arch-f-steps">${chips}</div>`;
            item.addEventListener('click', () => setFlow(f.id));
            list.appendChild(item);
        });
    }

    function renderLegend() {
        const legend = document.querySelector('#arch-legend');
        if (!legend) return;
        const seen = {};
        graph.flows.forEach(f => seen[f.color] = f);
        let html = '';
        Object.keys(seen).forEach(c => {
            html += `<span><span class="arch-sw" style="background:${c}"></span>${seen[c].name.split(' ')[0]}</span>`;
        });
        legend.innerHTML = html;
    }

    function initZoom(svg, pane) {
        svg.style.transformOrigin = '0 0';
        function apply() { svg.style.transform = `scale(${scale})`; }
        const zIn = document.querySelector('#arch-z-in');
        const zOut = document.querySelector('#arch-z-out');
        if (zIn) zIn.onclick = () => { scale = Math.min(2.2, scale * 1.15); apply(); };
        if (zOut) zOut.onclick = () => { scale = Math.max(0.5, scale * 0.87); apply(); };
        pane.addEventListener('wheel', e => {
            e.preventDefault();
            scale = Math.min(2.2, Math.max(0.5, scale * (e.deltaY < 0 ? 1.1 : 0.9)));
            apply();
        }, { passive: false });
    }

    function mount(rootEl) {
        if (!rootEl || graph) return;
        graph = null;
        fetch('architecture_graph.json')
            .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(data => {
                graph = data;
                rootEl.innerHTML = '';
                const pane = document.createElement('div');
                pane.className = 'arch-diagram-pane';
                const svg = document.createElementNS(NS, 'svg');
                svg.setAttribute('id', 'arch-svg');
                svg.setAttribute('viewBox', '0 0 1000 800');
                svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
                const tooltip = document.createElement('div');
                tooltip.className = 'arch-tooltip';
                const zoom = document.createElement('div');
                zoom.className = 'arch-zoom';
                zoom.innerHTML = '<button class="arch-zoom-btn" id="arch-z-in" title="Acercar">+</button><button class="arch-zoom-btn" id="arch-z-out" title="Alejar">−</button>';
                pane.appendChild(svg); pane.appendChild(tooltip); pane.appendChild(zoom);

                const side = document.createElement('aside');
                side.className = 'arch-flow-pane';
                side.innerHTML = `
                    <div class="arch-flow-head">
                        <h3>Flujos del sistema</h3>
                        <span id="arch-status-text" class="arch-status-text">Diagrama cargado</span>
                    </div>
                    <div class="arch-hint">Selecciona un flujo para resaltar su ruta en el diagrama.</div>
                    <div class="arch-legend" id="arch-legend"></div>
                    <div id="arch-flow-list"></div>`;

                rootEl.appendChild(pane);
                rootEl.appendChild(side);

                layout(pane);
                svg.setAttribute('viewBox', `${-20} ${-20} ${graphW + 40} ${graphH + 40}`);
                renderBands(svg, graph.meta.layers);
                renderEdges(svg);
                renderNodes(svg, tooltip, pane);
                renderLegend();
                renderFlowList();
                initZoom(svg, pane);
                setFlow('inference');
            })
            .catch(err => {
                rootEl.innerHTML = `<div class="arch-error">No se pudo cargar el diagrama: ${err.message}</div>`;
            });
    }

    // API pública para que script.js resalte flujos contextualmente
    global.ArchView = {
        mount: mount,
        setFlow: setFlow,
        isLoaded: function () { return !!graph; }
    };

    // Auto-mount: si existe un elemento #arch-view en la página (página
    // standalone architecture.html), montar el diagrama automáticamente.
    document.addEventListener('DOMContentLoaded', function () {
        const root = document.querySelector('#arch-view');
        if (root && root.getAttribute('data-mount') !== 'manual') {
            mount(root);
        }
    });
})(window);
