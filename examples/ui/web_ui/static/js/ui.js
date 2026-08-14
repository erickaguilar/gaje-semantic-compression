/* ============================================================
   GAJE UI — utilidades compartidas entre páginas
   Reveal on scroll · Botones copiar · Toggle de tema
   ============================================================ */
(function (global) {
  'use strict';

  /* ── Reveal on scroll ── */
  function initReveal() {
    var els = document.querySelectorAll('.reveal');
    if (!els.length) return;
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (e) {
            if (e.isIntersecting) {
              e.target.classList.add('visible');
              io.unobserve(e.target);
            }
          });
        },
        { threshold: 0.12 }
      );
      els.forEach(function (el) { io.observe(el); });
    } else {
      els.forEach(function (el) { el.classList.add('visible'); });
    }
  }

  /* ── Copiar texto ── */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    var ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    return Promise.resolve();
  }

  function initCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var targetId = btn.getAttribute('data-target');
        var src = targetId === 'term'
          ? document.getElementById('term')
          : document.getElementById(targetId);
        if (!src) return;
        var old = btn.textContent;
        copyText(src.textContent).then(function () {
          btn.textContent = '✓ Copiado';
          setTimeout(function () { btn.textContent = old; }, 1600);
        });
      });
    });
  }

  /* ── Toggle de tema claro/oscuro ── */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-label', theme === 'light' ? 'Activar Tema Oscuro' : 'Activar Tema Claro');
    }
  }

  function initTheme() {
    var toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    applyTheme(localStorage.getItem('theme') || 'dark');
    toggle.addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', next);
      applyTheme(next);
    });
  }

  /* ── Header Y2K compartido (parcial) ── */
  function initHeader() {
    var host = document.getElementById('gaje-header');
    if (!host) return;
    fetch('static/partials/header.html')
      .then(function (r) { return r.text(); })
      .then(function (html) {
        host.innerHTML = html;
        var page = host.getAttribute('data-page') || 'index';
        var map = { index: 'index.html', chat: 'index.html', architecture: 'architecture.html', docs: 'docs.html' };
        var target = map[page] || page;
        var links = host.querySelectorAll('.y2k-nav a');
        links.forEach(function (a) {
          if (a.getAttribute('href') === target) {
            a.classList.add('active');
            a.setAttribute('aria-current', 'page');
          }
        });
        initTheme();
      })
      .catch(function (e) { console.warn('No se pudo cargar header parcial:', e); });
  }

  /* ── Boot ── */
  function boot() {
    initReveal();
    initCopyButtons();
    initTheme();
    initHeader();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  global.GajeUI = { initReveal: initReveal, initTheme: initTheme, initCopyButtons: initCopyButtons, initHeader: initHeader };
})(window);
