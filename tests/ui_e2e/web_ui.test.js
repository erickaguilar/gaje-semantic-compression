const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('GAJE Helix Web UI — Formal E2E Test Suite', () => {
  const screenshotDir = path.join(__dirname, 'screenshots');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  test('1. Core Layout and Header Architecture', async ({ page }) => {
    await page.goto('/');

    // 1.1 Verificar Título de la Plataforma
    await expect(page).toHaveTitle(/GAJE Helix \| Genomic Semantic Compression Platform/);

    // 1.2 Verificar Marca y Logotipo Y2K
    const brand = page.locator('header .y2k-brand');
    await expect(brand).toBeVisible();
    await expect(brand).toContainText('GAJE');

    // 1.3 Verificar que el botón de menú hamburguesa esté visible y funcional
    const menuBtn = page.locator('#y2k-menu-btn');
    await expect(menuBtn).toBeVisible();

    // 1.4 Verificar el botón directo de alternancia de tema
    const directThemeBtn = page.locator('#direct-theme-toggle');
    await expect(directThemeBtn).toBeVisible();

    // 1.5 Verificar el área de entrada del chat
    const userInput = page.locator('#user-input');
    await expect(userInput).toBeVisible();
    await expect(userInput).toHaveAttribute('placeholder', /Escribe un mensaje/);
  });

  test('2. Three-Theme Selection System (Dark HIG, Light Scandinavian, Zen Focus)', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');

    // 2.1 Abrir menú hamburguesa
    const menuBtn = page.locator('#y2k-menu-btn');
    await menuBtn.click();

    const dropdown = page.locator('#y2k-menu-dropdown');
    await expect(dropdown).toBeVisible();

    // 2.2 Probar Tema Oscuro (Dark)
    const darkBtn = page.locator('.y2k-theme-opt-btn[data-theme-val="dark"]');
    await expect(darkBtn).toBeVisible();
    await darkBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'dark');
    await page.screenshot({ path: path.join(screenshotDir, 'theme-dark-hig.png') });

    // 2.3 Probar Tema Zen (Focus Mode)
    const zenBtn = page.locator('.y2k-theme-opt-btn[data-theme-val="zen"]');
    await expect(zenBtn).toBeVisible();
    await zenBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'zen');
    await page.screenshot({ path: path.join(screenshotDir, 'theme-zen-focus.png') });

    // 2.4 Probar Tema Claro (Light Scandinavian)
    const lightBtn = page.locator('.y2k-theme-opt-btn[data-theme-val="light"]');
    await expect(lightBtn).toBeVisible();
    await lightBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'light');
    await page.screenshot({ path: path.join(screenshotDir, 'theme-light-scandinavian.png') });

    // 2.5 Probar conmutación secuencial mediante botón rápido superior
    const directToggle = page.locator('#direct-theme-toggle');
    await directToggle.click(); // de light -> dark
    await expect(html).toHaveAttribute('data-theme', 'dark');
    await directToggle.click(); // de dark -> zen
    await expect(html).toHaveAttribute('data-theme', 'zen');
    await directToggle.click(); // de zen -> light
    await expect(html).toHaveAttribute('data-theme', 'light');
  });

  test('3. Chat Toolbar and Actions Dropdown Integration', async ({ page }) => {
    await page.goto('/');

    // 3.1 Abrir el menú de herramientas y acciones del chat (•••)
    const actionsBtn = page.locator('#chat-overflow-menu-btn');
    await expect(actionsBtn).toBeVisible();
    await actionsBtn.click();

    const actionsDropdown = page.locator('#chat-actions-dropdown');
    await expect(actionsDropdown).toBeVisible();

    // 3.2 Verificar selector de modelo
    const modelSelect = page.locator('#model-select');
    await expect(modelSelect).toBeVisible();

    // 3.3 Verificar botón de pantalla completa
    const fullscreenBtn = page.locator('#toggle-fullscreen-btn');
    await expect(fullscreenBtn).toBeVisible();
    await expect(fullscreenBtn).toContainText('Pantalla Completa');

    // 3.4 Verificar botón de exportar bitácora (.md)
    const exportLogBtn = page.locator('#export-log-btn');
    await expect(exportLogBtn).toBeVisible();
    await expect(exportLogBtn).toContainText('Exportar Bitácora (.md)');

    // 3.5 Verificar botón de borrar historial
    const clearHistoryBtn = page.locator('#clear-history-btn');
    await expect(clearHistoryBtn).toBeVisible();
    await expect(clearHistoryBtn).toContainText('Borrar Historial');
  });

  test('4. Starter Cards and Prompt Transfer', async ({ page }) => {
    await page.goto('/');

    // 4.1 Verificar que las tarjetas de inicio rápido existan
    const starterCards = page.locator('.starter-card');
    const count = await starterCards.count();
    expect(count).toBeGreaterThan(0);

    // 4.2 Hacer clic en la primera tarjeta de inicio
    const firstStarter = starterCards.first();
    const promptText = await firstStarter.getAttribute('data-prompt');
    await firstStarter.click();

    // 4.3 El texto debe haberse copiado al área de escritura
    const userInput = page.locator('#user-input');
    await expect(userInput).toHaveValue(promptText);

    // 4.4 El botón de enviar debe habilitarse
    const sendBtn = page.locator('#send-btn');
    await expect(sendBtn).toBeEnabled();
  });

  test('5. Minimalist Header Navigation Links', async ({ page }) => {
    await page.goto('/');

    // 5.1 Abrir menú de navegación
    const menuBtn = page.locator('#y2k-menu-btn');
    await menuBtn.click();

    const dropdown = page.locator('#y2k-menu-dropdown');
    await expect(dropdown).toBeVisible();

    // 5.2 Verificar enlaces de navegación en el menú
    const chatLink = dropdown.locator('a[href="index.html"]');
    const archLink = dropdown.locator('a[href="architecture.html"]');
    const docsLink = dropdown.locator('a[href="docs.html"]');

    await expect(chatLink).toBeVisible();
    await expect(archLink).toBeVisible();
    await expect(docsLink).toBeVisible();
  });
});
