const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('GAJE-Flow Visual Web UI Tests', () => {
  test('should load page and match main structures', async ({ page }) => {
    // 1. Navegar a la página
    await page.goto('/');

    // 2. Verificar el título de la página
    await expect(page).toHaveTitle(/GAJE Helix \| Semantic Genomic Compression Platform/);

    // 3. Verificar el encabezado principal
    const headerTitle = page.locator('header h1');
    await expect(headerTitle).toContainText('GAJE Helix');

    // 4. Verificar que las tarjetas de métricas y entorno existan
    const sidebar = page.locator('aside.sidebar');
    await expect(sidebar).toBeVisible();

    const modelSelect = page.locator('#model-select');
    await expect(modelSelect).toBeVisible();

    // 5. Verificar que el estado del sistema inicial esté en "Núcleo GAJE inicializado"
    const systemMessage = page.locator('.message.system').first();
    await expect(systemMessage).toContainText('Núcleo GAJE inicializado');

    // 6. Probar el cambio de tema (Theme Toggle)
    const htmlElement = page.locator('html');

    // Debería ser oscuro por defecto
    await expect(htmlElement).toHaveAttribute('data-theme', 'dark');

    // Hacer clic en el botón de tema
    const themeBtn = page.locator('#theme-toggle');
    await themeBtn.click();

    // Debería cambiar a claro
    await expect(htmlElement).toHaveAttribute('data-theme', 'light');

    // Volver a hacer clic
    await themeBtn.click();
    await expect(htmlElement).toHaveAttribute('data-theme', 'dark');

    // 7. Tomar una captura de pantalla del panel principal para control visual
    const screenshotDir = path.join(__dirname, 'screenshots');
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }
    const screenshotPath = path.join(screenshotDir, 'dashboard.png');
    await page.screenshot({ path: screenshotPath });
    console.log(`📸 Captura de pantalla de la interfaz guardada en: ${screenshotPath}`);
  });

  test('should verify model selection dropdown change triggers loading state', async ({ page }) => {
    await page.goto('/');

    // Asegurarse de que el dropdown esté visible
    const modelSelect = page.locator('#model-select');
    await expect(modelSelect).toBeVisible();

    // Esperar a que se carguen las opciones desde el backend
    await page.waitForTimeout(1000);

    const optionsCount = await modelSelect.locator('option').count();
    console.log(`Modelos detectados en el menú: ${optionsCount}`);

    if (optionsCount > 1) {
      // Obtener el valor de la segunda opción
      const secondOptionValue = await modelSelect.locator('option').nth(1).getAttribute('value');

      // Seleccionar la segunda opción
      await modelSelect.selectOption(secondOptionValue);

      // El selector debe deshabilitarse temporalmente mientras carga
      await expect(modelSelect).toBeDisabled();

      // Esperar a que termine de cargar y vuelva a habilitarse
      await expect(modelSelect).toBeEnabled({ timeout: 15000 });

      // Verificar que se haya impreso el mensaje de éxito en la consola de chat
      const lastSystemMessage = page.locator('.message.system').last();
      await expect(lastSystemMessage).toContainText('cargado y listo en memoria');
    }
  });
});
