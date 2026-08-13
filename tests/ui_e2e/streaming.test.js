const { test, expect } = require('@playwright/test');
const http = require('http');
const fs = require('fs');
const path = require('path');

const WEB_UI_DIR = path.resolve(__dirname, '../../examples/ui/web_ui');

function createMockServer() {
  const server = http.createServer((req, res) => {
    const url = req.url.split('?')[0];

    if (url === '/api/chat/stream' && req.method === 'POST') {
      let body = '';
      req.on('data', c => (body += c));
      req.on('end', () => {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        const tokens = ['Hola', ' ', 'esto', ' es', ' un', ' stream', ' real.'];
        let i = 0;
        const iv = setInterval(() => {
          if (i >= tokens.length) {
            res.write('data: [DONE]\n\n');
            res.end();
            clearInterval(iv);
            return;
          }
          res.write(`data: ${JSON.stringify(tokens[i])}\n\n`);
          i++;
        }, 30);
      });
      return;
    }

    if (url === '/api/load_model' && req.method === 'POST') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
      return;
    }

    if (url === '/api/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ models: [{ name: 'test_model.gaje.flat', date: '2026-08-13 00:00' }] }));
      return;
    }

    if (url === '/api/info') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        software: 'x', hardware: 'y', simd: 'AVX2',
        island: { pills: ['⚡'], memory_type: 'gmem', retrieval_latency_ms: 1, context_budget: 512 },
      }));
      return;
    }

    let filePath = url;
    if (filePath === '/') filePath = '/index.html';
    const file = path.join(WEB_UI_DIR, filePath);
    if (fs.existsSync(file) && fs.statSync(file).isFile()) {
      const contentType = filePath.endsWith('.css') ? 'text/css' : filePath.endsWith('.js') ? 'text/javascript' : 'text/html';
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(fs.readFileSync(file));
    } else {
      res.writeHead(404);
      res.end('not found');
    }
  });
  return new Promise(resolve => server.listen(0, () => resolve({ server, port: server.address().port })));
}

test('Streaming SSE + historial local', async ({ page }) => {
  const { server, port } = await createMockServer();
  const errors = [];
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));

  await page.goto(`http://localhost:${port}/`, { waitUntil: 'networkidle' });
  await page.evaluate(() => localStorage.removeItem('gaje_chat_history'));

  await expect(page.locator('#user-input')).toBeEnabled({ timeout: 10000 });
  await expect(page.locator('#send-btn')).toBeEnabled({ timeout: 10000 });

  await page.fill('#user-input', 'hola mundo');
  await page.click('#send-btn');

  await expect(page.locator('.message.user')).toHaveCount(1);
  await expect(page.locator('.message.bot.streaming')).toBeVisible({ timeout: 5000 });
  await expect(page.locator('.stream-text')).toHaveText('Hola esto es un stream real.', { timeout: 10000 });
  await expect(page.locator('#stop-btn')).toBeHidden();

  const hist = await page.evaluate(() => JSON.parse(localStorage.getItem('gaje_chat_history')));
  expect(hist.length).toBeGreaterThanOrEqual(2);
  expect(hist[0].role).toBe('user');
  expect(hist[hist.length - 1].role).toBe('assistant');
  expect(hist[hist.length - 1].content).toContain('stream real');

  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.locator('.message.user')).toHaveCount(1, { timeout: 5000 });
  await expect(page.locator('.message.bot')).toHaveCount(1);

  await page.click('#clear-history-btn');
  await expect(page.locator('.message.user')).toHaveCount(0);

  server.close();
  expect(errors).toEqual([]);
});