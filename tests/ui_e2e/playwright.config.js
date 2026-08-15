const { defineConfig, devices } = require('@playwright/test');
const path = require('path');

const repoRoot = path.join(__dirname, '..', '..');

module.exports = defineConfig({
  testDir: './',
  testIgnore: 'streaming.test.js',
  timeout: 30 * 1000,
  expect: {
    timeout: 5000
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8080',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: '.venv/bin/python3 examples/ui/web_ui/server.py',
    url: 'http://127.0.0.1:8080',
    cwd: repoRoot,
    reuseExistingServer: !process.env.CI,
    timeout: 10 * 1000,
  },
});
