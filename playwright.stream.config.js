const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/ui_e2e',
  testMatch: 'streaming.test.js',
  timeout: 60 * 1000,
  expect: {
    timeout: 5000
  },
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    browserName: 'chromium',
    channel: 'chrome',
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
