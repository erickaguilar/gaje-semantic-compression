---
name: playwright-cli
description: Run automated E2E browser tests and capture screenshots of web applications using Playwright.
---

# Playwright Integration Skill

This skill allows the agent to run automated E2E browser testing on local or remote web applications, capture screenshots, and verify user interface components.

## Core Commands

### 1. Run E2E Tests
Run the entire Playwright test suite headlessly:
```bash
npx playwright test
```

### 2. Run Specific Test File
Run only a specific test file:
```bash
npx playwright test tests/ui_e2e/web_ui.test.js
```

### 3. Run Tests in Headed Mode
Run tests with visible browser UI to debug visual interactions:
```bash
npx playwright test --headed
```

### 4. Debug Tests
Open the Playwright Inspector UI to step through the test execution:
```bash
npx playwright test --debug
```

### 5. View Test Report
Generate and show the HTML report of the latest test run:
```bash
npx playwright show-report
```

## Best Practices
- **Isolation**: Always use clean contexts or restart the server when testing system/session transitions.
- **Assertions**: Prefer semantic assertions (e.g. `toHaveTitle`, `toBeVisible`, `toContainText`) rather than hardcoding exact styles.
- **Screenshots**: Save screenshots to `tests/ui_e2e/screenshots/` to verify layout aesthetics and responsiveness.
