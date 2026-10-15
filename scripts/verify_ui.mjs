// Browser smoke check for the local FinSight UI. Run while the API is serving
// on http://127.0.0.1:8001/ (override with FINSIGHT_UI_URL).
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const url = process.env.FINSIGHT_UI_URL || 'http://127.0.0.1:8001/';
const chrome = process.env.CHROME_PATH || (process.platform === 'win32'
  ? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
  : '/usr/bin/google-chrome');
const profile = mkdtempSync(join(tmpdir(), 'finsight-ui-'));
const browser = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--window-size=1440,1000', url,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitFor(check, label, timeout = 15000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    try {
      const value = await check();
      if (value) return value;
    } catch { /* Browser is still starting. */ }
    await sleep(100);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

let socket;
let nextId = 0;
const pending = new Map();
function send(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const response = await send('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true,
  });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result?.value;
}
function assert(condition, description) {
  if (!condition) throw new Error(description);
  console.log('PASS', description);
}
async function screenshot(name) {
  const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  const path = join(profile, name);
  writeFileSync(path, Buffer.from(data, 'base64'));
  console.log('SCREENSHOT', path);
}

try {
  const expectedCompanies = (await (await fetch(new URL('/coverage', url))).json()).companies.length;
  assert(expectedCompanies > 0, 'filing corpus is available');
  const portFile = join(profile, 'DevToolsActivePort');
  const port = await waitFor(() => existsSync(portFile)
    && Number(readFileSync(portFile, 'utf8').split(/\r?\n/)[0]), 'Chrome debugging port');
  const tab = await waitFor(async () => {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return targets.find((target) => target.type === 'page' && target.url.startsWith(url));
  }, 'UI tab');
  socket = new WebSocket(tab.webSocketDebuggerUrl);
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  await send('Page.enable');
  await send('Runtime.enable');
  await waitFor(() => evaluate(`document.querySelector('#coverage-count')?.textContent === '${expectedCompanies}'`), 'coverage');
  assert(await evaluate(`document.querySelector('#coverage-count').textContent === '${expectedCompanies}'`),
    'coverage loads from the API');

  await evaluate("setTheme('light')");
  await screenshot('desktop-light.png');
  await evaluate("document.querySelector('#compare-tab').click()");
  assert(await evaluate("!document.querySelector('#compare-panel').hidden && document.querySelector('#ask-panel').hidden"),
    'compare tab opens');
  for (const [group, value] of [['companies', 'AAPL'], ['companies', 'MSFT'],
    ['metrics', 'Revenue'], ['years', '2025']]) {
    await evaluate(`document.querySelector('[data-group="${group}"][data-value="${value}"]').click()`);
  }
  assert(await evaluate("!document.querySelector('#compare-button').disabled"),
    'valid comparison enables the action');
  assert(await evaluate("document.querySelectorAll('#year-options .option-button').length >= 3"),
    'fiscal-year options come from coverage');
  await screenshot('compare-light.png');

  await evaluate(`renderResult({question:'How did revenue compare?',route:'NUMERIC',answer:'',
    data:[{ticker:'AAPL',period_fy:2025,label:'Revenue',value:100},
      {ticker:'MSFT',period_fy:2025,label:'Revenue',value:200}],
    latency_ms:100,system_ms:100,llm_ms:0,throttle_ms:0,citations_valid:null},
    {companies:['AAPL','MSFT','GOOGL'],metrics:['Revenue'],years:[2025]})`);
  assert(await evaluate("document.querySelectorAll('.data-table tbody tr').length === 3 && document.querySelector('.data-table').textContent.includes('Not returned') && document.querySelector('.chart svg')"),
    'partial comparisons disclose figures not returned');
  await evaluate("document.querySelector('#result-region').scrollIntoView({behavior:'instant',block:'start'})");
  await screenshot('figures-light.png');

  await evaluate(`renderResult({question:'What are the risks?',route:'NARRATIVE',
    answer:'* **Supply constraints:** Limited suppliers [1].',sources:[{n:1,ticker:'AAPL',section:'Risk factors',
    text:'Supplier disruption may affect operations.',fiscal_year:2025,
    filing_date:'2025-10-31',chunk_id:'demo'}],latency_ms:100,system_ms:50,
    llm_ms:50,throttle_ms:0,citations_valid:true})`);
  await evaluate("document.querySelector('.citation').click()");
  assert(await evaluate("document.querySelector('#source-1').open && document.querySelector('.source-excerpt').textContent.includes('Supplier disruption') && document.querySelector('.answer-text li strong')?.textContent === 'Supply constraints:'"),
    'Markdown answer and citation excerpt render correctly');
  await evaluate("document.querySelector('#result-region').scrollIntoView({behavior:'instant',block:'start'})");
  await screenshot('answer-light.png');

  await send('Emulation.setDeviceMetricsOverride', {
    width: 390, height: 844, deviceScaleFactor: 1, mobile: true,
  });
  await send('Page.reload', { ignoreCache: true });
  await waitFor(() => evaluate(`document.querySelector('#coverage-count')?.textContent === '${expectedCompanies}'`), 'mobile page');
  await evaluate("setTheme('light')");
  const width = await evaluate('({viewport:innerWidth,content:document.documentElement.scrollWidth})');
  assert(width.content <= width.viewport, `mobile has no horizontal overflow (${width.content}/${width.viewport})`);
  await screenshot('mobile-light.png');
} finally {
  socket?.close();
  browser.kill();
}
