const $ = (selector) => document.querySelector(selector);
const resultRegion = $('#result-region');
const questionInput = $('#question');
const coverageDialog = $('#coverage-dialog');
const selection = { companies: new Set(), metrics: new Set(), years: new Set() };
const METRICS = [
  { key: 'Revenue', label: 'Revenue', phrase: 'total revenue' },
  { key: 'Gross profit', label: 'Gross profit', phrase: 'gross profit' },
  { key: 'Net income', label: 'Net income', phrase: 'net income' },
  { key: 'Operating income', label: 'Operating income', phrase: 'operating income' },
  { key: 'R&D expense', label: 'R&D expense', phrase: 'research and development expense' },
  { key: 'Diluted EPS', label: 'Diluted EPS', phrase: 'diluted earnings per share' },
];
const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)', 'var(--chart-3)', 'var(--chart-4)'];
let coverage = [];
let requestId = 0;
let activeController = null;
let lastQuestion = '';
let lastOptions = {};
let lastResponse = null;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}
function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function scrollToElement(element, block = 'start') {
  element.scrollIntoView({
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    block,
  });
}
function formatYear(year) { return year == null ? 'Year unavailable' : 'FY' + year; }
function joinAnd(items) {
  if (items.length < 2) return items[0] || '';
  return items.slice(0, -1).join(', ') + ' and ' + items[items.length - 1];
}
function metricOf(label, concept = '') {
  const text = String(label || '').toLowerCase();
  if (/RevenueFromContract|_Revenues$/.test(concept)) return 'Revenue';
  if (/_GrossProfit$/.test(concept)) return 'Gross profit';
  if (/_NetIncomeLoss$/.test(concept)) return 'Net income';
  if (/_OperatingIncomeLoss$/.test(concept)) return 'Operating income';
  if (/_ResearchAndDevelopmentExpense$/.test(concept)) return 'R&D expense';
  if (/_EarningsPerShareDiluted$/.test(concept)) return 'Diluted EPS';
  if (/net sales|revenue|revenues/.test(text)) return 'Revenue';
  if (/gross (profit|margin)/.test(text)) return 'Gross profit';
  if (/net income|net loss/.test(text)) return 'Net income';
  if (/operating income|operating loss|income \(loss\) from operations|income from operations/.test(text)) return 'Operating income';
  if (/research and development/.test(text)) return 'R&D expense';
  if (/diluted|per share|earnings per share/.test(text)) return 'Diluted EPS';
  return label || 'Reported value';
}
function money(value, label, compact = false) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const perShare = /per share|eps|diluted/i.test(label || '');
  const sign = value < 0 ? '−' : '';
  const absolute = Math.abs(value);
  if (perShare || absolute < 1000) return sign + '$' + absolute.toFixed(2);
  if (!compact) return sign + '$' + Math.round(absolute).toLocaleString('en-US');
  if (absolute >= 1e12) return sign + '$' + (absolute / 1e12).toFixed(2) + 'T';
  if (absolute >= 1e9) return sign + '$' + (absolute / 1e9).toFixed(2) + 'B';
  if (absolute >= 1e6) return sign + '$' + (absolute / 1e6).toFixed(1) + 'M';
  return sign + '$' + (absolute / 1e3).toFixed(1) + 'K';
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const dark = theme === 'dark';
  $('#theme').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  $('#theme').innerHTML = dark
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M20 14.3A8.3 8.3 0 0 1 9.7 4 8.5 8.5 0 1 0 20 14.3Z"/></svg>';
  try { localStorage.setItem('finsight-theme', theme); } catch {}
}
function initialTheme() {
  let saved = null;
  try { saved = localStorage.getItem('finsight-theme'); } catch {}
  setTheme(saved === 'dark' || saved === 'light'
    ? saved : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
}
initialTheme();
$('#theme').addEventListener('click', () => setTheme(
  document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
));

function setMode(mode, focus = false) {
  const ask = mode === 'ask';
  $('#ask-panel').hidden = !ask;
  $('#compare-panel').hidden = ask;
  for (const [button, active] of [[$('#ask-tab'), ask], [$('#compare-tab'), !ask]]) {
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
  }
  if (focus) (ask ? $('#ask-tab') : $('#compare-tab')).focus();
}
$('#ask-tab').addEventListener('click', () => setMode('ask'));
$('#compare-tab').addEventListener('click', () => setMode('compare'));
$('.workspace-tabs').addEventListener('keydown', (event) => {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault();
  setMode($('#ask-panel').hidden ? 'ask' : 'compare', true);
});

function historyItems() {
  try {
    const items = JSON.parse(sessionStorage.getItem('finsight-history') || '[]');
    return Array.isArray(items) ? items.filter((item) => typeof item === 'string') : [];
  } catch { return []; }
}
function renderHistory() {
  const items = historyItems();
  $('#recent-section').hidden = items.length === 0;
  $('#recent-list').innerHTML = items.map((item) =>
    '<button class="recent-item" type="button" data-question="' + esc(item) + '" title="' + esc(item) + '">' + esc(item) + '</button>'
  ).join('');
}
function remember(question) {
  const items = [question, ...historyItems().filter((item) => item !== question)].slice(0, 6);
  try { sessionStorage.setItem('finsight-history', JSON.stringify(items)); } catch {}
  renderHistory();
}
renderHistory();
$('#recent-list').addEventListener('click', (event) => {
  const button = event.target.closest('[data-question]');
  if (button) { setMode('ask'); run(button.dataset.question); }
});
$('.example-grid').addEventListener('click', (event) => {
  const button = event.target.closest('[data-question]');
  if (button) run(button.dataset.question);
});

function allYears() {
  return [...new Set(coverage.flatMap((company) => company.years || []))].sort((a, b) => b - a);
}
function renderCoverage() {
  $('#coverage-count').textContent = coverage.length || '—';
  $('#coverage-list').innerHTML = coverage.length
    ? coverage.map((company) =>
      '<div class="coverage-row"><strong>' + esc(company.ticker) + '</strong><span>Latest filing ' +
      esc(formatYear(company.filing_year)) + ' · Figures ' +
      (company.years?.length ? company.years.map(formatYear).join(', ') : 'unavailable') +
      '</span></div>'
    ).join('')
    : '<div class="coverage-empty">Company coverage is currently unavailable.</div>';
}
async function loadCoverage() {
  try {
    const response = await fetch('/coverage');
    if (!response.ok) throw new Error('coverage unavailable');
    const payload = await response.json();
    coverage = Array.isArray(payload.companies) ? payload.companies : [];
  } catch {
    try {
      const response = await fetch('/companies');
      const payload = await response.json();
      coverage = (payload.companies || []).map((ticker) => ({ ticker, years: [], filing_year: null }));
    } catch { coverage = []; }
  }
  renderCoverage();
  renderComparisonOptions();
}

function optionButton(group, value, label, extraClass = '', disabled = false, title = '') {
  const pressed = selection[group].has(value);
  return '<button class="option-button ' + extraClass + '" type="button" data-group="' + group +
    '" data-value="' + esc(value) + '" aria-pressed="' + pressed + '"' +
    (disabled ? ' disabled' : '') + (title ? ' title="' + esc(title) + '"' : '') +
    '>' + esc(label) + '</button>';
}
function renderComparisonOptions() {
  $('#company-options').innerHTML = coverage.map((company) =>
    optionButton('companies', company.ticker, company.ticker)
  ).join('') || '<p class="field-help">Companies are unavailable right now.</p>';
  $('#metric-options').innerHTML = METRICS.map((metric) =>
    optionButton('metrics', metric.key, metric.label)
  ).join('');
  const selected = [...selection.companies];
  $('#year-options').innerHTML = allYears().map((year) => {
    const available = selected.filter((ticker) =>
      coverage.find((company) => company.ticker === ticker)?.years?.includes(year)
    ).length;
    const partial = selected.length > 0 && available > 0 && available < selected.length;
    const disabled = selected.length > 0 && available === 0;
    const title = partial ? 'Available for ' + available + ' of ' + selected.length + ' selected companies' : '';
    return optionButton('years', String(year), formatYear(year), partial ? 'partial' : '', disabled, title);
  }).join('') || '<p class="field-help">Fiscal-year coverage is unavailable right now.</p>';
  $('#year-help').textContent = selected.length
    ? 'Partial means some selected companies have no structured facts for that year. A metric can still be unavailable in an otherwise covered year.'
    : 'Choose companies to see which fiscal years are available.';
  updateComparisonPreview();
}
function updateComparisonPreview(message = '') {
  const companies = [...selection.companies];
  const metrics = [...selection.metrics];
  const years = [...selection.years].map(Number).sort((a, b) => a - b);
  const ready = companies.length > 0 && metrics.length > 0 && years.length > 0 &&
    (companies.length > 1 || metrics.length > 1 || years.length > 1);
  $('#compare-button').disabled = !ready;
  $('#compare-preview').textContent = message || (ready
    ? companies.join(' · ') + '  /  ' + metrics.join(' · ') + '  /  ' + years.map(formatYear).join(' · ')
    : 'Choose at least one company, metric, and fiscal year; select two or more values to compare.');
}
$('#compare-panel').addEventListener('click', (event) => {
  const button = event.target.closest('.option-button');
  if (!button) return;
  const group = button.dataset.group;
  const value = button.dataset.value;
  const set = selection[group];
  if (set.has(value)) set.delete(value);
  else {
    const limit = { companies: 4, metrics: 2, years: 3 }[group];
    if (set.size >= limit) {
      updateComparisonPreview('Select up to ' + limit + ' ' + group + '.');
      return;
    }
    set.add(value);
  }
  if (group === 'companies') {
    for (const year of [...selection.years]) {
      const supported = [...selection.companies].some((ticker) =>
        coverage.find((company) => company.ticker === ticker)?.years?.includes(Number(year))
      );
      if (selection.companies.size && !supported) selection.years.delete(year);
    }
  }
  renderComparisonOptions();
});
$('#compare-button').addEventListener('click', () => {
  const spec = {
    companies: [...selection.companies],
    metrics: [...selection.metrics],
    years: [...selection.years].map(Number).sort((a, b) => a - b),
  };
  const phrases = spec.metrics.map((key) => METRICS.find((metric) => metric.key === key).phrase);
  const yearText = spec.years.length === 1
    ? 'fiscal year ' + spec.years[0]
    : 'fiscal years ' + joinAnd(spec.years.map(String));
  const query = 'Compare ' + joinAnd(spec.companies) + ' ' + joinAnd(phrases) + ' for ' + yearText;
  run(query, { comparison: spec });
});

$('#coverage-open').addEventListener('click', () => coverageDialog.showModal());
$('#coverage-close').addEventListener('click', () => coverageDialog.close());
coverageDialog.addEventListener('click', (event) => {
  if (event.target === coverageDialog) coverageDialog.close();
});

function showAddStatus(message, error = false) {
  const status = $('#add-company-status');
  status.textContent = message;
  status.style.color = error ? 'var(--error)' : '';
}
async function pollCompany(ticker) {
  for (let i = 0; i < 120; i++) {
    await sleep(2500);
    try {
      const response = await fetch('/ingest/' + encodeURIComponent(ticker));
      if (!response.ok) continue;
      const payload = await response.json();
      if (payload.status === 'ready') return true;
      if (String(payload.status).startsWith('error')) throw new Error(payload.status);
    } catch (error) {
      if (String(error.message).startsWith('error')) throw error;
    }
    if (i % 4 === 0) showAddStatus('Indexing ' + ticker + '… ' + Math.round((i + 1) * 2.5) + ' seconds elapsed.');
  }
  throw new Error('Still processing. Check coverage again in a moment.');
}
$('#add-company-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const ticker = $('#new-ticker').value.trim().toUpperCase();
  if (!/^[A-Z.]{1,6}$/.test(ticker)) {
    showAddStatus('Enter a valid stock ticker of up to six letters.', true);
    return;
  }
  const button = $('#add-company-button');
  button.disabled = true;
  showAddStatus('Fetching ' + ticker + "'s latest filing…");
  try {
    const response = await fetch('/ingest', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ticker }),
    });
    const payload = await response.json();
    if (!response.ok && payload.status !== 'ready') throw new Error(payload.detail || 'Could not start ingestion.');
    if (payload.status !== 'ready') await pollCompany(ticker);
    await loadCoverage();
    $('#new-ticker').value = '';
    showAddStatus(ticker + ' is ready to research.');
  } catch (error) {
    showAddStatus(String(error.message || error), true);
  } finally { button.disabled = false; }
});

function trendSVG(series, highlightedYear, label) {
  const points = (series || []).filter((point) =>
    typeof point.value === 'number' && Number.isFinite(point.value)
  ).sort((a, b) => a.period_fy - b.period_fy);
  if (points.length < 2) return '';
  const values = points.map((point) => point.value);
  const low = Math.min(...values);
  const span = Math.max(...values) - low || 1;
  const x = (index) => 9 + index * 152 / (points.length - 1);
  const y = (value) => 8 + 38 * (1 - (value - low) / span);
  const line = points.map((point, index) =>
    (index ? 'L' : 'M') + x(index).toFixed(1) + ',' + y(point.value).toFixed(1)
  ).join(' ');
  const selected = Math.max(0, points.findIndex((point) => point.period_fy === highlightedYear));
  const first = points[0], last = points[points.length - 1];
  const change = first.value
    ? ((last.value - first.value) / Math.abs(first.value) * 100)
    : 0;
  return '<div class="trend"><svg viewBox="0 0 170 54" role="img" aria-label="' +
    esc(metricOf(label)) + ' from ' + esc(formatYear(first.period_fy)) + ' to ' +
    esc(formatYear(last.period_fy)) + '">' +
    '<path d="' + line + '" fill="none" stroke="var(--chart-1)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>' +
    '<circle cx="' + x(selected).toFixed(1) + '" cy="' + y(points[selected].value).toFixed(1) +
    '" r="4" fill="var(--chart-1)"/></svg><div class="trend-caption">From ' +
    esc(formatYear(first.period_fy)) + ' to ' + esc(formatYear(last.period_fy)) +
    ' <strong>' + (change >= 0 ? '+' : '−') + Math.abs(change).toFixed(1) + '%</strong></div></div>';
}
function niceStep(range) {
  if (range <= 0) return 1;
  const power = Math.pow(10, Math.floor(Math.log10(range)));
  const ratio = range / power;
  const size = ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 2.5 ? 2.5 : ratio <= 5 ? 5 : 10;
  return size * power;
}
function chartSVG(rows, metric) {
  const tickers = [...new Set(rows.map((row) => row.ticker))];
  const years = [...new Set(rows.map((row) => row.period_fy).filter((year) => year != null))].sort((a, b) => a - b);
  const byYear = years.length > 1;
  const categories = byYear ? years : tickers;
  const series = byYear ? tickers : ['value'];
  const values = rows.map((row) => row.value).filter((value) => typeof value === 'number');
  if (values.length < 2 || !categories.length) return '';
  const perShare = metric === 'Diluted EPS';
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(0, ...values);
  const step = niceStep((maximum - minimum) / 4);
  const bottom = Math.floor(minimum / step) * step;
  const top = Math.max(bottom + step, Math.ceil(maximum / step) * step);
  const divisor = perShare ? 1 : top >= 1e12 ? 1e12 : top >= 1e9 ? 1e9 : top >= 1e6 ? 1e6 : 1;
  const units = perShare ? 'USD per share' : divisor === 1e12 ? 'USD trillions' :
    divisor === 1e9 ? 'USD billions' : divisor === 1e6 ? 'USD millions' : 'USD';
  const W = 760, H = 270, left = 70, right = 18, upper = 20, lower = 45;
  const plotWidth = W - left - right, plotHeight = H - upper - lower;
  const y = (value) => upper + plotHeight * (top - value) / (top - bottom);
  const baseline = y(0);
  const groupWidth = plotWidth / categories.length;
  const gap = 4;
  const barWidth = Math.max(5, Math.min(36, (groupWidth * .72 - (series.length - 1) * gap) / series.length));
  const innerWidth = series.length * barWidth + (series.length - 1) * gap;
  let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
    esc(metric) + ' comparison chart. Exact values are listed in the table below.">';
  for (let i = 0; i <= 4; i++) {
    const value = bottom + (top - bottom) * i / 4;
    const yy = y(value);
    const tick = value / divisor;
    svg += '<line x1="' + left + '" y1="' + yy.toFixed(1) + '" x2="' + (W - right) +
      '" y2="' + yy.toFixed(1) + '" stroke="var(--grid)"/>' +
      '<text x="' + (left - 9) + '" y="' + (yy + 4).toFixed(1) +
      '" text-anchor="end" fill="var(--muted)" font-size="11">' +
      esc(Math.abs(tick) >= 10 ? tick.toFixed(0) : tick.toFixed(1)) + '</text>';
  }
  svg += '<line x1="' + left + '" y1="' + baseline.toFixed(1) + '" x2="' +
    (W - right) + '" y2="' + baseline.toFixed(1) + '" stroke="var(--line-strong)"/>';
  categories.forEach((category, groupIndex) => {
    const groupX = left + groupIndex * groupWidth + (groupWidth - innerWidth) / 2;
    series.forEach((ticker, seriesIndex) => {
      const row = rows.find((item) => byYear
        ? item.period_fy === category && item.ticker === ticker
        : item.ticker === category);
      if (!row || typeof row.value !== 'number') return;
      const barY = y(row.value);
      const color = CHART_COLORS[byYear ? seriesIndex % 4 : groupIndex % 4];
      svg += '<rect x="' + (groupX + seriesIndex * (barWidth + gap)).toFixed(1) +
        '" y="' + Math.min(barY, baseline).toFixed(1) +
        '" width="' + barWidth.toFixed(1) +
        '" height="' + Math.max(1, Math.abs(baseline - barY)).toFixed(1) +
        '" rx="2" fill="' + color + '"><title>' + esc(row.ticker) + ' ' +
        esc(formatYear(row.period_fy)) + ' ' + esc(metric) + ': ' +
        esc(money(row.value, row.label)) + '</title></rect>';
    });
    svg += '<text x="' + (left + (groupIndex + .5) * groupWidth).toFixed(1) +
      '" y="' + (H - 15) + '" text-anchor="middle" fill="var(--ink-soft)" font-size="12">' +
      esc(byYear ? formatYear(category) : category) + '</text>';
  });
  svg += '</svg>';
  const legend = byYear && tickers.length > 1
    ? '<div class="chart-legend">' + tickers.map((ticker, index) =>
      '<span><i style="background:' + CHART_COLORS[index % 4] + '"></i>' + esc(ticker) + '</span>'
    ).join('') + '</div>' : '';
  return '<div class="chart"><h4>' + esc(metric) + '</h4><p class="chart-subtitle">' +
    esc(units) + '</p>' + svg + legend + '</div>';
}
function tableRows(rows, comparison) {
  if (!comparison) return rows.map((row) => ({ ...row, metric: metricOf(row.label, row.concept) }));
  const matches = [];
  for (const metric of comparison.metrics) {
    for (const ticker of comparison.companies) {
      for (const year of comparison.years) {
        const found = rows.find((row) => row.ticker === ticker &&
          Number(row.period_fy) === year && metricOf(row.label, row.concept) === metric);
        matches.push(found ? { ...found, metric } :
          { ticker, period_fy: year, metric, value: null, label: metric });
      }
    }
  }
  return matches;
}
function dataTable(rows, comparison) {
  const items = tableRows(rows, comparison);
  if (!items.length) return '';
  return '<div class="table-wrap"><table class="data-table">' +
    '<caption>Reported values and figures not returned by this search</caption>' +
    '<thead><tr><th scope="col">Company</th><th scope="col">Fiscal year</th>' +
    '<th scope="col">Metric</th><th scope="col" class="number">Reported value</th></tr></thead><tbody>' +
    items.map((row) =>
      '<tr><td><strong>' + esc(row.ticker || '') + '</strong></td><td>' +
      esc(formatYear(row.period_fy)) + '</td><td>' + esc(row.metric) +
      '</td><td class="number ' + (row.value == null ? 'empty-cell' : '') + '">' +
      (row.value == null ? 'Not returned' : esc(money(row.value, row.label))) + '</td></tr>'
    ).join('') + '</tbody></table></div>';
}
function numericHTML(data, comparison) {
  const rows = (data.data || []).filter((row) =>
    typeof row.value === 'number' && Number.isFinite(row.value)
  );
  if (!rows.length) {
    let empty = '<div class="answer-text">' + esc(data.answer || 'No reported figure found.') + '</div>';
    if (comparison) {
      empty += '<p class="missing-note">No requested figures were returned. The figures may still exist in the filings; try fewer companies, one metric, or different years.</p>' +
        '<section class="result-section"><h3>Requested data</h3>' + dataTable(rows, comparison) + '</section>';
    } else {
      empty += '<p class="result-note">Try naming a specific company, metric, and fiscal year.</p>';
    }
    return empty;
  }
  const single = rows.length === 1 && !comparison;
  let html = '';
  if (single) {
    const row = rows[0];
    html += '<div class="figure-hero"><div><div class="figure-label">' +
      esc(metricOf(row.label, row.concept)) + '</div><div class="figure-value">' +
      esc(money(row.value, row.label, true)) + '</div><div class="figure-context"><strong>' +
      esc(row.ticker) + '</strong> · ' + esc(formatYear(row.period_fy)) +
      ' · Reported ' + esc(money(row.value, row.label)) + '</div></div>' +
      trendSVG(data.trend, row.period_fy, row.label) + '</div>';
  } else {
    const groups = new Map();
    for (const row of rows) {
      const key = metricOf(row.label, row.concept);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    const charts = [...groups].map(([metric, group]) => chartSVG(group, metric)).filter(Boolean);
    if (charts.length) html += '<div class="chart-grid">' + charts.join('') + '</div>';
  }
  const displayRows = tableRows(rows, comparison);
  if (displayRows.some((row) => row.value == null)) {
    html += '<p class="missing-note">Some requested figures were not returned. This does not necessarily mean they are absent from the filings; try a narrower comparison.</p>';
  }
  html += '<section class="result-section"><h3>Reported data</h3>' +
    dataTable(rows, comparison) + '<p class="result-note">Figures come from structured financial statement facts. Expand the details below to inspect the generated SQL.</p></section>';
  return html;
}

function citedInline(text, sources) {
  const valid = new Set((sources || []).map((source) => source.n));
  return esc(text || '')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d+)\]/g, (match, number) => {
    const n = Number(number);
    return valid.has(n)
      ? '<a class="citation" href="#source-' + n + '" aria-label="View source ' + n + '">[' + n + ']</a>'
      : match;
  });
}
function citedAnswer(text, sources) {
  const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
  let html = '';
  let list = null;
  let paragraph = [];
  const flushParagraph = () => {
    if (!paragraph.length) return;
    html += '<p>' + paragraph.map((line) => citedInline(line, sources)).join('<br>') + '</p>';
    paragraph = [];
  };
  const closeList = () => {
    if (list) html += '</' + list + '>';
    list = null;
  };
  for (const original of lines) {
    const line = original.trim();
    if (!line) { flushParagraph(); closeList(); continue; }
    const bullet = line.match(/^[-*•]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? 'ul' : 'ol';
      if (list !== type) { closeList(); html += '<' + type + '>'; list = type; }
      html += '<li>' + citedInline((bullet || numbered)[1], sources) + '</li>';
      continue;
    }
    closeList();
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      html += '<h3>' + citedInline(heading[1], sources) + '</h3>';
    } else paragraph.push(line);
  }
  flushParagraph();
  closeList();
  return html;
}
function sourceHTML(sources) {
  if (!sources?.length) return '';
  const groups = new Map();
  for (const source of sources) {
    if (!groups.has(source.ticker)) groups.set(source.ticker, []);
    groups.get(source.ticker).push(source);
  }
  return '<section class="result-section" aria-label="Filing evidence"><h3>Evidence from filings</h3>' +
    '<p class="source-help">Select a source number in the answer to open its filing excerpt.</p>' +
    [...groups].map(([ticker, items]) =>
      '<div class="source-group"><h4 class="source-group-title">' + esc(ticker) + '</h4>' +
      items.map((source) =>
        '<details class="source-card" id="source-' + source.n + '"><summary><span class="source-number">[' +
        source.n + ']</span><span class="source-label">' + esc(source.section) + '</span>' +
        '<span class="source-year">' + esc(formatYear(source.fiscal_year)) + '</span>' +
        '<span class="source-chevron" aria-hidden="true">⌄</span></summary>' +
        '<div class="source-content"><p class="source-excerpt">' + esc(source.text || 'Excerpt unavailable.') +
        '</p><div class="source-foot">' + esc(source.ticker) + ' · ' +
        esc(source.filing_date ? 'Filed ' + source.filing_date : formatYear(source.fiscal_year)) +
        ' · Chunk ' + esc(source.chunk_id) + '</div></div></details>'
      ).join('') + '</div>'
    ).join('') + '</section>';
}
function narrativeHTML(data) {
  let html = '<div class="answer-text">' + citedAnswer(data.answer, data.sources) + '</div>';
  if (data.citations_valid === false && data.sources?.length) {
    html += '<p class="citation-issue">Some citation markers are missing or do not match the listed sources. Check the excerpts before relying on this answer.</p>';
  }
  html += '<div class="answer-actions"><button class="secondary-button" id="copy-answer" type="button">Copy answer</button></div>';
  html += sourceHTML(data.sources);
  return html;
}
function technicalHTML(data) {
  const total = Math.max(Number(data.latency_ms) || 0, 1);
  const system = Number(data.system_ms) || 0;
  const llm = Number(data.llm_ms) || 0;
  const throttle = Number(data.throttle_ms) || 0;
  const desc = {
    NUMERIC: 'Structured financial facts queried with generated SQL',
    NARRATIVE: 'Filing passages retrieved and reranked before answer generation',
    MULTI_HOP: 'Separate filing searches, a sufficiency check, and cited synthesis',
  }[data.route] || 'Filing research';
  let html = '<details class="technical"><summary>How this answer was produced</summary><div class="technical-content">' +
    '<p>' + esc(desc) + '.</p>';
  if (data.cached) html += '<p>Returned from the answer cache in ' + Math.round(total) + ' ms.</p>';
  else {
    html += '<p>' + Math.round(total) + ' ms total · ' + Math.round(system) +
      ' ms in the service · ' + Math.round(llm) + ' ms at the language model' +
      (throttle > 0 ? ' · ' + Math.round(throttle) + ' ms waiting for rate limits' : '') + '</p>' +
      '<div class="latency-bar" aria-hidden="true"><span style="width:' + (system / total * 100).toFixed(1) +
      '%"></span><span style="width:' + (llm / total * 100).toFixed(1) +
      '%"></span><span style="width:' + (throttle / total * 100).toFixed(1) + '%"></span></div>';
  }
  if (data.route !== 'NUMERIC' && data.citations_valid !== null) {
    html += '<p>Citation check: ' + (data.citations_valid
      ? 'all used citation numbers match listed sources'
      : 'citation numbers are missing or invalid') +
      '. This check does not assess whether each claim is supported by the cited text.</p>';
  }
  if (data.steps?.length) {
    html += '<p>Search steps</p><ol>' + data.steps.map((step) =>
      '<li>' + esc(step.q) + ' · ' + Number(step.n_sources || 0) + ' new sources' +
      (step.recovery ? ' · follow-up search' : '') + '</li>'
    ).join('') + '</ol>';
  }
  if (data.sql) html += '<p style="margin-top:16px">Generated SQL</p><pre><code>' + esc(data.sql) + '</code></pre>';
  return html + '</div></details>';
}
function renderResult(data, comparison) {
  lastResponse = data;
  const routeLabel = {
    NUMERIC: 'Reported figures',
    NARRATIVE: 'Filing insight',
    MULTI_HOP: 'Across filings',
  }[data.route] || 'Research answer';
  const body = data.route === 'NUMERIC' ? numericHTML(data, comparison) : narrativeHTML(data);
  resultRegion.innerHTML = '<article class="result-card"><header class="result-header">' +
    '<div class="result-kicker">Your result</div><h2 class="result-title">' +
    esc(data.question) + '</h2><div class="result-meta"><span class="route-label">' +
    esc(routeLabel) + '</span><span aria-hidden="true">·</span><span>' +
    (data.cached ? 'From cache' : Math.max(0, (data.latency_ms / 1000)).toFixed(1) + ' seconds') +
    '</span></div></header><div class="result-body">' + body +
    technicalHTML(data) + '</div></article>';
  scrollToElement(resultRegion);
}
function showStatus(kind, title, detail, retry = false) {
  resultRegion.innerHTML = '<div class="status-card ' + kind + '" role="' +
    (kind === 'error' ? 'alert' : 'status') + '"><div class="status-row">' +
    (kind === 'loading' ? '<span class="spinner" aria-hidden="true"></span>' : '') +
    '<div><h2>' + esc(title) + '</h2><p>' + esc(detail) + '</p>' +
    (retry ? '<button class="secondary-button" id="retry-button" type="button">Try again</button>' : '') +
    '</div></div></div>';
}
resultRegion.addEventListener('click', async (event) => {
  const citation = event.target.closest('.citation');
  if (citation) {
    event.preventDefault();
    const target = resultRegion.querySelector(citation.getAttribute('href'));
    if (target) {
      target.open = true;
      scrollToElement(target, 'center');
      target.querySelector('summary')?.focus();
    }
    return;
  }
  if (event.target.closest('#retry-button')) {
    run(lastQuestion, { ...lastOptions, autoIngested: false });
    return;
  }
  const copy = event.target.closest('#copy-answer');
  if (copy && lastResponse) {
    try {
      await navigator.clipboard.writeText(lastResponse.answer || '');
      copy.textContent = 'Copied';
      copy.classList.add('copied');
    } catch { copy.textContent = 'Copy unavailable'; }
  }
});

async function pollAutoIngest(ticker, question, id, options) {
  const started = Date.now();
  showStatus('loading', 'Preparing ' + ticker + "'s filing",
    'Fetching and indexing the latest annual report. This may take a few minutes.');
  let checks = 0;
  while (Date.now() - started < 300000 && id === requestId) {
    await sleep(2500);
    if (id !== requestId) return;
    checks++;
    if (checks % 4 === 0) {
      const detail = resultRegion.querySelector('.status-card.loading p');
      if (detail) detail.textContent = 'Fetching and indexing the latest annual report · ' +
        Math.round((Date.now() - started) / 1000) + ' seconds elapsed.';
    }
    try {
      const response = await fetch('/ingest/' + encodeURIComponent(ticker));
      if (!response.ok) continue;
      const payload = await response.json();
      if (payload.status === 'ready') {
        await loadCoverage();
        if (id === requestId) run(question, { ...options, autoIngested: true });
        return;
      }
      if (String(payload.status).startsWith('error')) {
        showStatus('error', 'Could not add ' + ticker,
          'The filing could not be indexed. Try another company or retry in a moment.', true);
        return;
      }
    } catch {}
  }
  if (id === requestId) showStatus('error', 'Filing still processing',
    'This filing is taking longer than expected. You can ask again shortly.', true);
}
async function run(question, options = {}) {
  const text = String(question || '').trim();
  if (text.length < 3 || text.length > 500) {
    showStatus('error', 'Check your question', 'Enter a question between 3 and 500 characters.');
    return;
  }
  lastQuestion = text;
  lastOptions = options;
  questionInput.value = text;
  activeController?.abort();
  const controller = new AbortController();
  activeController = controller;
  const id = ++requestId;
  const started = Date.now();
  showStatus('loading', 'Researching your question', 'Searching the available filing data…');
  scrollToElement(resultRegion);
  const timer = setInterval(() => {
    const detail = id === requestId && resultRegion.querySelector('.status-card.loading p');
    if (detail) detail.textContent = 'Working on the answer · ' +
      Math.round((Date.now() - started) / 1000) + ' seconds elapsed.';
  }, 5000);
  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: text }),
      signal: controller.signal,
    });
    if (id !== requestId) return;
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 429) showStatus('error', 'The answer service is busy',
        'The language model has reached its current request limit. Please try again shortly.', true);
      else if (response.status === 503) showStatus('error', 'Still getting ready',
        'The filing models are loading. Try again in a few seconds.', true);
      else showStatus('error', 'We could not complete that question',
        payload.detail || 'Please try again or rephrase your question.', true);
      return;
    }
    if (payload.route === 'INGESTING') {
      clearInterval(timer);
      if (options.autoIngested || !payload.pending_ticker) {
        showStatus('error', 'Filing not ready yet', 'The company is still being indexed. Try again shortly.', true);
        return;
      }
      await pollAutoIngest(payload.pending_ticker, text, id, options);
      return;
    }
    renderResult(payload, options.comparison);
    remember(text);
  } catch (error) {
    if (id !== requestId || error.name === 'AbortError') return;
    showStatus('error', 'Connection interrupted',
      'The answer could not be loaded. Check the connection and try again.', true);
  } finally {
    clearInterval(timer);
    if (activeController === controller) activeController = null;
  }
}
$('#ask-form').addEventListener('submit', (event) => {
  event.preventDefault();
  run(questionInput.value);
});
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#ask-form').requestSubmit();
  }
});
document.addEventListener('keydown', (event) => {
  if (event.key === '/' && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
    event.preventDefault();
    setMode('ask');
    questionInput.focus();
  }
});
loadCoverage();
