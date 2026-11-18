(() => {
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };
const TYPES = ['cafes', 'gyms', 'hospitals', 'salons', 'pharmacies', 'PG hostels', 'restaurants', 'dentists'];
const KINDS = { mobile: 'Mobile (likely)', landline: 'Landline', 'toll-free': 'Toll-free', none: 'No phone' };
let areas = [], current = null, history = [], historyTotal = 0, area = { name: 'Koramangala', city: 'Bangalore' };
const where = a => a.name + (a.city ? ' ' + a.city : '');

function params() {
  return { query: $('#query').value.trim(), lat: $('#lat').value.trim(), lng: $('#lng').value.trim(), radius: $('#radius').value.trim(), lang: $('#lang').value };
}
function preview() {
  const p = params();
  const parts = ['gmaps', '-fast-mode', '-json', '-c 1', `-geo "${p.lat || '?'},${p.lng || '?'}"`];
  if (p.radius) parts.push(`-radius ${p.radius}`);
  if (p.lang) parts.push(`-lang ${p.lang}`);
  parts.push('-input q.txt');
  $('#cmd').textContent = parts.join(' ') + '\n# q.txt contains one line: ' + (p.query || '(empty)');
  $('#form-error').textContent = '';
}
function setNotice(text, kind) { const n = $('#notice'); n.textContent = text; n.className = 'notice' + (kind ? ' ' + kind : ''); }
function safeLink(url) { try { const u = new URL(url); return ['https:', 'http:'].includes(u.protocol) ? u.href : null; } catch { return null; } }
function filled(v) { if (v === null || v === undefined || v === '' || v === 0) return false; if (Array.isArray(v)) return v.length > 0; if (typeof v === 'object') return Object.values(v).some(filled); return true; }

function presets() {
  const box = $('#presets');
  TYPES.forEach(t => {
    const b = el('button', t); b.type = 'button';
    b.addEventListener('click', () => { $('#query').value = `${t} in ${where(area)}`; document.querySelectorAll('#presets button').forEach(x => x.classList.toggle('active', x === b)); preview(); });
    box.append(b);
  });
}
function areaSelect(list) {
  const sel = $('#area');
  list.forEach((a, i) => { const o = el('option', a.name); o.value = String(i); sel.append(o); });
  const c = el('option', 'Custom'); c.value = 'custom'; sel.append(c);
  sel.addEventListener('change', () => {
    if (sel.value === 'custom') return;
    const a = list[Number(sel.value)]; area = a; $('#lat').value = a.lat; $('#lng').value = a.lng;
    const q = $('#query').value; const m = TYPES.find(t => q.startsWith(t + ' in '));
    if (m) $('#query').value = `${m} in ${where(a)}`;
    preview();
  });
}

function metric(label, value, sub) { const m = el('div', undefined, 'metric'); m.append(el('span', label), el('b', String(value))); if (sub) m.append(el('small', sub)); return m; }
function pct(a, b) { return b ? Math.round(100 * a / b) + '%' : '–'; }
function metrics(run) {
  const box = $('#metrics'); box.replaceChildren();
  const m = run.metrics; if (!m) return;
  box.append(metric('Places', m.places, `in ${(run.elapsed_ms / 1000).toFixed(1)} s`), metric('With phone', m.with_phone, pct(m.with_phone, m.places) + ' of places'),
    metric('Mobile (likely)', m.mobile, 'can receive WhatsApp'), metric('With website', m.with_website, pct(m.with_website, m.places) + ' of places'),
    metric('Shared numbers', m.shared_phone_places, 'places on a common line'));
}

function scatter(run) {
  const W = 420, H = 320, pad = 24, pts = run.places.filter(p => p.latitude && p.longitude);
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'scatter'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Map of returned places around the search pin');
  const lat0 = Number(run.params.lat), lng0 = Number(run.params.lng);
  const kmLat = 111, kmLng = 111 * Math.cos(lat0 * Math.PI / 180);
  const dx = pts.map(p => (p.longitude - lng0) * kmLng), dy = pts.map(p => (p.latitude - lat0) * kmLat);
  const span = Math.max(1, ...dx.map(Math.abs), ...dy.map(Math.abs)) * 1.15;
  const sx = v => W / 2 + (v / span) * (W / 2 - pad), sy = v => H / 2 - (v / span) * (H / 2 - pad);
  const add = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text) e.textContent = text; svg.append(e); return e; };
  add('line', { x1: pad, y1: H / 2, x2: W - pad, y2: H / 2, stroke: '#dfe6df' }); add('line', { x1: W / 2, y1: pad, x2: W / 2, y2: H - pad, stroke: '#dfe6df' });
  pts.forEach((p, i) => { const c = add('circle', { cx: sx(dx[i]), cy: sy(dy[i]), r: 5, fill: p.phone ? '#12644d' : '#c75646', 'fill-opacity': .75 }); const t = document.createElementNS(ns, 'title'); t.textContent = p.title; c.append(t); });
  add('path', { d: `M${W / 2 - 7},${H / 2} h14 M${W / 2},${H / 2 - 7} v14`, stroke: '#15342e', 'stroke-width': 2 });
  add('text', { x: pad, y: H - 6, 'font-size': 10, fill: '#6c7d76' }, `Scale: edge ≈ ${span.toFixed(1)} km from the pin (+) · green = has phone, red = none`);
  return svg;
}

function placeRow(p, i) {
  const tr = el('tr', undefined, 'place'); tr.tabIndex = 0;
  tr.append(el('td', String(i + 1), 'num'));
  const name = el('td', undefined, 'name'); name.append(document.createTextNode(p.title || 'Untitled')); const cat = (p.categories || [])[0]; if (cat) name.append(el('small', cat)); tr.append(name);
  const ph = el('td', undefined, 'ph'); if (p.phone) { ph.append(document.createTextNode(p.phone)); ph.append(el('span', p._phone_kind, 'badge b-' + p._phone_kind)); if (p._phone_shared > 1) ph.append(el('span', 'shared ×' + p._phone_shared, 'badge b-shared')); } else ph.append(el('span', 'none', 'badge b-none')); tr.append(ph);
  tr.append(el('td', p.review_rating ? `${p.review_rating} (${p.review_count})` : '–', 'rate'));
  tr.append(el('td', p.address || ''));
  const web = el('td', undefined, 'web'); const href = safeLink(p.web_site); if (href) { const a = el('a', new URL(href).hostname.replace(/^www\./, '')); a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer'; web.append(a); } else web.textContent = '–'; tr.append(web);
  const detail = el('tr', undefined, 'detailrow'); detail.hidden = true; const dt = el('td'); dt.colSpan = 6; detail.append(dt);
  const toggle = () => {
    detail.hidden = !detail.hidden;
    if (!detail.hidden && !dt.childNodes.length) Object.entries(p).forEach(([k, v]) => {
      const row = el('div', undefined, 'kv' + (filled(v) ? '' : ' empty-field')); row.append(el('strong', k));
      row.append(v !== null && typeof v === 'object' ? el('pre', JSON.stringify(v, null, 1)) : el('span', v === '' || v === null ? '(empty)' : String(v))); dt.append(row);
    });
  };
  tr.addEventListener('click', toggle); tr.addEventListener('keydown', e => { if (e.key === 'Enter') toggle(); });
  return [tr, detail];
}
function renderPlaces(run) {
  const box = $('#places'); box.replaceChildren();
  if (!run.places?.length) { const e = el('div', undefined, 'empty'); e.append(el('h3', 'No places came back'), el('p', run.message || 'Nothing to show.')); box.append(e); return; }
  const grid = el('div', undefined, 'grid2'); const left = el('div'); left.append(el('p', 'Click a row to see every field the scraper returned for that place. Greyed fields came back empty.', 'hint'));
  const wrap = el('div', undefined, 'tablewrap'), t = el('table'), head = el('tr');
  ['#', 'Place', 'Phone', 'Rating', 'Address', 'Website'].forEach(h => head.append(el('th', h)));
  const thead = el('thead'); thead.append(head); t.append(thead); const tb = el('tbody');
  run.places.forEach((p, i) => placeRow(p, i).forEach(r => tb.append(r))); t.append(tb); wrap.append(t); left.append(wrap);
  const right = el('div'); right.append(el('p', 'Where the results are around your search pin', 'hint'), scatter(run));
  grid.append(left, right); box.append(grid);
}
function renderPhones(run) {
  const box = $('#phones'); box.replaceChildren(); if (!run.places?.length) return;
  const m = run.metrics;
  box.append(el('p', `Of ${m.places} places, ${m.with_phone} have a phone: ${m.mobile} look like mobile numbers, ${m.landline} landlines, ${m.toll_free} toll-free. “Kind” is a guess from the number's prefix. Only mobile numbers can receive WhatsApp. Numbers shared by several places are usually a head-office or call-centre line.`, 'hint'));
  const wrap = el('div', undefined, 'tablewrap'), t = el('table'), head = el('tr');
  ['Place', 'Phone as Google shows it', '+91 form', 'Kind', 'Shared by'].forEach(h => head.append(el('th', h)));
  const thead = el('thead'); thead.append(head); t.append(thead); const tb = el('tbody');
  run.places.forEach(p => { const tr = el('tr'); tr.append(el('td', p.title), el('td', p.phone || '–'), el('td', p._phone_e164 || '–')); const k = el('td'); k.append(el('span', KINDS[p._phone_kind], 'badge b-' + p._phone_kind)); tr.append(k, el('td', p._phone_shared > 1 ? p._phone_shared + ' places' : '–')); tb.append(tr); });
  t.append(tb); wrap.append(t); box.append(wrap);
}
function renderAttrs(run) {
  const box = $('#attrs'); box.replaceChildren(); if (!run.places?.length) return;
  const keys = []; run.places.forEach(p => Object.keys(p).filter(k => !k.startsWith('_')).forEach(k => { if (!keys.includes(k)) keys.push(k); }));
  const rows = keys.map(k => [k, run.places.filter(p => filled(p[k])).length]).sort((a, b) => b[1] - a[1]);
  const used = rows.filter(r => r[1] > 0).length;
  box.append(el('p', `${keys.length} attributes are returned per place. ${used} had data in this run; ${keys.length - used} were empty for every place (fast mode collects a reduced set).`, 'hint'));
  const wrap = el('div', undefined, 'tablewrap'), t = el('table'), head = el('tr');
  ['Attribute', 'Filled', 'Coverage', ''].forEach(h => head.append(el('th', h))); const thead = el('thead'); thead.append(head); t.append(thead); const tb = el('tbody');
  rows.forEach(([k, n]) => { const tr = el('tr'); const bar = el('div', undefined, 'bar'), fill = el('i'); fill.style.width = pct(n, run.places.length); bar.append(fill); const bc = el('td'); bc.append(bar); tr.append(el('td', k), el('td', `${n} / ${run.places.length}`), el('td', pct(n, run.places.length)), bc); tb.append(tr); });
  t.append(tb); wrap.append(t); box.append(wrap);
}
function render(run) {
  current = run;
  $('#result-title').textContent = run.params ? `“${run.params.query}” · ${run.places?.length ?? 0} places` : 'Result';
  metrics(run); renderPlaces(run); renderPhones(run); renderAttrs(run);
  $('#raw-text').textContent = run.raw ? JSON.stringify(run.places?.map(p => Object.fromEntries(Object.entries(p).filter(([k]) => !k.startsWith('_')))) ?? [], null, 1) : (run.log || 'No output.');
  $('#export-csv').disabled = $('#export-json').disabled = !run.places?.length;
  if (run.ok) setNotice(`Scrape finished in ${(run.elapsed_ms / 1000).toFixed(1)} s with ${run.places.length} places.`, 'success');
  else setNotice(run.message || 'The scrape failed.' + (run.log ? ' ' + run.log : ''), 'error');
}

function download(name, text, type) { const a = el('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.append(a); a.click(); a.remove(); }
function csvCell(v) { const s = v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
function exportCsv() { const cols = ['title', 'phone', '_phone_e164', '_phone_kind', 'review_rating', 'review_count', 'address', 'web_site', 'latitude', 'longitude']; download('places.csv', [cols.join(',')].concat(current.places.map(p => cols.map(c => csvCell(p[c])).join(','))).join('\n'), 'text/csv'); }
function exportJson() { download('run.json', JSON.stringify({ params: current.params, elapsed_ms: current.elapsed_ms, places: current.places }, null, 2), 'application/json'); }

async function run() {
  const btn = $('#run'); btn.disabled = true; $('#form-error').textContent = '';
  setNotice('Scraping Google Maps… this usually takes 1–10 seconds.');
  try {
    const response = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ params: params() }) });
    const result = await response.json();
    if (!response.ok || result.error) throw Error(result.error || 'Local request failed.');
    render(result); await loadHistory();
  } catch (e) { $('#form-error').textContent = e.message; setNotice(e.message, 'error'); }
  btn.disabled = false;
}

function restore(p) { $('#query').value = p.query || ''; $('#lat').value = p.lat ?? ''; $('#lng').value = p.lng ?? ''; $('#radius').value = p.radius ?? ''; $('#lang').value = p.lang || ''; $('#area').value = 'custom'; preview(); }
function renderHistory() {
  const box = $('#history-list'); box.replaceChildren();
  history.forEach(r => {
    const b = el('button', undefined, 'historyrow'); b.type = 'button';
    b.append(el('b', r.params?.query || '(no query)'), el('small', `${new Date(r.timestamp).toLocaleString()} · ${r.metrics?.places ?? 0} places · ${r.metrics?.with_phone ?? 0} with phone · ${r.ok ? 'ok' : 'failed'}`));
    b.addEventListener('click', async () => { try { const resp = await fetch('/api/history/' + encodeURIComponent(r.id)); if (!resp.ok) throw Error('Could not load the saved run.'); const full = await resp.json(); restore(full.params); render(full); document.querySelector('[data-tab=places]').click(); window.scrollTo({ top: $('#result-title').getBoundingClientRect().top + scrollY - 120, behavior: 'smooth' }); } catch (e) { $('#history-status').textContent = e.message; } });
    box.append(b);
  });
  $('#history-more').hidden = history.length >= historyTotal;
  $('#history-status').textContent = historyTotal ? `${historyTotal} saved run${historyTotal === 1 ? '' : 's'}` : 'No saved runs yet.';
}
async function loadHistory(more) {
  try { const resp = await fetch('/api/history?offset=' + (more ? history.length : 0)); const d = await resp.json(); if (!resp.ok) throw Error(d.error); history = more ? history.concat(d.runs) : d.runs; historyTotal = d.total; renderHistory(); }
  catch (e) { $('#history-status').textContent = e.message; }
}

async function init() {
  presets();
  try {
    const cfg = await (await fetch('/api/config')).json(); areas = cfg.areas; areaSelect(areas); $('#area').value = '0';
    const s = $('#binary-status'); s.textContent = cfg.binary_found ? 'Scraper ready · no key needed' : 'Scraper binary NOT found'; if (!cfg.binary_found) s.classList.add('bad');
  } catch { $('#binary-status').textContent = 'Server unreachable'; }
  ['query', 'lat', 'lng', 'radius'].forEach(id => $('#' + id).addEventListener('input', preview)); $('#lang').addEventListener('change', preview);
  $('#run').addEventListener('click', run);
  $('#export-csv').addEventListener('click', exportCsv); $('#export-json').addEventListener('click', exportJson);
  $('#copy-raw').addEventListener('click', async () => { try { await navigator.clipboard.writeText($('#raw-text').textContent); } catch { $('#form-error').textContent = 'Clipboard unavailable. Select and copy the text manually.'; } });
  $('#history-more').addEventListener('click', () => loadHistory(true));
  document.querySelectorAll('.tabs [role=tab]').forEach(t => t.addEventListener('click', () => {
    document.querySelectorAll('.tabs [role=tab]').forEach(x => x.setAttribute('aria-selected', String(x === t)));
    ['places', 'phones', 'attrs', 'raw', 'history'].forEach(id => { $('#' + id).hidden = id !== t.dataset.tab; });
  }));
  preview(); loadHistory();
}
init();
})();
