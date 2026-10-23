import { STRINGS, LANGS } from './i18n.js';
import { loadModel, prepare, diagnose, camFor, paintHeat, benchmark, backendName } from './model.js';

// ---------- state ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
};
let lang = store.get('lang', 'en');
if (!STRINGS[lang]) lang = 'en';
let cards = null, game = null, meta = null, PHOTOS = {};
const photo = f => PHOTOS[f] || f;    // bundled photos are kept as data URLs in one file
let modelState = 'loading';
let lastCheck = null;        // { img, res, when, source }
let speedMs = null;
let offlineReady = false;
let installPrompt = null;
const OPTION_ORDER = ['Healthy', 'Early_blight', 'Late_blight', 'Spotted Wilt Virus', 'Leaf Miner', 'Magnesium Deficiency', 'Nitrogen Deficiency'];
const GAME = { lab: 5, field: 3 };

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function t(key, vars = {}) {
  let s = (STRINGS[lang] && STRINGS[lang][key]) || STRINGS.en[key] || key;
  for (const [k, v] of Object.entries(vars)) s = s.split('{' + k + '}').join(v);
  return s;
}
const loc = o => (o && typeof o === 'object' && !Array.isArray(o)) ? (o[lang] ?? o.en) : o;
const title = cls => loc(cards[cls].title);
const pct = p => Math.round(p * 100);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const TH = () => (meta ? meta.threshold : 0.85);

// ---------- model ----------
const modelReady = loadModel().then(m => { meta = m; modelState = 'ready'; renderStatus(); renderAboutIfOpen(); return m; })
  .catch(err => { console.error(err); modelState = 'failed'; renderStatus(); throw err; });

// ---------- language ----------
function renderLangButtons() {
  $('#lang').innerHTML = LANGS.map(l => `<button type="button" data-lang="${l.code}" aria-pressed="${l.code === lang}" lang="${l.code}">${l.label}</button>`).join('');
}
function setLang(code) {
  lang = code; store.set('lang', code);
  document.documentElement.lang = code;
  stopSpeaking();
  renderLangButtons(); applyStatic(); renderStatus(); renderSamples();
  if (lastCheck) renderCheckResult();
  renderGame(); renderAbout();
}
function applyStatic() {
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.title = t('app_name');
}

// ---------- status line ----------
function renderStatus() {
  const m = modelState === 'ready' ? `<span><i class="dot ok"></i>${esc(t('status_ready'))}</span>`
    : modelState === 'failed' ? `<span><i class="dot bad"></i>${esc(t('status_failed'))}</span>`
    : `<span><i class="dot"></i>${esc(t('status_loading'))}</span>`;
  const o = ('serviceWorker' in navigator) ? (offlineReady ? `<span><i class="dot ok"></i>${esc(t('offline_ready'))}</span>` : `<span><i class="dot"></i>${esc(t('offline_saving'))}</span>`) : '';
  $('#status').innerHTML = m + o;
}

// ---------- tabs ----------
function showView(name) {
  if (!['check', 'game', 'about'].includes(name)) name = 'check';
  document.querySelectorAll('.view').forEach(v => { v.hidden = v.id !== 'view-' + name; });
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.view === name)));
  if (name === 'about') renderAbout();
  window.scrollTo(0, 0);
}

// ---------- photos ----------
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('image'));
    img.src = src;
  });
}

// A mounted photo with a heatmap layer on top.
function makeSpecimen(img, vh = 42) {
  const wrap = document.createElement('div'); wrap.className = 'specimen';
  const w = img.naturalWidth, h = img.naturalHeight, ar = w / h;
  const dw = Math.min(900, Math.max(w, 448)), dh = Math.round(dw / ar);
  const stack = document.createElement('div'); stack.className = 'stack';
  stack.style.width = `min(100%, calc(${vh}vh * ${ar.toFixed(4)}))`;
  const photo = document.createElement('canvas'); photo.width = dw; photo.height = dh;
  const pctx = photo.getContext('2d'); pctx.imageSmoothingQuality = 'high'; pctx.drawImage(img, 0, 0, dw, dh);
  const heat = document.createElement('canvas'); heat.className = 'heat';
  heat.width = 280; heat.height = Math.max(1, Math.round(280 / ar));
  heat.setAttribute('aria-hidden', 'true');
  stack.append(photo, heat); wrap.append(stack);
  return {
    el: wrap,
    setCam(cam) { paintHeat(heat, cam); },
    showHeat(on) { wrap.classList.toggle('show-heat', on); }
  };
}

// ---------- check a leaf ----------
function renderSamples() {
  if (!game || !cards) return;
  $('#samples').innerHTML = game.samples.map((s, i) => {
    const name = s.source.startsWith('PlantDoc') ? t('sample_field') : title(s.label);
    return `<button type="button" class="sample" data-sample="${i}"><img src="${photo(s.file)}" alt=""><span>${esc(name)}</span></button>`;
  }).join('');
}

async function checkPhoto(src, source) {
  $('#check-start').hidden = true;
  const box = $('#check-result'); box.hidden = false;
  box.innerHTML = `<div class="busy"><div class="spinner"></div>${esc(t('reading'))}</div>`;
  window.scrollTo(0, 0);
  let img;
  try { img = await loadImage(src); } catch (e) {
    box.innerHTML = `<div class="error">${esc(t('bad_image'))}</div><button class="btn wide" type="button" data-act="again">${esc(t('check_another'))}</button>`;
    return;
  }
  try { await modelReady; } catch (e) { box.innerHTML = `<div class="error">${esc(t('status_failed'))}</div>`; return; }
  const res = await diagnose(prepare(img));
  lastCheck = { img, res, when: new Date(), source, showHeat: false };
  renderCheckResult();
}

function cardSections(cls, { openFirst = true } = {}) {
  const c = cards[cls], m = cards._meta;
  const list = arr => `<ul>${loc(arr).map(x => `<li>${esc(x)}</li>`).join('')}</ul>`;
  let html = `<div class="advice">`;
  html += `<details ${openFirst ? 'open' : ''}><summary>${esc(t('sec_now'))}</summary><div class="body">${list(c.act_now)}</div></details>`;
  if (c.sprays.length) {
    const rows = c.sprays.map(s => `<tr><td>${esc(s.product)}</td><td>${esc(loc(s.dose))}</td></tr>`).join('');
    html += `<details><summary>${esc(t('sec_sprays'))}</summary><div class="body"><div class="warn">${esc(loc(m.spray_warning))}</div>
      <table class="spray"><thead><tr><th>${esc(t('product'))}</th><th>${esc(t('dose'))}</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
  }
  html += `<details><summary>${esc(t('sec_prevent'))}</summary><div class="body">${list(c.prevent)}</div></details>`;
  html += `<details><summary>${esc(t('sec_signs'))}</summary><div class="body">${list(c.signs)}</div></details>`;
  html += `<details><summary>${esc(t('sec_expert'))}</summary><div class="body"><p>${esc(loc(c.expert))}</p></div></details>`;
  const links = c.sources.map(k => `<a href="${m.sources[k].url}" target="_blank" rel="noopener">${esc(m.sources[k].title)}</a>`).join('; ');
  html += `<p class="sources">${esc(t('advice_from'))}: ${links}</p></div>`;
  return html;
}

function ruler(conf, cls) {
  const th = pct(TH());
  return `<div class="ruler">
    <div class="ruler-top"><b>${esc(t('sure_pct', { p: pct(conf) }))}</b></div>
    <div class="ruler-track" role="img" aria-label="${esc(t('sure_pct', { p: pct(conf) }))}, ${esc(t('sure_line', { t: th }))}">
      <div class="ruler-scale"></div>
      <div class="ruler-fill ${cls}" data-w="${(conf * 100).toFixed(1)}"></div>
      <div class="ruler-mark" style="left:${th}%"><span>${esc(t('sure_line', { t: th }))}</span></div>
    </div><div class="ruler-legend"></div>
  </div>`;
}

function renderCheckResult() {
  const { img, res, when } = lastCheck;
  const box = $('#check-result');
  const top = meta.classes[res.order[0]], conf = res.probs[res.order[0]], sure = conf >= TH();
  const sp = makeSpecimen(img);
  sp.setCam(camFor(res.features, res.order[0]));
  sp.showHeat(lastCheck.showHeat);
  const timeStr = when.toLocaleString(lang === 'en' ? 'en-IN' : lang + '-IN', { dateStyle: 'medium', timeStyle: 'short' });
  let html = `<div class="seg" role="group"><button type="button" data-heat="0" aria-pressed="${!lastCheck.showHeat}">${esc(t('view_photo'))}</button><button type="button" data-heat="1" aria-pressed="${lastCheck.showHeat}">${esc(t('view_heat'))}</button></div>
    <p class="heat-caption" ${lastCheck.showHeat ? '' : 'hidden'}>${esc(t('heat_caption'))}</p>`;
  if (sure) {
    const c = cards[top], u = c.urgency;
    html += `<div class="slip ${u}">
      <div class="slip-row"><span>${esc(t('slip_checked'))}</span><span>${esc(timeStr)}</span></div>
      <div class="slip-row"><span>${esc(t('slip_finding'))}</span><span>${esc(loc(c.kind))}</span></div>
      <h2 class="slip-title">${esc(title(top))}</h2>
      ${loc(c.cause) ? `<p class="slip-kind">${esc(loc(c.cause))}</p>` : ''}
      ${ruler(conf, '')}
      <span class="chip ${u}">${esc(t('urgency_' + u))}</span>
    </div>`;
    html += cardSections(top);
  } else {
    const u = cards._uncertain, a = res.order[0], b = res.order[1];
    html += `<div class="slip unsure">
      <div class="slip-row"><span>${esc(t('slip_checked'))}</span><span>${esc(timeStr)}</span></div>
      <h2 class="slip-title">${esc(loc(u.title))}</h2>
      <p class="slip-kind">${esc(loc(u.message))}</p>
      ${ruler(conf, '')}
      <p style="margin:.4rem .2rem 0">${esc(t('best_guesses', { a: title(meta.classes[a]), pa: pct(res.probs[a]), b: title(meta.classes[b]), pb: pct(res.probs[b]) }))}</p>
    </div>
    <h3 class="h3">${esc(t('better_photo'))}</h3><ul>${loc(u.retake_tips).map(x => `<li>${esc(x)}</li>`).join('')}</ul>
    <p class="small">${esc(loc(u.expert))}</p>`;
    for (const k of [a, b]) {
      html += `<h3 class="h3">${esc(t('if_it_is', { x: title(meta.classes[k]) }))}</h3>${cardSections(meta.classes[k], { openFirst: false })}`;
    }
  }
  html += `<div class="row-btns">
      ${canSpeak() ? `<button class="btn quiet" type="button" data-act="speak">${esc(t('read_aloud'))}</button>` : ''}
      <button class="btn" type="button" data-act="again">${esc(t('check_another'))}</button>
    </div>
    <h3 class="h3">${esc(t('all_matches'))}</h3><div class="bars">${res.order.map(i => `
      <div class="barline"><span>${esc(title(meta.classes[i]))}</span><span class="pct">${(res.probs[i] * 100).toFixed(1)}%</span>
      <div class="track"><i style="width:${(res.probs[i] * 100).toFixed(1)}%"></i></div></div>`).join('')}</div>
    <p class="foot">${esc(t('worked_out', { s: (res.ms / 1000).toFixed(2) }))}<br>${esc(loc(cards._meta.app_warning))}</p>`;
  box.innerHTML = '';
  box.append(sp.el);
  box.insertAdjacentHTML('beforeend', html);
  requestAnimationFrame(() => requestAnimationFrame(() => box.querySelectorAll('.ruler-fill').forEach(f => { f.style.width = f.dataset.w + '%'; })));
  lastCheck.specimen = sp;
}

function resetCheck() {
  stopSpeaking();
  lastCheck = null;
  $('#check-result').hidden = true; $('#check-result').innerHTML = '';
  $('#check-start').hidden = false;
  $('#in-camera').value = ''; $('#in-gallery').value = '';
}

// ---------- read aloud ----------
const VOICE_LANG = { en: ['en-IN', 'en'], hi: ['hi-IN', 'hi'], mr: ['mr-IN', 'mr', 'hi-IN', 'hi'] };
function pickVoice() {
  if (!('speechSynthesis' in window)) return null;
  const voices = speechSynthesis.getVoices();
  for (const want of VOICE_LANG[lang]) {
    const v = voices.find(x => x.lang.replace('_', '-').toLowerCase().startsWith(want.toLowerCase()));
    if (v) return v;
  }
  return null;
}
function canSpeak() { return !!pickVoice(); }
let speaking = false;
function speakResult() {
  if (speaking) { stopSpeaking(); return; }
  const { res } = lastCheck, top = meta.classes[res.order[0]], conf = res.probs[res.order[0]];
  let parts;
  if (conf >= TH()) {
    const c = cards[top];
    parts = [title(top), t('sure_pct', { p: pct(conf) }), t('urgency_' + c.urgency), t('sec_now'), ...loc(c.act_now)];
  } else {
    const u = cards._uncertain;
    parts = [loc(u.title), loc(u.message), t('better_photo'), ...loc(u.retake_tips)];
  }
  const v = pickVoice(), u = new SpeechSynthesisUtterance(parts.join('. '));
  u.voice = v; u.lang = v.lang; u.rate = 0.95;
  u.onend = u.onerror = () => { speaking = false; updateSpeakBtn(); };
  speechSynthesis.cancel(); speechSynthesis.speak(u); speaking = true; updateSpeakBtn();
}
function stopSpeaking() { if ('speechSynthesis' in window) speechSynthesis.cancel(); speaking = false; updateSpeakBtn(); }
function updateSpeakBtn() { const b = document.querySelector('[data-act="speak"]'); if (b) b.textContent = speaking ? t('stop_reading') : t('read_aloud'); }
if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = () => { if (lastCheck && !document.querySelector('[data-act="speak"]')) renderCheckResult(); };

// ---------- the game ----------
let G = { phase: 'intro' };
const shuffle = a => { const b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; };

function newGame() {
  const byClass = {};
  for (const it of game.lab) (byClass[it.label] ||= []).push(it);
  const labClasses = shuffle(Object.keys(byClass)).slice(0, GAME.lab);
  const lab = labClasses.map(c => ({ ...shuffle(byClass[c])[0], stage: 'lab' }));
  const fByClass = {};
  for (const it of game.field) (fByClass[it.label] ||= []).push(it);
  const field = shuffle(Object.keys(fByClass)).slice(0, GAME.field).map(c => ({ ...shuffle(fByClass[c])[0], stage: 'field' }));
  G = { phase: 'round', rounds: [...lab, ...field], i: 0, you: 0, ai: 0, log: [] };
  renderGame();
}

function scoreBar(bump) {
  return `<div class="score" aria-live="polite">
    <div class="side you"><span class="who">${esc(t('you'))}</span><span class="n ${bump === 'you' ? 'bump' : ''}">${G.you}</span></div>
    <div class="vs"></div>
    <div class="side ai"><span class="n ${bump === 'ai' ? 'bump' : ''}">${G.ai}</span><span class="who">${esc(t('ai'))}</span></div>
  </div>`;
}
function progressBar() {
  return `<div class="progress">${G.rounds.map((r, k) => `<i class="${r.stage} ${k < G.i || (k === G.i && G.phase === 'reveal') ? 'done' : ''} ${k === G.i && G.phase === 'round' ? 'now' : ''}"></i>`).join('')}</div>`;
}

function renderGame() {
  const v = $('#view-game');
  if (!cards || !game) return;
  if (G.phase === 'intro') {
    const best = store.get('best', null);
    v.innerHTML = `<h1 class="h1">${esc(t('game_title'))}</h1>
      <p class="lead">${esc(t('game_lead', { n: GAME.lab + GAME.field }))}</p>
      <div class="intro-stage"><span class="num lab">1</span><div><b>${esc(t('stage1_name'))}</b><br><span class="small">${esc(t('stage1_desc', { n: GAME.lab }))}</span></div></div>
      <div class="intro-stage"><span class="num field">2</span><div><b>${esc(t('stage2_name'))}</b><br><span class="small">${esc(t('stage2_desc', { n: GAME.field }))}</span></div></div>
      <button class="btn primary wide" type="button" data-game="start">${esc(t('start_game'))}</button>
      ${best ? `<p class="small" style="margin-top:.8rem;text-align:center">${esc(t('best_score', best))}</p>` : ''}`;
    return;
  }
  if (G.phase === 'stage2') {
    v.innerHTML = `${scoreBar()}${progressBar()}<div class="mount" style="margin-top:1rem">
      <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <h2 class="h1">${esc(t('stage2_intro_title'))}</h2><p class="lead">${esc(t('stage2_intro'))}</p>
      <button class="btn primary wide" type="button" data-game="continue">${esc(t('continue'))}</button></div>`;
    return;
  }
  if (G.phase === 'results') { renderResults(v); return; }

  // round or reveal
  const r = G.rounds[G.i], labRounds = G.rounds.filter(x => x.stage === 'lab').length;
  const stageName = r.stage === 'lab' ? t('stage1_name') : t('stage2_name');
  const idx = r.stage === 'lab' ? G.i + 1 : G.i - labRounds + 1, n = r.stage === 'lab' ? labRounds : G.rounds.length - labRounds;
  const log = G.log[G.i];
  let body;
  if (!log) {
    body = `<h2 class="h3">${esc(t('question'))}</h2>
    ${G.busy ? `<div class="thinking"><div class="spinner"></div>${esc(t('ai_thinking'))}</div>` : ''}
    <div class="options">${OPTION_ORDER.map(c => `<button type="button" class="opt ${G.busy && c === G.chosen ? 'chosen' : ''}" data-opt="${esc(c)}" ${G.busy ? 'disabled' : ''}>
      <b>${esc(title(c))}</b><small>${esc(loc(cards[c].hint))}</small>${G.busy && c === G.chosen ? `<span class="tag"><span class="t-you">${esc(t('you'))}</span></span>` : ''}</button>`).join('')}</div>`;
  } else {
    const last = G.i === G.rounds.length - 1, beforeStage2 = r.stage === 'lab' && G.rounds[G.i + 1] && G.rounds[G.i + 1].stage === 'field';
    G.nextPhase = last ? 'results' : beforeStage2 ? 'stage2' : 'round';
    body = `<div class="reveal">
      <p class="answer">${esc(t('it_was', { x: title(r.label) }))}</p>
      <div class="verdict"><span class="mark ${log.you === r.label ? 'ok' : 'no'}">${log.you === r.label ? '✓' : '✗'}</span><span>${esc(t('you_said', { x: title(log.you) }))}</span></div>
      <div class="verdict"><span class="mark ${log.ai === r.label ? 'ok' : 'no'}">${log.ai === r.label ? '✓' : '✗'}</span><span>${esc(t('ai_said', { x: title(log.ai), p: pct(log.conf) }))}${log.conf < TH() ? `<br><span class="note">${esc(t('ai_unsure_note'))}</span>` : ''}</span></div>
      <p class="tell"><b>${esc(t('how_to_tell'))}:</b> ${esc(loc(cards[r.label].signs)[0])}</p>
      <div class="row-btns" style="margin-bottom:0">
        <button class="btn quiet" type="button" data-game="heat">${esc(G.heat ? t('hide_heat') : t('show_heat'))}</button>
        <button class="btn primary" type="button" data-game="next">${esc(last ? t('see_results') : t('next_leaf'))}</button>
      </div></div>`;
  }
  v.innerHTML = `${scoreBar(G.bump)}${progressBar()}<div class="stage-label">${esc(t('leaf_of', { stage: stageName, i: idx, n }))}</div><div id="g-spec"></div>${body}`;
  G.bump = null;
  const specFor = G.i;
  loadImage(photo(r.file)).then(img => {
    if (specFor !== G.i || !$('#g-spec')) return;
    const sp = makeSpecimen(img, 34); G.spec = sp;
    $('#g-spec').replaceChildren(sp.el);
    if (log) { sp.setCam(log.cam); sp.showHeat(!!G.heat); }
  });
}

async function answer(choice) {
  if (G.busy || G.log[G.i]) return;
  G.busy = true; G.chosen = choice; renderGame();
  const r = G.rounds[G.i];
  try { await modelReady; } catch (e) { G.busy = false; return; }
  const img = await loadImage(photo(r.file));
  const [res] = await Promise.all([diagnose(prepare(img)), sleep(650)]);
  const ai = meta.classes[res.order[0]], conf = res.probs[res.order[0]];
  const youRight = choice === r.label, aiRight = ai === r.label;
  if (youRight) G.you++;
  if (aiRight) G.ai++;
  G.bump = youRight && aiRight ? null : youRight ? 'you' : aiRight ? 'ai' : null;
  G.log[G.i] = { you: choice, ai, conf, cam: camFor(res.features, res.order[0]), stage: r.stage, label: r.label };
  G.busy = false; G.phase = 'reveal'; G.heat = false;
  if (navigator.vibrate) navigator.vibrate(youRight ? 25 : [20, 60, 20]);
  renderGame();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function renderResults(v) {
  const L = G.log, lab = L.filter(x => x.stage === 'lab'), field = L.filter(x => x.stage === 'field');
  const cnt = (a, who) => a.filter(x => x[who] === x.label).length;
  const sureWrong = L.filter(x => x.ai !== x.label && x.conf >= TH()).length;
  const anyWrong = L.some(x => x.ai !== x.label);
  const verdict = G.you > G.ai ? t('result_you_win') : G.you < G.ai ? t('result_ai_wins') : t('result_draw');
  const s = meta.stats, th = pct(TH());
  v.innerHTML = `<div class="final"><div class="verdict-line">${esc(verdict)}</div>
      <div class="big"><span class="y">${G.you}</span><span class="sep">|</span><span class="a">${G.ai}</span></div>
      <div class="small">${esc(t('you'))} | ${esc(t('ai'))}</div></div>
    <table class="stages-table"><thead><tr><th></th><th>${esc(t('you'))}</th><th>${esc(t('ai'))}</th></tr></thead><tbody>
      <tr><td>${esc(t('stage1_name'))}</td><td>${cnt(lab, 'you')} / ${lab.length}</td><td>${cnt(lab, 'ai')} / ${lab.length}</td></tr>
      <tr><td>${esc(t('stage2_name'))}</td><td>${cnt(field, 'you')} / ${field.length}</td><td>${cnt(field, 'ai')} / ${field.length}</td></tr>
    </tbody></table>
    <h2 class="h3">${esc(t('what_this_shows'))}</h2>
    <ul class="lessons">
      <li>${esc(t('lesson_lab', { ai: cnt(lab, 'ai'), n: lab.length, acc: pct(s.test_accuracy) }))}</li>
      <li>${esc(t('lesson_field', { ai: cnt(field, 'ai'), n: field.length, acc: pct(s.plantdoc_accuracy) }))}</li>
      ${anyWrong ? `<li>${esc(sureWrong ? t('lesson_sure', { k: sureWrong, t: th }) : t('lesson_sure_none', { t: th }))}</li>` : ''}
    </ul>
    <button class="btn primary wide" type="button" data-game="start">${esc(t('play_again'))}</button>`;
}

function nextRound() {
  if (G.nextPhase === 'results') {
    G.phase = 'results';
    const best = store.get('best', null);
    if (!best || G.you > best.you || (G.you === best.you && G.ai < best.ai)) store.set('best', { you: G.you, ai: G.ai });
  } else if (G.nextPhase === 'stage2') { G.i++; G.phase = 'stage2'; }
  else { G.i++; G.phase = 'round'; }
  renderGame(); window.scrollTo(0, 0);
}

// ---------- about ----------
function renderAboutIfOpen() { if (!$('#view-about').hidden) renderAbout(); }
function renderAbout() {
  const v = $('#view-about');
  if (!cards) return;
  const s = meta ? meta.stats : null, th = pct(TH());
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  const mb = meta ? (meta.model_bytes / 1e6).toFixed(1) + ' MB' : '…';
  v.innerHTML = `<h1 class="h1">${esc(t('about_title'))}</h1><p class="lead">${esc(t('about_lead'))}</p>
    <ol class="steps">${['step1', 'step2', 'step3', 'step4', 'step5', 'step6'].map(k => `<li><span>${esc(t(k, { t: th }))}</span></li>`).join('')}</ol>
    <h2 class="h2">${esc(t('how_well'))}</h2>
    ${s ? `<table class="metrics"><tbody>
      <tr><td>${esc(t('m_lab'))}</td><td>${esc(t('m_lab_v', { acc: (s.test_accuracy * 100).toFixed(1), f1: s.test_macro_f1.toFixed(2) }))}</td></tr>
      <tr><td>${esc(t('m_gate', { t: th }))}</td><td>${esc(t('m_gate_v', { share: pct(s.threshold_answered_share), acc: pct(s.threshold_accuracy_on_answered) }))}</td></tr>
      <tr><td>${esc(t('m_field'))}</td><td>${esc(t('m_field_v', { f1: s.field_cv_macro_f1.toFixed(2) }))}</td></tr>
      <tr class="bad"><td>${esc(t('m_pd'))}</td><td>${esc(t('m_pd_v', { acc: (s.plantdoc_accuracy * 100).toFixed(1) }))}</td></tr>
    </tbody></table>` : ''}
    <h2 class="h2">${esc(t('on_phone'))}</h2>
    <div class="panel"><table class="metrics"><tbody>
      <tr><td>${esc(t('p_size'))}</td><td>${mb}</td></tr>
      <tr><td>${esc(t('p_backend'))}</td><td>${esc(backendName ? (backendName === 'wasm' ? 'WebAssembly (CPU)' : backendName === 'webgl' ? 'WebGL (GPU)' : 'CPU') : '…')}</td></tr>
      <tr><td>${esc(t('p_speed'))}</td><td id="speed-cell">${esc(speedMs ? t('speed_result', { ms: Math.round(speedMs) }) : t('not_measured'))}</td></tr>
      <tr><td>${esc(t('p_offline'))}</td><td>${esc(offlineReady ? t('yes') : t('not_yet'))}</td></tr>
    </tbody></table>
    <div class="row-btns" style="margin-bottom:0">
      <button class="btn" type="button" data-act="speed" ${modelState !== 'ready' ? 'disabled' : ''}>${esc(t('run_speed'))}</button>
      ${standalone ? `<button class="btn quiet" type="button" disabled>${esc(t('installed'))}</button>`
        : installPrompt ? `<button class="btn primary" type="button" data-act="install">${esc(t('install'))}</button>` : ''}
    </div>
    ${!standalone && !installPrompt && ios ? `<p class="small" style="margin-top:.7rem">${esc(t('install_ios'))}</p>` : ''}
    </div>
    <h2 class="h2">${esc(t('data_title'))}</h2><p>${esc(t('data_body'))}</p>
    <p class="warn">${esc(t('disclaimer'))}</p>
    <p class="small">${esc(t('credit'))}${lang !== 'en' ? '<br>' + esc(t('lang_note')) : ''}</p>`;
}

async function runSpeedTest(btn) {
  btn.disabled = true; $('#speed-cell').textContent = t('speed_running');
  await modelReady;
  const img = await loadImage(photo(game.samples[0].file));
  speedMs = await benchmark(prepare(img), 10);
  renderAbout();
}

// ---------- offline + install ----------
async function checkOffline() {
  if (!('caches' in window)) return;
  try {
    const hit = await caches.match('model/group1-shard1of1.bin', { ignoreSearch: true });
    offlineReady = !!hit && !!navigator.serviceWorker.controller;
  } catch (e) { offlineReady = false; }
  renderStatus(); renderAboutIfOpen();
}
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').then(() => navigator.serviceWorker.ready).then(checkOffline).catch(() => {});
    navigator.serviceWorker.addEventListener('message', e => { if (e.data === 'cached') checkOffline(); });
    navigator.serviceWorker.addEventListener('controllerchange', checkOffline);
  });
}
window.addEventListener('beforeinstallprompt', e => { e.preventDefault(); installPrompt = e; renderAboutIfOpen(); });
window.addEventListener('appinstalled', () => { installPrompt = null; renderAboutIfOpen(); });

// ---------- events ----------
document.addEventListener('click', async e => {
  const el = e.target.closest('button, [data-sample]');
  if (!el) return;
  if (el.dataset.lang) setLang(el.dataset.lang);
  else if (el.dataset.view) { location.hash = el.dataset.view; }
  else if (el.dataset.sample !== undefined) { const s = game.samples[+el.dataset.sample]; checkPhoto(photo(s.file), 'sample'); }
  else if (el.dataset.heat !== undefined && lastCheck) {
    lastCheck.showHeat = el.dataset.heat === '1';
    lastCheck.specimen.showHeat(lastCheck.showHeat);
    document.querySelectorAll('[data-heat]').forEach(b => b.setAttribute('aria-pressed', String(b === el)));
    document.querySelector('.heat-caption').hidden = !lastCheck.showHeat;
  }
  else if (el.dataset.act === 'again') resetCheck();
  else if (el.dataset.act === 'speak') speakResult();
  else if (el.dataset.act === 'speed') runSpeedTest(el);
  else if (el.dataset.act === 'install' && installPrompt) { installPrompt.prompt(); await installPrompt.userChoice; installPrompt = null; renderAbout(); }
  else if (el.dataset.opt) answer(el.dataset.opt);
  else if (el.dataset.game === 'start') { newGame(); window.scrollTo(0, 0); }
  else if (el.dataset.game === 'continue') { G.phase = 'round'; renderGame(); window.scrollTo(0, 0); }
  else if (el.dataset.game === 'next') nextRound();
  else if (el.dataset.game === 'heat') { G.heat = !G.heat; G.spec && G.spec.showHeat(G.heat); el.textContent = G.heat ? t('hide_heat') : t('show_heat'); }
});
for (const id of ['#in-camera', '#in-gallery']) {
  $(id).addEventListener('change', e => {
    const f = e.target.files && e.target.files[0];
    if (f) checkPhoto(URL.createObjectURL(f), 'user');
  });
}
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

// For testing from the browser console.
window.leafDebug = { modelReady, prepare, diagnose, loadImage, camFor, photo };

// ---------- start ----------
(async function init() {
  document.documentElement.lang = lang;
  renderLangButtons(); applyStatic(); renderStatus();
  const [c, g, ph] = await Promise.all(['data/cards.json', 'data/game.json', 'data/photos.json'].map(u => fetch(u).then(r => r.json())));
  cards = c; game = g; PHOTOS = ph;
  renderSamples(); renderGame();
  showView(location.hash.slice(1));
})();
