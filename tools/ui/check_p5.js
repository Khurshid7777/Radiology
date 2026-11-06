// D210 UI phase 5: axe (WCAG 2.2 AA tags) on four states at 360 and 1280 px; the whole flow by keyboard
// only; reflow at 320 px (= 200 % zoom); interaction latency (Event Timing, CPU 4x); no third-party
// requests; no console errors. Needs axe-core: AXE=<path to axe.min.js>.
//   AXE=node_modules/axe-core/axe.min.js node tools/ui/check_p5.js http://127.0.0.1:8765 <clinic-key>
const fs = require('fs');
const { chromium } = require('playwright');
const [base, key] = [process.argv[2] || 'http://127.0.0.1:8765', process.argv[3] || 'demo-key'];
const AXE = fs.readFileSync(process.env.AXE, 'utf8');
(async () => {
  const b = await chromium.launch(); const out = {}; let fail = 0;
  const check = (name, ok, info = '') => { out[name] = ok ? 'ok' : 'FAIL ' + info; if (!ok) fail++; };
  const axe = async (p, label) => {
    await p.addScriptTag({ content: AXE });
    const r = await p.evaluate(async () => (await axe.run(document, { runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'] })).violations
      .map(v => `${v.id}(${v.impact}) x${v.nodes.length}: ${v.nodes.slice(0, 2).map(n => n.target.join(' ')).join(' | ')}`));
    check(`axe ${label}`, r.length === 0, JSON.stringify(r));
  };
  for (const w of [360, 1280]) {
    const ctx = await b.newContext({ viewport: { width: w, height: 900 } });
    await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
    const p = await ctx.newPage(); const errs = [], foreign = [];
    p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    p.on('request', r => { if (!r.url().startsWith(base) && !r.url().startsWith('blob:') && !r.url().startsWith('data:')) foreign.push(r.url()); });
    await p.goto(base); await axe(p, `${w} step 1`);
    // --- keyboard only: focus + Enter / Space / arrows, never a mouse click
    const press = async (sel, k = 'Enter') => { await p.focus(sel); await p.keyboard.press(k); };
    if (await p.isVisible('#next1')) await press('#next1');
    await axe(p, `${w} step 2`);
    await p.focus('#age'); await p.keyboard.type('54'); await press('#sex_m', 'Space');
    await press('#svc summary');
    await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
    await p.focus('#case'); await p.selectOption('#case', 'stemi');       // a select is keyboard-native
    await press('#run');
    await p.waitForSelector('#results:not(.hidden)'); await p.waitForTimeout(800);
    check(`${w}: flow by keyboard reaches the result`, await p.isVisible('#lay1'));
    await p.waitForFunction(() => document.querySelector('#pdf').getAttribute('href'));
    const head = await p.evaluate(async () => { const b = await (await fetch(document.querySelector('#pdf').href)).arrayBuffer();
      return String.fromCharCode(...new Uint8Array(b).slice(0, 4)); });
    check(`${w}: smoke — the PDF is a PDF`, head === '%PDF', head);
    await axe(p, `${w} result`);
    await p.focus('#tab_res'); await p.keyboard.press('ArrowRight');
    await p.waitForTimeout(500);
    check(`${w}: arrow key switches to the laboratory`, await p.isVisible('#lay2'));
    await axe(p, `${w} laboratory`);
    const ring = await p.evaluate(() => { const el = document.querySelector('#tab_lab'); el.focus(); const s = getComputedStyle(el);
      return s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) >= 2; });
    check(`${w}: visible focus ring`, ring);
    check(`${w}: no third-party requests`, foreign.length === 0, JSON.stringify(foreign));
    check(`${w}: no console errors`, errs.length === 0, JSON.stringify(errs));
    await ctx.close();
  }
  // Reflow at 320 px (WCAG 1.4.10 = 1280 px at 400 %, 640 px at 200 %): no horizontal scroll on any step.
  for (const w of [320, 640]) {
    const ctx = await b.newContext({ viewport: { width: w, height: 800 } });
    await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
    const p = await ctx.newPage(); await p.goto(base);
    const widths = [];
    for (const s of ['1', '2', '3']) { await p.evaluate(x => { document.body.dataset.step = x; document.querySelector('#results').classList.remove('hidden'); }, s);
      widths.push(await p.evaluate(() => document.documentElement.scrollWidth)); }
    check(`reflow ${w} px: no horizontal scroll`, widths.every(x => x <= w), JSON.stringify(widths));
    await ctx.close();
  }
  // Interaction latency (INP proxy): the longest Event Timing entry over the main actions, CPU 4x slower.
  const ctx = await b.newContext({ viewport: { width: 412, height: 915 } });
  await ctx.addInitScript(k => { localStorage.setItem('cardiolens_key', k); window.__ev = [];
    new PerformanceObserver(l => l.getEntries().forEach(e => window.__ev.push({ n: e.name, d: e.duration }))).observe({ type: 'event', durationThreshold: 16, buffered: true }); }, key);
  const p = await ctx.newPage(); const cdp = await ctx.newCDPSession(p);
  await p.goto(base); await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  await p.click('#next1'); await p.click('#sex_m'); await p.click('#mode_b'); await p.click('#mode_a');
  await p.evaluate(() => { document.querySelector('#svc').open = true; });
  await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
  await p.selectOption('#case', 'stemi'); await p.fill('#age', '54'); await p.click('#run');
  await p.waitForSelector('#results:not(.hidden)'); await p.waitForTimeout(800);
  await p.click('#tab_lab'); await p.waitForTimeout(600); await p.click('#tab_res'); await p.click('#cl_uz'); await p.click('#cl_ru');
  const ev = await p.evaluate(() => window.__ev); const worst = ev.reduce((m, e) => e.d > m.d ? e : m, { d: 0, n: '-' });
  check(`interaction latency at CPU x4 <= 200 ms (worst ${Math.round(worst.d)} ms, ${worst.n})`, worst.d <= 200);
  await b.close(); console.log(JSON.stringify(out, null, 1)); process.exit(fail ? 1 : 0);
})();
