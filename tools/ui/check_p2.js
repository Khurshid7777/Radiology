// D210 UI P2 check (Playwright, demo case only): the key values in the DOM are the API's values; every
// quality warning is in the first layer; the layers switch; "new patient" clears; no console errors.
//   node tools/ui/check_p2.js http://127.0.0.1:8765 <clinic-key>
const { chromium } = require('playwright');
const [base, key] = [process.argv[2] || 'http://127.0.0.1:8765', process.argv[3] || 'demo-key'];
(async () => {
  const b = await chromium.launch(); const out = {}; let fail = 0;
  const check = (name, ok, info = '') => { out[name] = ok ? 'ok' : 'FAIL ' + info; if (!ok) fail++; };
  for (const [w, h, rm] of [[1280, 800, 'no-preference'], [360, 800, 'reduce']]) {
    const ctx = await b.newContext({ viewport: { width: w, height: h }, reducedMotion: rm });
    await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
    const p = await ctx.newPage(); const errs = [];
    p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await p.goto(base);
    if (await p.isVisible('#next1')) await p.click('#next1');
    await p.evaluate(() => { document.querySelector('#svc').open = true; });
    await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
    await p.selectOption('#case', 'stemi');
    await p.fill('#age', '54'); await p.click('#sex_m');
    const respP = p.waitForResponse(r => r.url().endsWith('/api/analyze'));
    await p.click('#run');
    const api = await (await respP).json();
    const t0 = Date.now(); await p.waitForSelector('#results:not(.hidden)'); const shownMs = Date.now() - t0;
    check(`${w}: result shown <= 400 ms after the response`, shownMs <= 400, shownMs + ' ms');
    const m = api.measurements, iv = m.intervals;
    const want = [m.heart_rate_bpm, iv.pr_ms, iv.qrs_ms, iv.qtc_bazett_ms, m.axes.qrs_deg].filter(v => v != null).map(String);
    await p.waitForTimeout(700);   // the count-up ends on the API's text (P3)
    const got = await p.$$eval('#keys .v', els => els.map(e => e.textContent));
    check(`${w}: key values == API`, JSON.stringify(got) === JSON.stringify(want), JSON.stringify({ got, want }));
    const q = (api.findings || []).filter(f => f.group === 'quality').length + (api.speed_note_ru ? 1 : 0) + (api.device_ru ? 1 : 0);
    check(`${w}: every warning in layer 1`, (await p.$$eval('#lay1 #warnbox .notice', e => e.length)) === q);
    check(`${w}: criticals in layer 1`, (await p.$$eval('#lay1 #critical .crit', e => e.length)) === (api.critical_alerts || []).length);
    check(`${w}: disclaimer in layer 1`, (await p.textContent('#lay1 #disc')).length > 10);
    check(`${w}: lab hidden first`, await p.isHidden('#lay2'));
    await p.click('#tab_lab'); check(`${w}: lab opens`, await p.isVisible('#lay2') && await p.isHidden('#lay1'));
    await p.click('#tab_res');
    await p.click('#newpat');
    check(`${w}: new patient clears`, (await p.inputValue('#age')) === '' && await p.isHidden('#results'));
    check(`${w}: no console errors`, errs.length === 0, JSON.stringify(errs));
    check(`${w}: no horizontal scroll`, (await p.evaluate(() => document.documentElement.scrollWidth)) <= w);
    await ctx.close();
  }
  await b.close();
  console.log(JSON.stringify(out, null, 1)); process.exit(fail ? 1 : 0);
})();
