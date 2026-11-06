// D210 UI phase 3 check (Playwright, demo cases only): norm scales come only from the server; the ruler
// agrees with a hand check (two R peaks of the API -> the RR in ms); the ruler works from the keyboard;
// the graph has the API's nodes, lights a chain and has a text list.
//   node tools/ui/check_p3.js http://127.0.0.1:8765 <clinic-key>
const { chromium } = require('playwright');
const [base, key] = [process.argv[2] || 'http://127.0.0.1:8765', process.argv[3] || 'demo-key'];
(async () => {
  const b = await chromium.launch(); const out = {}; let fail = 0;
  const check = (name, ok, info = '') => { out[name] = ok ? 'ok' : 'FAIL ' + info; if (!ok) fail++; };
  const ctx = await b.newContext({ viewport: { width: 1280, height: 900 } });
  await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
  const p = await ctx.newPage(); const errs = []; p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(base);
  await p.evaluate(() => { document.querySelector('#svc').open = true; });
  await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
  for (const cs of ['normal', 'hyperk']) {
    await p.selectOption('#case', cs); await p.fill('#age', '54'); await p.click('#sex_m');
    const respP = p.waitForResponse(r => r.url().endsWith('/api/analyze')); await p.click('#run');
    const api = await (await respP).json(); await p.waitForSelector('#results:not(.hidden)'); await p.waitForTimeout(700);
    const n = api.norms || {};
    const scales = await p.$$eval('#keys .scale', e => e.length);
    const want = ['heart_rate_bpm', 'pr_ms', 'qrs_ms', 'qtc_ms', 'qrs_axis_deg']
      .filter((k, i) => n[k] && n[k][0] != null && n[k][1] != null &&
        [api.measurements.heart_rate_bpm, api.measurements.intervals.pr_ms, api.measurements.intervals.qrs_ms,
         api.measurements.intervals.qtc_bazett_ms, api.measurements.axes.qrs_deg][i] != null).length;
    check(`${cs}: scales only where the server gave a range`, scales === want, `${scales} vs ${want}`);
    await p.click('#tab_lab'); await p.waitForSelector('#ecgv:not(.hidden)', { timeout: 15000 });
    await p.waitForTimeout(450);   // the curtain (350 ms) has opened
    // The hand check: the RR between the 2nd and 3rd R peaks of the answer.
    const fs = 500, rp = api.measurements.r_peaks_samples;
    const [ta, tb] = [rp[1] / fs, rp[2] / fs], rr = Math.round((tb - ta) * 1000);
    await p.$eval('#ecgsvg', e => e.scrollIntoView({ block: 'center' }));
    const box = await p.$eval('#ecgsvg', e => { const r = e.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, vb: e.viewBox.baseVal.width }; });
    const X = t => box.x + t * 25 * (box.vb / 125) * box.w / box.vb;
    await p.mouse.click(X(ta), box.y + 30); await p.mouse.click(X(tb), box.y + 30);
    const read = await p.textContent('#ecgread'), got = Number((read.match(/Линейка: (\d+)\s*мс/) || [])[1]);
    check(`${cs}: ruler == hand RR (${rr} ms, +-8)`, Math.abs(got - rr) <= 8, read);
    await p.click('#ecgclear'); await p.focus('#ecgsvg');
    for (let i = 0; i < 25; i++) await p.keyboard.press('ArrowRight');
    await p.keyboard.press('Enter');
    for (let i = 0; i < 5; i++) await p.keyboard.press('Shift+ArrowRight');
    await p.keyboard.press('Enter');
    const kread = await p.textContent('#ecgread');
    check(`${cs}: keyboard ruler 200 ms`, /\b200 мс/.test(kread), kread);
    const nodes = (api.causal_graph || {}).nodes || [];
    const drawn = await p.$$eval('#cgsvg .node', e => e.length);
    check(`${cs}: graph nodes == API`, drawn === nodes.length, `${drawn} vs ${nodes.length}`);
    if (nodes.length) {
      await p.focus('#cgsvg .node'); await p.keyboard.press('Enter');
      check(`${cs}: Enter lights a chain`, await p.$eval('#cgv', e => e.classList.contains('lit')));
      await p.click('#cglistbtn');
      check(`${cs}: graph as a list`, (await p.$$eval('#cglist li', e => e.length)) === Math.max(1, ((api.causal_graph || {}).edges || []).length));
      await p.click('#cglistbtn');
    }
    await p.click('#tab_res'); await p.click('#newpat');
    await p.evaluate(() => { document.querySelector('#svc').open = true; });
  }
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  await b.close(); console.log(JSON.stringify(out, null, 1)); process.exit(fail ? 1 : 0);
})();
