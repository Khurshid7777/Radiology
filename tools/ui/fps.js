// D210 UI: frame rate during the animations (curtain + waveform drawing + count-up), CPU slowed 4x
// (a mid-range phone). node tools/ui/fps.js http://127.0.0.1:8765 <clinic-key>
const { chromium } = require('playwright');
const [base, key] = [process.argv[2] || 'http://127.0.0.1:8765', process.argv[3] || 'demo-key'];
(async () => {
  const b = await chromium.launch(); const ctx = await b.newContext({ viewport: { width: 412, height: 915 } });
  await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
  const p = await ctx.newPage(); const cdp = await ctx.newCDPSession(p);
  await p.goto(base);
  if (await p.isVisible('#next1')) await p.click('#next1');
  await p.evaluate(() => { document.querySelector('#svc').open = true; });
  await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
  await p.selectOption('#case', 'stemi'); await p.fill('#age', '54'); await p.click('#sex_m');
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  const measure = async (action, ms) => {
    await p.evaluate(() => { window.__f = []; const tick = t => { window.__f.push(t); if (window.__f.length < 2000) requestAnimationFrame(tick); }; requestAnimationFrame(tick); });
    await action(); await p.waitForTimeout(ms);
    const f = await p.evaluate(() => window.__f);
    const d = f.slice(1).map((t, i) => t - f[i]); d.sort((a, b) => a - b);
    return { fps: Math.round(1000 * d.length / (f[f.length - 1] - f[0])), p95_ms: Math.round(d[Math.floor(d.length * 0.95)]) };
  };
  const r1 = await measure(async () => { await p.click('#run'); await p.waitForSelector('#results:not(.hidden)'); }, 900);
  await p.waitForSelector('#ecgv:not(.hidden)', { timeout: 20000 }).catch(() => {});
  const r2 = await measure(() => p.click('#tab_lab'), 1500);
  console.log(JSON.stringify({ result_countup: r1, lab_curtain_and_draw: r2 }));
  await b.close();
})();
