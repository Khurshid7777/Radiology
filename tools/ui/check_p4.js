// D210 UI phase 4 check: speed/gain segments reach the request; the camera screen (fake device) opens,
// draws its guide and closes, with the parts switch; the header line is still under reduced motion.
//   node tools/ui/check_p4.js http://127.0.0.1:8765 <clinic-key>
const { chromium } = require('playwright');
const [base, key] = [process.argv[2] || 'http://127.0.0.1:8765', process.argv[3] || 'demo-key'];
(async () => {
  const b = await chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
  const out = {}; let fail = 0;
  const check = (name, ok, info = '') => { out[name] = ok ? 'ok' : 'FAIL ' + info; if (!ok) fail++; };
  const ctx = await b.newContext({ viewport: { width: 1280, height: 900 }, permissions: ['camera'] });
  await ctx.addInitScript(k => localStorage.setItem('cardiolens_key', k), key);
  const p = await ctx.newPage(); const errs = []; p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(base);
  await p.click('[data-for="speed"] [data-v="50"]'); await p.click('[data-for="gain"] [data-v="20"]');
  check('segment sets the select', (await p.inputValue('#speed')) === '50' && (await p.inputValue('#gain')) === '20');
  await p.evaluate(() => { document.querySelector('#svc').open = true; });
  await p.waitForFunction(() => document.querySelectorAll('#case option').length > 1);
  await p.selectOption('#case', 'normal'); await p.fill('#age', '54'); await p.click('#sex_m');
  const reqP = p.waitForRequest(r => r.url().endsWith('/api/analyze'));
  await p.click('#run'); const body = (await reqP).postData() || '';
  check('speed 50 and gain 20 reach the request', /paper_speed_mm_s"\r?\n\r?\n50\r?\n/.test(body) && /gain_mm_mv"\r?\n\r?\n20\r?\n/.test(body));
  await p.waitForSelector('#results:not(.hidden)');
  await p.click('#newpat');
  check('new patient returns the segments to auto', (await p.inputValue('#speed')) === '' &&
    await p.$eval('[data-for="speed"] [data-v=""]', e => e.classList.contains('on')));
  await p.click('#camopen'); await p.waitForSelector('#cam.on', { timeout: 10000 });
  await p.waitForTimeout(2500);
  check('camera screen opens with the parts switch', await p.isVisible('#campartsbox') && await p.isVisible('#camshot'));
  const painted = await p.$eval('#camdraw', c => { const g = c.getContext('2d'); if (!c.width) return false;
    const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n > 100; });
  check('camera guide drawn', painted);
  await p.click('#camclose'); check('camera closes', await p.isHidden('#cam.on'));
  check('no console errors', errs.length === 0, JSON.stringify(errs));
  const rm = await b.newContext({ reducedMotion: 'reduce' }); const q = await rm.newPage(); await q.goto(base);
  check('header line still on reduced motion', (await q.$eval('header .pulse path', e => getComputedStyle(e).animationName)) === 'none');
  await b.close(); console.log(JSON.stringify(out, null, 1)); process.exit(fail ? 1 : 0);
})();
