// QA harness, injected into an app page (same origin) by evaluate_script.
// window.__qa.runPage(url, slug, widths, theme) loads the page in a sized iframe
// for each width, waits for it to settle, and returns overflow, axe, console and
// failed-request results. The caption at the top says what is being checked.
(() => {
  const H = { 375: 812, 768: 1024, 1024: 768, 1280: 800, 1440: 900 };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function overlay() {
    let o = document.getElementById("__qa");
    if (o) return o;
    o = document.createElement("div");
    o.id = "__qa";
    o.style.cssText = "position:fixed;inset:0;z-index:2147483647;background:#2b2b30;display:flex;flex-direction:column;font:13px/1.45 system-ui,sans-serif;color:#f5f5f5";
    o.innerHTML = '<div id="__qa_cap" style="padding:8px 12px;background:#111;white-space:pre-wrap;min-height:40px"></div>' +
      '<div id="__qa_stage" style="flex:1;display:flex;justify-content:center;align-items:flex-start;overflow:hidden;padding:10px"></div>';
    document.body.appendChild(o);
    return o;
  }
  const caption = (t) => { overlay(); document.getElementById("__qa_cap").textContent = t; };

  function measure() {
    const vw = document.documentElement.clientWidth;
    const describe = (el) => {
      const cls = typeof el.className === "string" ? el.className.split(/\s+/).slice(0, 4).join(".") : "";
      const txt = (el.innerText || el.getAttribute("aria-label") || "").trim().replace(/\s+/g, " ").slice(0, 50);
      return `${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}${cls ? "." + cls : ""} "${txt}"`;
    };
    const clipped = (el) => {
      for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
        if (getComputedStyle(a).overflowX !== "visible") return true;
      }
      return false;
    };
    const offenders = [];
    const scrollers = [];
    for (const el of document.body.querySelectorAll("*")) {
      if (el.closest("nextjs-portal")) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const st = getComputedStyle(el);
      if (st.visibility === "hidden" || st.display === "none") continue;
      if ((r.right > vw + 1 || r.left < -1) && st.position !== "fixed" && !clipped(el)) {
        offenders.push({ el: describe(el), left: Math.round(r.left), right: Math.round(r.right) });
      }
      if ((st.overflowX === "auto" || st.overflowX === "scroll") && el.scrollWidth > el.clientWidth + 4) {
        scrollers.push({ el: describe(el), scroll: el.scrollWidth, client: el.clientWidth });
      }
    }
    const failed = performance.getEntriesByType("resource")
      .filter((e) => e.responseStatus >= 400)
      .map((e) => `${e.responseStatus} ${e.name.replace(location.origin, "")}`);
    return {
      vw,
      docScroll: document.documentElement.scrollWidth,
      pageOverflow: document.documentElement.scrollWidth > vw + 1,
      h1: [...document.querySelectorAll("h1")].map((h) => h.innerText.trim().slice(0, 60)),
      offenders: offenders.slice(0, 5),
      offenderCount: offenders.length,
      scrollers: scrollers.slice(0, 5),
      height: document.documentElement.scrollHeight,
      themeApplied: document.documentElement.classList.contains("dark") ? "dark" : "light",
      failed,
    };
  }

  async function settle(w, maxMs = 20000) {
    const t0 = performance.now();
    let last = -1;
    let quietSince = performance.now();
    while (performance.now() - t0 < maxMs) {
      await sleep(250);
      let n = 0;
      let busy = 0;
      try {
        n = w.performance.getEntriesByType("resource").length;
        busy = w.document.querySelectorAll('[data-slot="skeleton"],[aria-busy="true"]').length;
      } catch { /* navigating */ }
      if (n !== last || busy > 0) { quietSince = performance.now(); last = n; }
      if (performance.now() - quietSince > 1500) return Math.round(performance.now() - t0);
    }
    return -1;
  }

  async function load(url, width, theme) {
    localStorage.setItem("theme", theme);
    overlay();
    const stage = document.getElementById("__qa_stage");
    stage.innerHTML = "";
    const height = H[width] ?? 900;
    const scale = Math.min(1, (stage.clientWidth - 20) / width, (stage.clientHeight - 20) / height);
    const wrap = document.createElement("div");
    wrap.style.cssText = `width:${Math.floor(width * scale)}px;height:${Math.floor(height * scale)}px;overflow:hidden;outline:2px solid #e879f9;flex:none`;
    const f = document.createElement("iframe");
    f.title = "QA frame";
    f.style.cssText = `width:${width}px;height:${height}px;border:0;transform:scale(${scale});transform-origin:0 0;background:#fff`;
    wrap.appendChild(f);
    stage.appendChild(wrap);
    const logs = [];
    const hook = setInterval(() => {
      try {
        const w = f.contentWindow;
        if (!w || w.__qaHooked || w.location.href === "about:blank") return;
        w.__qaHooked = true;
        for (const k of ["error", "warn"]) {
          const orig = w.console[k].bind(w.console);
          w.console[k] = (...a) => {
            logs.push(`${k}: ` + a.map((x) => {
              try { return typeof x === "string" ? x : (x && x.message) || JSON.stringify(x); } catch { return String(x); }
            }).join(" ").slice(0, 300));
            orig(...a);
          };
        }
        w.addEventListener("error", (e) => logs.push(`pageerror: ${e.message || (e.target && (e.target.src || e.target.href)) || e.type}`), true);
        w.addEventListener("unhandledrejection", (e) => logs.push(`rejection: ${String(e.reason).slice(0, 300)}`));
      } catch { /* not yet */ }
    }, 4);
    const t0 = performance.now();
    await new Promise((res) => { f.onload = res; f.src = url; setTimeout(res, 60000); });
    clearInterval(hook);
    const settled = await settle(f.contentWindow);
    return { f, logs, settled, ms: Math.round(performance.now() - t0), scale };
  }

  async function check(f) {
    const w = f.contentWindow;
    const m = w.eval("(" + measure.toString() + ")()");
    if (!w.axe) w.eval(window.__qaAxe);
    const r = await w.axe.run(w.document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"] },
      resultTypes: ["violations"],
    });
    m.axe = r.violations.map((v) => ({
      id: v.id, impact: v.impact, n: v.nodes.length,
      nodes: v.nodes.slice(0, 3).map((n) => ({
        target: n.target.join(" ").slice(0, 120), html: n.html.slice(0, 140),
        why: (n.failureSummary || "").split("\n").slice(1, 2).join(" ").slice(0, 160),
      })),
    }));
    return m;
  }

  const flags = (r) => [
    r.pageOverflow ? `OVERFLOW ${r.docScroll}/${r.vw}` : "",
    r.offenderCount ? `offenders ${r.offenderCount}` : "",
    r.h1.length !== 1 ? `h1=${r.h1.length}` : "",
    r.axe.length ? `axe ${r.axe.map((v) => `${v.id}(${v.n})`).join(",")}` : "",
    r.logs.length ? `console ${r.logs.length}` : "",
    r.failed.length ? `failed ${r.failed.length}` : "",
    r.settled < 0 ? "UNSETTLED" : "",
    r.themeApplied !== r.theme ? `THEME ${r.themeApplied}` : "",
  ].filter(Boolean).join(" · ") || "ok";

  async function runPage(url, slug, widths, theme) {
    const out = [];
    for (const width of widths) {
      caption(`QA matrix · ${slug} (${url}) · ${width} px · ${theme} · loading…`);
      const { f, logs, settled, ms } = await load(url, width, theme);
      caption(`QA matrix · ${slug} · ${width} px · ${theme} · checking (overflow, axe, console)…`);
      let r;
      try { r = await check(f); } catch (e) { r = { error: String(e).slice(0, 200), axe: [], h1: [], failed: [] }; }
      Object.assign(r, { slug, url, width, theme, ms, settled, logs });
      r.flags = flags(r);
      out.push(r);
      caption(`QA matrix · ${slug} · ${width} px · ${theme} · ${r.flags}`);
      await sleep(400);
    }
    window.__qaResults = (window.__qaResults || []).concat(out);
    return out;
  }

  async function show(url, width, theme, scrollY = 0) {
    caption(`QA · ${url} · ${width} px · ${theme}`);
    const { f } = await load(url, width, theme);
    if (scrollY) f.contentWindow.scrollTo(0, scrollY);
    await sleep(300);
    return { height: f.contentDocument.documentElement.scrollHeight };
  }

  async function run(url, slug, theme, widths = [375, 768, 1024, 1280, 1440]) {
    const r = await runPage(url, slug, widths, theme);
    return r.map((x) => `${x.theme} ${x.slug} ${x.width}: ${x.pageOverflow ? "OVERFLOW " : ""}` +
      (x.axe.filter((v) => v.id !== "scrollable-region-focusable").map((v) => `${v.id}(${v.n}) ${v.nodes[0].html.slice(0, 70)} ${v.nodes[0].why.slice(2, 70)}`).join(" | ") || "clean") +
      (x.logs.length ? " LOGS " + x.logs.join(";") : "") + (x.failed.length ? " FAILED " + x.failed.join(";") : "") + ` (${x.ms} ms)`);
  }

  // Focus every tabbable element inside the frame and compare its look focused
  // vs blurred. No pointer has touched the frame, so script focus matches
  // :focus-visible (checked: `focusVisible` must equal the focused count).
  async function focusAudit() {
    const SEL = 'a[href], button:not([disabled]), input:not([disabled]):not([type=hidden]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"]), summary';
    const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
    const desc = (el) => {
      const name = (el.getAttribute("aria-label") || el.innerText || el.value || el.title || "").trim().replace(/\s+/g, " ").slice(0, 40);
      return `${el.tagName.toLowerCase()}${el.getAttribute("role") ? "[" + el.getAttribute("role") + "]" : ""} "${name}"`;
    };
    const look = (el) => {
      const s = getComputedStyle(el);
      return [s.outlineStyle, s.outlineWidth, s.outlineColor, s.boxShadow, s.backgroundColor, s.borderColor, s.textDecorationLine, s.color].join("|");
    };
    const els = [...document.querySelectorAll(SEL)].filter((el) => {
      const r = el.getBoundingClientRect();
      const s = getComputedStyle(el);
      return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && !el.closest("[inert],[aria-hidden='true'],nextjs-portal");
    });
    const noFocus = [], notFocusable = [], unnamed = [];
    let fv = 0, focused = 0;
    for (const el of els) {
      if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
      await raf();
      const before = look(el);
      el.focus({ preventScroll: true });
      await raf();
      if (document.activeElement !== el) { notFocusable.push(desc(el)); continue; }
      focused++;
      if (el.matches(":focus-visible")) fv++;
      const wrap = el.closest("[data-slot=input-group]");
      const changed = look(el) !== before || (wrap && getComputedStyle(wrap).boxShadow !== "none");
      if (!changed) noFocus.push(desc(el));
      const name = (el.getAttribute("aria-label") || el.getAttribute("aria-labelledby") || el.innerText || el.title || el.getAttribute("placeholder") || "").trim();
      if (!name && !(el.labels && el.labels.length)) unnamed.push(el.outerHTML.slice(0, 120));
      el.blur();
    }
    return { tabbable: els.length, focused, focusVisible: fv, noVisibleFocus: noFocus.slice(0, 10), noVisibleFocusCount: noFocus.length, notFocusable: notFocusable.slice(0, 5), unnamed: unnamed.slice(0, 5) };
  }

  async function focusRun(pages, theme = "dark", width = 1280) {
    const out = {};
    for (const [slug, url] of pages) {
      caption(`Keyboard · ${slug} · ${width} px · ${theme} · focusing every control…`);
      const { f } = await load(url, width, theme);
      try { out[slug] = await f.contentWindow.eval("(" + focusAudit.toString() + ")()"); }
      catch (e) { out[slug] = { error: String(e).slice(0, 200) }; }
      const o = out[slug];
      caption(`Keyboard · ${slug} · ${o.focused}/${o.tabbable} focused · ${o.noVisibleFocusCount} without a visible focus · ${o.unnamed?.length ?? 0} unnamed`);
    }
    return out;
  }

  window.__qa = { runPage, run, show, caption, measure, focusRun };
  window.__qaResults = window.__qaResults || [];
})();
