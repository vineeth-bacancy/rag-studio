// Builds the QA harness: axe-core bundled in front of harness-src.js, so one
// Chrome DevTools `evaluate_script` (sourcePath, format "script") injects both
// into an open app page. Method and results: docs/qa/2026-10-23-m8-full-matrix.md.
//
//   npm install --prefix /tmp/rag-qa axe-core@4.14.0
//   node scripts/qa/build.cjs /tmp/rag-qa        -> /tmp/rag-qa/harness.js
//
// Then, in the DevTools Chrome on any app page (http://127.0.0.1:3100/health):
//   await window.__qa.run("/learn/figures", "learn-figures", "dark")  // 5 widths
//   await window.__qa.focusRun([["ask", "/"]])                        // focus audit
// Keep each call under DevTools' 180 s protocol limit (two or three pages).
const fs = require("fs");
const path = require("path");

const dir = path.resolve(process.argv[2] || ".");
const axe = fs.readFileSync(require.resolve("axe-core/axe.min.js", { paths: [dir] }), "utf8");
const src = fs.readFileSync(path.join(__dirname, "harness-src.js"), "utf8");
const out = path.join(dir, "harness.js");
fs.writeFileSync(out, "window.__qaAxe = " + JSON.stringify(axe) + ";\n" + src);
console.log(`${out}: ${fs.statSync(out).size} bytes`);
