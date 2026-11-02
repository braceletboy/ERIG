// Run with Node after editing a data-tex attribute in docs/index.html.
// Pre-rendering preserves typeset math offline and without browser JavaScript.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const katex = require('./assets/vendor/katex/katex.min.js');
const file = path.join(__dirname, 'index.html');
const decode = value => value.replace(/&(quot|#x27|lt|gt|amp);/g,
  (_, entity) => ({quot: '"', '#x27': "'", lt: '<', gt: '>', amp: '&'})[entity]);
let count = 0;
const html = fs.readFileSync(file, 'utf8').replace(
  /<(div|span) class="([^"]*)" data-tex="([^"]*)"><!--math-->[\s\S]*?<!--\/math--><\/\1>/g,
  (_, tag, classes, source) => {
    const rendered = katex.renderToString(decode(source), {
      displayMode: classes.split(' ').includes('equation'),
      output: 'htmlAndMathml',
      throwOnError: true,
      strict: 'error',
      trust: false,
      macros: {'\\bm': '\\boldsymbol{#1}'},
    });
    count++;
    return `<${tag} class="${classes}" data-tex="${source}"><!--math-->${rendered}<!--/math--></${tag}>`;
  });
if (!count) throw new Error('No marked LaTeX expressions found.');
fs.writeFileSync(file, html);
console.log(`Rendered ${count} LaTeX expressions with KaTeX ${katex.version}.`);
