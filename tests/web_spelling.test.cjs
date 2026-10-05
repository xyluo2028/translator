const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "../translator_app/web/index.html"), "utf8");
const source = html.slice(html.indexOf("function spellingParts("), html.indexOf("async function copyText("));

// Minimal DOM stand-in for testing the rendered text and suggestion button actions.
function el(tag, props = {}, ...children) {
  return {
    tag, ...props, children,
    append(...items) { this.children.push(...items); },
    insertBefore(child, before) { this.children.splice(this.children.indexOf(before), 0, child); },
  };
}
const calls = [];
const context = vm.createContext({ el, run: (...args) => calls.push(args) });
vm.runInContext(source, context);
function text(node) {
  return typeof node === "string" ? node : (node.textContent || "") + node.children.map(text).join("");
}

function fixture(prefix = "") {
  const offset = Array.from(prefix).length;
  return {
    original: prefix + "The echelon helo helo",
    corrected: prefix + "The echelon help help",
    corrections: [
      { word: "helo", suggestion: "help", alternatives: ["hell"], start: offset + 12, end: offset + 16 },
      { word: "helo", suggestion: "help", alternatives: ["held"], start: offset + 17, end: offset + 21 },
    ],
  };
}

for (const prefix of ["", "😀 "]) {
  test(`render and replace only the selected token (prefix ${JSON.stringify(prefix)})`, () => {
    const fix = fixture(prefix);
    const rendered = context.renderSpelling(fix);
    assert.equal(text(rendered.children[0]), `Did you mean: ${fix.corrected}?`);
    assert.equal(context.replaceWord(fix, fix.corrections[1], "held"), prefix + "The echelon help held");
    const links = rendered.children[1];
    const alternatives = links.children.filter((node) => node.tag === "button").slice(1);
    alternatives[0].onclick();
    assert.equal(calls.at(-1)[1].text, prefix + "The echelon hell help");
    assert.equal(calls.at(-1)[1].spellcheck, false);
  });
}

test("legacy history without offsets displays saved correction without unsafe alternatives", () => {
  const fix = fixture();
  for (const correction of fix.corrections) {
    delete correction.start;
    delete correction.end;
  }
  const rendered = context.renderSpelling(fix);
  assert.equal(text(rendered.children[0]), `Did you mean: ${fix.corrected}?`);
  const buttons = rendered.children[1].children.filter((node) => node.tag === "button");
  assert.equal(buttons.length, 1);
  buttons[0].onclick();
  assert.equal(calls.at(-1)[1].text, fix.original);
});
