const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const html = fs.readFileSync(path.join(__dirname, "../translator_app/web/index.html"), "utf8");
const script = html.split("<script>")[1].split("</script>")[0].replace(/\ninit\(\);\s*$/, "");

function harness({ speech = false, voices = [{ name: "Japanese local", lang: "ja-JP", localService: true }] } = {}) {
  function node(tag = "div") {
    return {
      tag, value: "", textContent: "", children: [], hidden: false, disabled: false,
      firstChild: { textContent: "" }, dataset: {}, events: {},
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      setAttribute(key, value) { this[key] = value; },
      addEventListener(key, fn) { this.events[key] = fn; },
      get options() { return this.children; },
    };
  }
  const ids = new Map();
  const get = (id) => { if (!ids.has(id)) ids.set(id, node()); return ids.get(id); };
  const buttons = ["translate", "dictionary", "relatives", "enhance"].map((mode) => Object.assign(node("button"), { dataset: { mode } }));
  const storage = new Map();
  const requests = [];
  const spoken = [];
  const cancelled = [];
  const browser = { scrollTo() {} };
  if (speech) {
    browser.SpeechSynthesisUtterance = function(text) { this.text = text; };
    browser.speechSynthesis = {
      getVoices: () => voices, speak: (utterance) => spoken.push(utterance),
      cancel: () => cancelled.push(true), addEventListener() {},
    };
  }
  const context = vm.createContext({
    document: { getElementById: get, createElement: node, querySelectorAll: () => buttons, addEventListener() {} },
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    window: browser,
    fetch: async (url, options) => {
      requests.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ mode: "enhance", enhanced_prompt: "Improved text", clarifications: [], model: "gemini" }) };
    },
  });
  vm.runInContext(script, context);
  vm.runInContext(`state.info = {
    models: [{provider: "gemini", name: "gemini", available: true}], dictionary_models: {},
    enhancement_scenarios: [{id:"general",label:"General",description:"Polish grammar and clarity."},
      {id:"prompt",label:"Prompt",description:"Improve AI instructions."},
      {id:"email",label:"Email",description:"Make emails clear and courteous."}]
  };`, context);
  get("model").value = "gemini::gemini";
  get("model").append({ value: "gemini::gemini", disabled: false });
  get("scenario").value = "general";
  get("input").value = "Draft text";
  return { context, get, requests, storage, spoken, cancelled };
}

test("Enhance shows scenarios, hides translation controls, and clears stale translation results", () => {
  const { context, get } = harness();
  vm.runInContext('state.last = {mode:"translate", translation:"old result"}; state.lastRequest = {text:"old"};', context);
  context.setMode("enhance");
  assert.equal(get("scenarioLabel").hidden, false);
  assert.equal(get("languages").hidden, true);
  assert.equal(get("toneLabel").hidden, true);
  assert.equal(get("input")["aria-label"], "Text to enhance");
  assert.equal(get("output").children[0].textContent, "Your enhanced text appears here.");
  assert.equal(vm.runInContext("state.last", context), null);
  context.setMode("translate");
  assert.equal(get("scenarioLabel").hidden, true);
  assert.equal(get("languages").hidden, false);
});

test("scenario reaches requests, Retry, preferences, and restored history", async () => {
  const { context, get, requests, storage } = harness();
  context.setMode("enhance");
  get("scenario").value = "email";
  get("scenario").events.change();
  assert.equal(JSON.parse(storage.get("translator.prefs.v1")).scenario, "email");
  await context.run();
  await context.run("retry");
  assert.equal(requests.length, 2);
  assert.equal(requests[0].scenario, "email");
  assert.equal(requests[1].scenario, "email");
  assert.equal(requests[1].rerun, "retry");
  const saved = JSON.parse(storage.get("translator.history.v1"))[0];
  get("scenario").value = "general";
  context.restore(saved);
  assert.equal(get("scenario").value, "email");
  assert.equal(get("output").children[0].textContent, "Improved text");
});

test("old enhancement history restores Prompt for subsequent Retry", async () => {
  const { context, get, requests } = harness();
  context.restore({ req: {mode:"enhance", text:"old prompt", model:"gemini", provider:"gemini"},
    data: {mode:"enhance", enhanced_prompt:"Old result", clarifications:[]} });
  assert.equal(get("scenario").value, "prompt");
  await context.run("retry");
  assert.equal(requests[0].scenario, "prompt");
});

test("changing scenarios clears the previous result and Retry state", async () => {
  const { context, get } = harness();
  context.setMode("enhance");
  await context.run();
  get("scenario").value = "email";
  get("scenario").events.change();
  assert.equal(get("retry").disabled, true);
  assert.equal(get("copy").disabled, true);
  assert.equal(get("output").children[0].textContent, "Your enhanced text appears here.");
});

test("scenario explanation changes with the selection and hides outside Enhance", () => {
  const { context, get } = harness();
  context.setMode("enhance");
  assert.equal(get("scenarioExplanation").hidden, false);
  assert.equal(get("scenarioExplanation").textContent, "Polish grammar and clarity.");
  get("scenario").value = "email";
  get("scenario").events.change();
  assert.equal(get("scenarioExplanation").textContent, "Make emails clear and courteous.");
  context.setMode("translate");
  assert.equal(get("scenarioExplanation").hidden, true);
});

test("Relatives restricts input before network requests and keeps translation-only controls hidden", async () => {
  const { context, get, requests } = harness();
  context.setMode("relatives");
  get("input").value = "This is far too long for a vocabulary lookup";
  context.updateControls();
  assert.equal(get("go").disabled, true);
  assert.equal(get("languages").hidden, true);
  assert.equal(get("toneLabel").hidden, true);
  assert.equal(get("literal").hidden, true);
  assert.equal(get("play").hidden, true);
  await context.run();
  assert.equal(requests.length, 0);
  for (const term of ["friendly", "break the ice", "创造力", "don't"]) assert.equal(context.validRelativesInput(term), true);
  for (const term of ["123", "this is a sentence.", "hello\nworld", "a".repeat(61)]) assert.equal(context.validRelativesInput(term), false);
  get("input").value = "friendly";
  context.updateControls();
  assert.equal(get("go").disabled, false);
  await context.run();
  assert.equal(requests[0].mode, "relatives");
  assert.equal("source_lang" in requests[0], false);
  assert.equal("target_lang" in requests[0], false);
});

test("relatives renders separate tables and empty groups", () => {
  const { context, get } = harness();
  context.setMode("relatives");
  context.render({mode:"relatives",term:"create",derivations:[{term:"creative",pos:"adjective",meaning:"有创造力的"}],
    synonyms:[{term:"make",pos:"verb",meaning:"制作"}],antonyms:[],notes:"No direct opposite in this sense."});
  const tables = get("output").children.filter((child) => child.tag === "table");
  assert.equal(tables.length, 2);
  assert.equal(tables[0].children[0].textContent, "Derivations");
  assert.equal(tables[0].children[2].children[0].children[1].textContent, "adjective");
  assert.ok(get("output").children.some((child) => child.textContent === "None found for this term."));
});

test("Dictionary displays UK and US IPA while older history still renders", () => {
  const { context, get } = harness();
  context.setMode("dictionary");
  context.render({mode:"dictionary",term:"schedule",entries:[],pronunciations:[
    {label:"UK",ipa:"/ˈʃedjuːl/"},{label:"US",ipa:"/ˈskedʒuːl/"}]});
  assert.ok(get("output").children.some((child) => child.textContent === "UK IPA /ˈʃedjuːl/"));
  assert.ok(get("output").children.some((child) => child.textContent === "US IPA /ˈskedʒuːl/"));
  context.render({mode:"dictionary",term:"schedule",entries:[]});
  assert.equal(get("output").children[0].textContent, "schedule");
});

test("Play uses raw translation and the saved target language, then Stop cancels it", () => {
  const { context, get, spoken, cancelled } = harness({speech:true});
  vm.runInContext(`state.last = {mode:"translate",translation:"日本語",target_lang:"JA",furigana:{result:{translation:"日本語（にほんご）"}}};
    state.lastRequest={target_lang:"JA"};`, context);
  get("target").value = "EN";
  context.updateControls();
  assert.equal(get("play").disabled, false);
  context.playTranslation();
  assert.equal(spoken[0].text, "日本語");
  assert.equal(spoken[0].lang, "ja-JP");
  assert.equal(get("play").textContent, "■ Stop");
  context.playTranslation();
  assert.equal(cancelled.length, 1);
  assert.equal(get("play").textContent, "▶ Play");
  spoken[0].onend();
  assert.equal(spoken.length, 1, "late speech callbacks must not resume cancelled audio");
});

test("long speech chunks preserve text, complete in order, and stop when modes change", () => {
  const { context, get, spoken, cancelled } = harness({speech:true});
  const text = "日本語を読みます。".repeat(80);
  vm.runInContext(`state.last={mode:"translate",translation:${JSON.stringify(text)},target_lang:"JA"};`, context);
  assert.equal(context.speechChunks(text).join(""), text);
  context.playTranslation();
  for (let index = 0; index < spoken.length; index++) spoken[index].onend();
  assert.equal(spoken.map((utterance) => utterance.text).join(""), text);
  assert.equal(get("play").textContent, "▶ Play");
  context.playTranslation();
  context.setMode("enhance");
  assert.equal(cancelled.length, 1);
  assert.equal(get("play").hidden, true);
});

test("missing speech support or language voices is reported without playing the wrong voice", () => {
  for (const options of [{speech:false},{speech:true,voices:[{name:"English",lang:"en-US"}]}]) {
    const { context, get, spoken } = harness(options);
    vm.runInContext('state.last={mode:"translate",translation:"日本語",target_lang:"JA"};', context);
    context.updateControls();
    assert.equal(get("play").disabled, true);
    assert.equal(get("audioStatus").hidden, false);
    context.playTranslation();
    assert.equal(spoken.length, 0);
  }
});
