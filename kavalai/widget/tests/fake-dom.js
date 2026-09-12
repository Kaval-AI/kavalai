/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The smallest DOM the widget's mount() needs, so its behaviour can be tested
under plain `node --test` without a browser or an npm dependency. Selectors
are class selectors only (".a" or ".a.b"); layout is not simulated, which is
what preview.html and the screenshots are for.
*/

class FakeClassList {
  constructor(node) {
    this.node = node;
  }

  names() {
    return (this.node.className || "").split(/\s+/).filter(Boolean);
  }

  add(...names) {
    const current = this.names();
    for (const name of names) {
      if (!current.includes(name)) current.push(name);
    }
    this.node.className = current.join(" ");
  }

  remove(...names) {
    this.node.className = this.names()
      .filter((name) => !names.includes(name))
      .join(" ");
  }

  contains(name) {
    return this.names().includes(name);
  }

  toggle(name, force) {
    const wanted = force === undefined ? !this.contains(name) : Boolean(force);
    if (wanted) this.add(name);
    else this.remove(name);
    return wanted;
  }
}

/* Custom properties go through setProperty and are what iteration yields,
   as `Array.from(element.style)` does in a browser; plain properties such
   as `style.width` are ordinary fields. */
class FakeStyle {
  constructor() {
    this.custom = new Map();
  }

  setProperty(name, value) {
    this.custom.set(name, value);
  }

  removeProperty(name) {
    this.custom.delete(name);
  }

  getPropertyValue(name) {
    return this.custom.get(name) || "";
  }

  [Symbol.iterator]() {
    return this.custom.keys();
  }
}

class FakeElement {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase();
    this.ownerDocument = doc;
    this.children = [];
    this.parentNode = null;
    this.className = "";
    this.classList = new FakeClassList(this);
    this.style = new FakeStyle();
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.ownText = "";
  }

  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this.replaceChildren();
    this.ownText = String(value);
  }

  appendChild(child) {
    return this.insertBefore(child, null);
  }

  insertBefore(child, reference) {
    child.remove();
    child.parentNode = this;
    const at = reference ? this.children.indexOf(reference) : -1;
    if (at === -1) this.children.push(child);
    else this.children.splice(at, 0, child);
    return child;
  }

  remove() {
    if (this.parentNode) {
      const siblings = this.parentNode.children;
      siblings.splice(siblings.indexOf(this), 1);
      this.parentNode = null;
    }
  }

  replaceChildren(...nodes) {
    for (const child of this.children) child.parentNode = null;
    this.children = [];
    this.ownText = "";
    for (const node of nodes) this.appendChild(node);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }

  removeEventListener(type, listener) {
    const registered = this.listeners.get(type) || [];
    const at = registered.indexOf(listener);
    if (at !== -1) registered.splice(at, 1);
  }

  dispatch(type, extra) {
    const event = Object.assign({ type: type, target: this, preventDefault() {} }, extra);
    for (const listener of [...(this.listeners.get(type) || [])]) listener(event);
  }

  click() {
    if (!this.disabled) this.dispatch("click");
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  setPointerCapture() {}

  releasePointerCapture() {}

  matches(selector) {
    return selector
      .split(".")
      .filter(Boolean)
      .every((name) => this.classList.contains(name));
  }

  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (child.matches(selector)) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

/* True when neither the node nor an ancestor carries `hidden`. */
function isShown(node) {
  for (let current = node; current; current = current.parentNode) {
    if (current.hidden) return false;
  }
  return true;
}

function installDom() {
  const doc = { activeElement: null };
  doc.createElement = (tag) => new FakeElement(tag, doc);
  doc.createElementNS = (_ns, tag) => new FakeElement(tag, doc);
  doc.body = new FakeElement("body", doc);
  doc.querySelector = (selector) => doc.body.querySelector(selector);

  const win = new FakeElement("window", doc);
  win.innerWidth = 1280;
  win.innerHeight = 800;

  globalThis.document = doc;
  globalThis.window = win;
  globalThis.getComputedStyle = (node) => ({
    getPropertyValue: (name) => node.style.getPropertyValue(name),
  });
  return { document: doc, window: win };
}

module.exports = { installDom, isShown };
