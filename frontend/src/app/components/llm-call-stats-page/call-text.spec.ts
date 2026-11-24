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
*/
import { contentText, parsePayload, requestMessages, responseText } from './call-text';

describe('parsePayload', () => {
  it('parses JSON text', () => {
    expect(parsePayload('{"a": 1}')).toEqual({ a: 1 });
    expect(parsePayload(' [1, 2] ')).toEqual([1, 2]);
  });

  it('returns other values as they are', () => {
    expect(parsePayload('plain words')).toBe('plain words');
    expect(parsePayload('{not json')).toBe('{not json');
    expect(parsePayload(null)).toBeNull();
    const value = { a: 1 };
    expect(parsePayload(value)).toBe(value);
  });
});

describe('contentText', () => {
  it('reads strings, part lists and other values', () => {
    expect(contentText(null)).toBe('');
    expect(contentText(undefined)).toBe('');
    expect(contentText('Hello')).toBe('Hello');
    expect(
      contentText([
        'first',
        { type: 'text', text: 'second' },
        { type: 'tool_use', name: 'metric', input: { name: 'trips' } },
        { type: 'tool_use', name: 'ping' },
        { type: 'tool_result', content: [{ type: 'text', text: 'done' }] },
        { type: 'image_url', image_url: { url: 'x' } },
        { type: 'audio', data: 'x' },
        7,
        { type: 'text', text: '' },
      ]),
    ).toBe(
      'first\n\nsecond\n\nmetric({"name":"trips"})\n\nping({})\n\ndone\n\n[image]\n\n{"type":"audio","data":"x"}\n\n7',
    );
    expect(contentText({ a: 1 })).toBe('{\n  "a": 1\n}');
  });
});

describe('requestMessages', () => {
  it('reads Chat Completions messages from JSON text', () => {
    const request = JSON.stringify({
      model: 'glm-5.3',
      messages: [
        { role: 'system', content: 'Be brief.' },
        { role: 'user', content: 'How many trips?' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [{ id: 'c1', type: 'function', function: { name: 'metric', arguments: '{"n":2}' } }],
        },
        { role: 'tool', tool_call_id: 'c1', content: '42' },
        { role: 'critic', content: 'Fine.' },
      ],
    });
    expect(requestMessages(request)).toEqual([
      { label: 'system', role: 'system', detail: null, text: 'Be brief.' },
      { label: 'user', role: 'user', detail: null, text: 'How many trips?' },
      { label: 'assistant', role: 'assistant', detail: null, text: '→ metric({"n":2})' },
      { label: 'tool', role: 'tool', detail: 'c1', text: '42' },
      { label: 'critic', role: 'other', detail: null, text: 'Fine.' },
    ]);
  });

  it('puts a separate system prompt first, as Anthropic records it', () => {
    const lines = requestMessages({
      system: [{ type: 'text', text: 'Be brief.' }],
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }],
    });
    expect(lines!.map((l) => [l.role, l.text])).toEqual([
      ['system', 'Be brief.'],
      ['user', 'Hi'],
    ]);
  });

  it('reads the Responses API input and instructions', () => {
    const lines = requestMessages({
      instructions: 'Be brief.',
      input: [
        { role: 'developer', content: 'Rules' },
        { type: 'function_call', name: 'metric', arguments: '{}', call_id: 'c1' },
        { type: 'function_call', name: 'ping', call_id: 'c2' },
        { type: 'function_call_output', call_id: 'c1', output: '42' },
        { content: 'no role' },
        'loose text',
        { role: 'model', content: 'Done' },
        { role: 'function', name: 'metric', content: '1' },
      ],
    });
    expect(lines!.map((l) => [l.label, l.role, l.detail, l.text])).toEqual([
      ['system', 'system', null, 'Be brief.'],
      ['developer', 'system', null, 'Rules'],
      ['assistant', 'assistant', 'metric', 'metric({})'],
      ['assistant', 'assistant', 'ping', 'ping()'],
      ['tool', 'tool', 'c1', '42'],
      ['message', 'other', null, 'no role'],
      ['message', 'other', null, 'loose text'],
      ['model', 'assistant', null, 'Done'],
      ['function', 'tool', 'metric', '1'],
    ]);
  });

  it('reads a plain-string input under instructions as one user message', () => {
    expect(requestMessages({ instructions: 'Be brief.', input: 'Hello' })).toEqual([
      { label: 'system', role: 'system', detail: null, text: 'Be brief.' },
      { label: 'user', role: 'user', detail: null, text: 'Hello' },
    ]);
  });

  it("does not take an embedding request's texts for messages", () => {
    expect(requestMessages({ model: 'embed', input: ['first text', 'second text'] })).toBeNull();
    expect(requestMessages({ model: 'embed', input: 'one text' })).toBeNull();
  });

  it('reads tool calls without a function wrapper', () => {
    const lines = requestMessages({
      messages: [{ role: 'assistant', content: 'Checking', tool_calls: [{ name: 'metric' }] }],
    });
    expect(lines![0].text).toBe('Checking\n\n→ metric()');
  });

  it('has nothing to show for a request without messages', () => {
    expect(requestMessages(null)).toBeNull();
    expect(requestMessages('not json')).toBeNull();
    expect(requestMessages('[1, 2]')).toBeNull();
    expect(requestMessages({ model: 'gemini', contents: ['Content(...)'] })).toBeNull();
  });
});

describe('responseText', () => {
  it('returns plain text as it is', () => {
    expect(responseText('Hello there')).toBe('Hello there');
  });

  it('indents a response that is JSON text', () => {
    expect(responseText('{"cards": []}')).toBe('{\n  "cards": []\n}');
  });

  it('reads the text field of a structured response', () => {
    expect(responseText({ content: 'Hi' })).toBe('Hi');
    expect(responseText({ text: 'Hi' })).toBe('Hi');
  });

  it('has nothing to show for a structure without text', () => {
    expect(responseText({ model: 'embed', usage: { tokens: 3 } })).toBeNull();
    expect(responseText(null)).toBeNull();
    expect(responseText([1])).toBeNull();
  });
});
