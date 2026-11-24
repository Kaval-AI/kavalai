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

/**
 * Reading a model call's recorded request and response as text.
 *
 * The clients record the request as the JSON text of the provider call's
 * arguments, so its shape is the provider's: Chat Completions and Ollama send
 * `messages`, the OpenAI Responses API `input` (and `instructions`),
 * Anthropic `messages` with a top-level `system`. Message content is a string
 * or a list of parts. The response is recorded as the model's text, which is
 * often JSON itself. Gemini's `contents` are recorded as Python reprs and an
 * embedding request holds no messages; those have no text view.
 */

/** The four roles a message is drawn with; anything else is `other`. */
export type ChatRole = 'system' | 'user' | 'assistant' | 'tool' | 'other';

export interface ChatLine {
  /** The role as recorded, shown as the message's label. */
  label: string;
  role: ChatRole;
  /** The tool or call a message names, when it names one. */
  detail: string | null;
  text: string;
}

/** A recorded payload as a value: JSON text is parsed, anything else returned as it is. */
export function parsePayload(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return value;
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function roleOf(label: string): ChatRole {
  switch (label) {
    case 'system':
    case 'developer':
      return 'system';
    case 'user':
      return 'user';
    case 'assistant':
    case 'model':
      return 'assistant';
    case 'tool':
    case 'function':
      return 'tool';
    default:
      return 'other';
  }
}

/** The readable text of message content: a string, or a list of parts joined. */
export function contentText(content: unknown): string {
  if (content === null || content === undefined) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map(partText).filter((text) => text !== '').join('\n\n');
  }
  return JSON.stringify(content, null, 2);
}

function partText(part: unknown): string {
  if (typeof part === 'string') return part;
  if (!isRecord(part)) return String(part);
  if (typeof part['text'] === 'string') return part['text'];
  switch (part['type']) {
    case 'tool_use':
      return `${part['name']}(${JSON.stringify(part['input'] ?? {})})`;
    case 'tool_result':
      return contentText(part['content']);
    case 'image':
    case 'image_url':
    case 'input_image':
      return '[image]';
  }
  return JSON.stringify(part);
}

function line(label: string, text: string, detail: string | null = null): ChatLine {
  return { label, role: roleOf(label), detail, text };
}

function messageLine(message: unknown): ChatLine {
  if (!isRecord(message)) return line('message', String(message));
  if (message['type'] === 'function_call') {
    return line('assistant', `${message['name']}(${message['arguments'] ?? ''})`, message['name'] ?? null);
  }
  if (message['type'] === 'function_call_output') {
    return line('tool', contentText(message['output']), message['call_id'] ?? null);
  }
  const label = typeof message['role'] === 'string' ? message['role'] : 'message';
  const parts = [contentText(message['content'])];
  if (Array.isArray(message['tool_calls'])) {
    for (const call of message['tool_calls']) {
      const fn = isRecord(call) && isRecord(call['function']) ? call['function'] : call;
      parts.push(`→ ${fn?.['name']}(${fn?.['arguments'] ?? ''})`);
    }
  }
  const detail = message['name'] ?? message['tool_call_id'] ?? null;
  return line(label, parts.filter((text) => text !== '').join('\n\n'), detail);
}

/**
 * Whether a Responses API `input` holds messages. An embedding request's
 * `input` is a list of texts (or one text), which is not a conversation.
 */
function isMessageList(input: unknown): input is unknown[] {
  return Array.isArray(input) && input.some((item) => isRecord(item) && ('role' in item || 'type' in item));
}

/**
 * The request's messages in order, a separate system prompt first, or `null`
 * when the request holds no messages to show.
 */
export function requestMessages(request: unknown): ChatLine[] | null {
  const data = parsePayload(request);
  if (!isRecord(data)) return null;
  const lines: ChatLine[] = [];
  if (data['system'] !== undefined && data['system'] !== null) {
    lines.push(line('system', contentText(data['system'])));
  }
  if (typeof data['instructions'] === 'string') {
    lines.push(line('system', data['instructions']));
  }
  const messages = Array.isArray(data['messages'])
    ? data['messages']
    : isMessageList(data['input'])
      ? data['input']
      : typeof data['input'] === 'string' && lines.length > 0
        ? [{ role: 'user', content: data['input'] }]
        : [];
  lines.push(...messages.map(messageLine));
  return lines.length > 0 ? lines : null;
}

/**
 * The response as text: the model's answer as recorded, indented when it is
 * JSON, or `null` when the response is a structure with no text in it.
 */
export function responseText(response: unknown): string | null {
  if (typeof response === 'string') {
    const parsed = parsePayload(response);
    return parsed === response ? response : JSON.stringify(parsed, null, 2);
  }
  if (isRecord(response)) {
    if (typeof response['content'] === 'string') return response['content'];
    if (typeof response['text'] === 'string') return response['text'];
  }
  return null;
}
