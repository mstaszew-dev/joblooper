import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { compactConversationState, createConversationStateStore } from '../index.mjs';

test('compactConversationState trims older history into a summary', async () => {
  const compacted = compactConversationState({
    id: 'conv-test',
    messages: [
      { role: 'user', content: 'First turn' },
      { role: 'assistant', content: 'First response' },
      { role: 'user', content: 'Second turn' },
      { role: 'assistant', content: 'Second response' },
      { role: 'user', content: 'Third turn' },
      { role: 'assistant', content: 'Third response' },
      { role: 'user', content: 'Fourth turn' },
      { role: 'assistant', content: 'Fourth response' },
    ],
    status: 'in_progress',
    createdAt: 1,
    updatedAt: 2,
  }, { maxMessages: 4, maxChars: 200 });

  assert.ok(compacted.messages.length <= 4);
  assert.ok(compacted.messages.some((item) => String(item?.content || '').includes('summary')));
});

test('createConversationStateStore persists and reloads state', async () => {
  const store = createConversationStateStore('/tmp/joblooper-state-test.json');
  try {
    fs.unlinkSync('/tmp/joblooper-state-test.json');
  } catch {}
  const initial = await store.load();
  assert.equal(initial, null);

  await store.save({
    id: 'conv-test',
    messages: [{ role: 'user', content: 'hello' }],
    status: 'in_progress',
    createdAt: 1,
    updatedAt: 2,
    workflow: { step: 'idle' },
  });

  const reloaded = await store.load();
  assert.equal(reloaded?.messages[0]?.content, 'hello');
  assert.equal(reloaded?.workflow?.step, 'idle');
});
