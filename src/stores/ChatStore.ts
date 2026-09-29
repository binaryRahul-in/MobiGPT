import AsyncStorage from '@react-native-async-storage/async-storage';
import {makeAutoObservable, runInAction} from 'mobx';
import {makePersistable} from 'mobx-persist-store';

import {ChatMessage, getLlmEngine} from '../services/llm';
import {uid} from '../utils/format';
import type {ModelStore} from './ModelStore';
import type {SettingsStore} from './SettingsStore';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  createdAt: number;
  stats?: {genTps: number; promptTps: number; tokens: number; interrupted: boolean};
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  modelName: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

/** Approximate chars-per-token used to keep history inside the context window. */
const CHARS_PER_TOKEN = 3.2;

export class ChatStore {
  conversations: Conversation[] = [];
  activeId: string | null = null;
  generating = false;
  lastError: string | null = null;

  constructor(private settings: SettingsStore, private models: ModelStore, persist = true) {
    makeAutoObservable(this, {}, {autoBind: true});
    if (persist) {
      makePersistable(this, {name: 'mobigpt.chats.v1', properties: ['conversations', 'activeId'], storage: AsyncStorage}).catch(
        () => undefined,
      );
    }
  }

  get active(): Conversation | undefined {
    return this.conversations.find(c => c.id === this.activeId);
  }

  get sorted(): Conversation[] {
    return [...this.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  newConversation(): Conversation {
    const c: Conversation = {
      id: uid('c_'),
      title: 'New chat',
      modelName: this.models.loaded?.name ?? '',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    this.conversations.push(c);
    this.activeId = c.id;
    // Return the observable copy MobX stored, not the plain object we pushed.
    return this.conversations[this.conversations.length - 1];
  }

  select(id: string) {
    this.activeId = id;
  }

  remove(id: string) {
    this.conversations = this.conversations.filter(c => c.id !== id);
    if (this.activeId === id) {
      this.activeId = this.sorted[0]?.id ?? null;
    }
  }

  clearAll() {
    this.conversations = [];
    this.activeId = null;
  }

  /** Builds the prompt: system + as much recent history as fits in ~75 % of n_ctx. */
  buildMessages(conv: Conversation): ChatMessage[] {
    const budget = this.settings.llm.nCtx * CHARS_PER_TOKEN * 0.75 - this.settings.llm.maxTokens * CHARS_PER_TOKEN * 0.5;
    const history: ChatMessage[] = [];
    let used = this.settings.llm.systemPrompt.length;
    for (let i = conv.messages.length - 1; i >= 0; i--) {
      const m = conv.messages[i];
      if (!m.content || m.error) {
        continue;
      }
      if (used + m.content.length > budget && history.length > 0) {
        break;
      }
      used += m.content.length;
      history.unshift({role: m.role, content: m.content});
    }
    const sys = this.settings.llm.systemPrompt.trim();
    return sys ? [{role: 'system', content: sys}, ...history] : history;
  }

  async send(text: string): Promise<void> {
    const content = text.trim();
    if (!content || this.generating) {
      return;
    }
    if (!this.models.loadedId) {
      this.lastError = 'Load a model first (Models tab).';
      throw new Error(this.lastError);
    }
    const conv = this.active ?? this.newConversation();
    const user: Message = {id: uid('m_'), role: 'user', content, createdAt: Date.now()};
    const reply: Message = {id: uid('m_'), role: 'assistant', content: '', createdAt: Date.now()};
    conv.messages.push(user, reply);
    if (conv.title === 'New chat') {
      conv.title = content.slice(0, 48) + (content.length > 48 ? '…' : '');
    }
    conv.modelName = this.models.loaded?.name ?? conv.modelName;
    conv.updatedAt = Date.now();
    this.generating = true;
    this.lastError = null;
    const target = () => conv.messages.find(m => m.id === reply.id);
    let pending = '';
    let lastFlush = 0;
    try {
      const s = this.settings.llm;
      const res = await getLlmEngine().chat(
        this.buildMessages({...conv, messages: conv.messages.filter(m => m.id !== reply.id)}),
        {
          temperature: s.temperature,
          topP: s.topP,
          topK: s.topK,
          minP: s.minP,
          maxTokens: s.maxTokens,
          repeatPenalty: s.repeatPenalty,
          enableThinking: s.enableThinking,
        },
        t => {
          pending += t.token;
          const now = Date.now();
          // Batch UI updates (~30 fps) — per-token re-renders are wasteful.
          if (now - lastFlush > 33) {
            lastFlush = now;
            const snapshot = t.content ?? pending;
            const reasoning = t.reasoning;
            runInAction(() => {
              const m = target();
              if (m) {
                m.content = snapshot;
                if (reasoning) {
                  m.reasoning = reasoning;
                }
              }
            });
          }
        },
      );
      runInAction(() => {
        const m = target();
        if (m) {
          m.content = res.text.trim();
          m.reasoning = res.reasoning || m.reasoning;
          m.stats = {genTps: res.genTps, promptTps: res.promptTps, tokens: res.tokensPredicted, interrupted: res.interrupted};
        }
      });
    } catch (e: any) {
      runInAction(() => {
        const m = target();
        if (m) {
          m.error = e?.message ?? String(e);
          m.content = m.content || '';
        }
        this.lastError = e?.message ?? String(e);
      });
    } finally {
      runInAction(() => {
        this.generating = false;
        conv.updatedAt = Date.now();
      });
    }
  }

  async stop() {
    await getLlmEngine().stop();
  }
}
