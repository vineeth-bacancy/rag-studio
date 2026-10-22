"use client";

import { mutate } from "swr";
import { create } from "zustand";

import {
  ApiError,
  api,
  type AskDone,
  type AskError,
  type AskReady,
  type ChatFull,
  type SearchMethod,
  type TraceStep,
} from "@/lib/api";

/**
 * The conversation on this page, kept while you move between pages.
 *
 * A conversation is saved on this computer (data/chat/). Memory acts only
 * through a rewrite: a follow-up ("And for 2027?") is first made a full
 * question by Remote AI, and the answer model never sees the conversation.
 * "Ask as typed" sends a follow-up without that rewrite.
 */
export interface Exchange {
  id: number;
  question: string;
  status: "streaming" | "done" | "error" | "stopped";
  /** Steps that finished, in order. Never a prediction of the next one. */
  stages: TraceStep[];
  ready?: AskReady;
  /** The model's text as it arrives. Replaced by `done.answer` at the end. */
  text: string;
  done?: AskDone;
  error?: AskError;
  /** Loaded from a saved conversation: the answer and where it came from, not the passages. */
  saved?: { standalone: string | null; rewritten: boolean | null; citations: { n: number; chunk_id: string; doc_id: string; page: number }[]; refused: boolean; ms: number | null };
  /** Sent without the follow-up rewrite, on purpose. */
  asTyped?: boolean;
}

interface AskState {
  exchanges: Exchange[];
  busy: boolean;
  /** The saved conversation these exchanges belong to; null until the first answer. */
  chatId: string | null;
  /** Which index to search, how, and which prompt version to answer with.
   * Each answer records what it actually used (ready.index, ready.search,
   * ready.prompt). */
  profile: string | null;
  prompt: string | null;
  search: SearchMethod;
  /** "Limit per section": at most this many passages per section ahead of the rest; null is off. */
  sectionCap: number | null;
  /** A reranker id, or null for no reranking. */
  rerank: string | null;
  setRerank: (rerank: string | null) => void;
  /** decompose, multi or hyde; null is no rewriting. Needs Remote AI. */
  rewrite: string | null;
  setRewrite: (rewrite: string | null) => void;
  /** Multi-step search: its round limit, or null for off. Needs Remote AI. */
  agentRounds: number | null;
  setAgentRounds: (rounds: number | null) => void;
  setProfile: (profile: string) => void;
  setPrompt: (prompt: string) => void;
  setSearch: (search: SearchMethod) => void;
  setSectionCap: (cap: number | null) => void;
  send: (question: string, opts?: { asTyped?: boolean }) => Promise<void>;
  stop: () => void;
  /** Start a new conversation (the old one stays saved). */
  clear: () => void;
  openChat: (chat: ChatFull) => void;
}

let inFlight: AbortController | null = null;
let nextId = 1;

export const useAsk = create<AskState>((set, get) => {
  const patch = (id: number, p: Partial<Exchange>) =>
    set({ exchanges: get().exchanges.map((e) => (e.id === id ? { ...e, ...p } : e)) });
  const current = (id: number) => get().exchanges.find((e) => e.id === id);

  return {
    exchanges: [],
    busy: false,
    chatId: null,
    // null means "the app's default", which the API names (indexes
    // `default_profile`, prompts `default`); a request then omits the field.
    profile: null,
    prompt: null,
    search: "dense",
    sectionCap: null,
    rerank: null,
    setRerank: (rerank) => set({ rerank }),
    rewrite: null,
    setRewrite: (rewrite) => set({ rewrite }),
    agentRounds: null,
    setAgentRounds: (agentRounds) => set({ agentRounds }),
    setProfile: (profile) => set({ profile }),
    setPrompt: (prompt) => set({ prompt }),
    setSearch: (search) => set({ search }),
    setSectionCap: (sectionCap) => set({ sectionCap }),

    async send(question, opts) {
      const text = question.trim();
      if (!text || get().busy) return;
      inFlight?.abort();
      const controller = new AbortController();
      inFlight = controller;
      const id = nextId++;
      set({
        busy: true,
        exchanges: [...get().exchanges, { id, question: text, status: "streaming", stages: [], text: "", asTyped: opts?.asTyped }],
      });
      try {
        await api.askStream(
          {
            question: text,
            profile: get().profile ?? undefined,
            prompt: get().prompt ?? undefined,
            search: get().search,
            section_cap: get().sectionCap,
            rerank: get().rerank ?? undefined,
            rewrite: get().rewrite ?? undefined,
            agent_rounds: get().agentRounds ?? undefined,
            chat_id: get().chatId ?? undefined,
            as_typed: opts?.asTyped || undefined,
          },
          (event) => {
            switch (event.event) {
              case "stage":
                patch(id, { stages: [...(current(id)?.stages ?? []), event.data] });
                break;
              case "ready":
                patch(id, { ready: event.data });
                if (event.data.chat_id) set({ chatId: event.data.chat_id });
                break;
              case "delta":
                patch(id, { text: (current(id)?.text ?? "") + event.data.text });
                break;
              case "done":
                patch(id, { done: event.data, text: event.data.answer, status: "done" });
                // The answer is now saved in its conversation: refresh the list.
                void mutate("/chats");
                break;
              case "error":
                patch(id, { error: event.data, status: "error" });
                if (event.data.chat_id && get().exchanges.some((e) => e.id !== id && e.status === "done")) set({ chatId: event.data.chat_id });
                break;
            }
          },
          controller.signal,
        );
        // A stream that closes without `done` or `error` was cut off.
        if (current(id)?.status === "streaming") {
          patch(id, {
            status: "error",
            error: { stage: "network", error: "The answer stopped arriving. Ask again." },
          });
        }
      } catch (err) {
        if (controller.signal.aborted) {
          patch(id, { status: "stopped" });
        } else {
          // An ApiError means the app service answered (with a refusal such as
          // a too-long question); anything else means nothing answered.
          const message = err instanceof Error ? err.message : "The request failed.";
          patch(id, { status: "error", error: { stage: err instanceof ApiError ? "request" : "network", error: message } });
        }
      } finally {
        if (inFlight === controller) inFlight = null;
        set({ busy: false });
      }
    },

    stop() {
      inFlight?.abort();
    },

    clear() {
      inFlight?.abort();
      set({ exchanges: [], chatId: null });
    },

    openChat(chat) {
      inFlight?.abort();
      set({
        chatId: chat.id,
        exchanges: chat.turns.map((t) => ({
          id: nextId++,
          question: t.question,
          status: "done" as const,
          stages: [],
          text: t.answer,
          saved: { standalone: t.standalone_question, rewritten: t.rewritten, citations: t.citations, refused: t.refused, ms: t.total_ms },
        })),
      });
    },
  };
});
