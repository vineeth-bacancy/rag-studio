"use client";

import { create } from "zustand";

import { ApiError, api, type AskDone, type AskError, type AskReady, type TraceStep } from "@/lib/api";

/**
 * The questions asked on this page, kept while you move between pages.
 *
 * Memory between questions (follow-ups rewritten into standalone questions)
 * arrives in M5; until then each question is answered on its own, and the
 * list is only a record of what you asked.
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
}

interface AskState {
  exchanges: Exchange[];
  busy: boolean;
  /** Which index to search and which prompt version to answer with. Each
   * answer records what it actually used (ready.index, ready.prompt). */
  profile: string;
  prompt: string;
  setProfile: (profile: string) => void;
  setPrompt: (prompt: string) => void;
  send: (question: string) => Promise<void>;
  stop: () => void;
  clear: () => void;
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
    profile: "baseline",
    prompt: "v2",
    setProfile: (profile) => set({ profile }),
    setPrompt: (prompt) => set({ prompt }),

    async send(question) {
      const text = question.trim();
      if (!text || get().busy) return;
      inFlight?.abort();
      const controller = new AbortController();
      inFlight = controller;
      const id = nextId++;
      set({
        busy: true,
        exchanges: [...get().exchanges, { id, question: text, status: "streaming", stages: [], text: "" }],
      });
      try {
        await api.askStream(
          { question: text, profile: get().profile, prompt: get().prompt },
          (event) => {
            switch (event.event) {
              case "stage":
                patch(id, { stages: [...(current(id)?.stages ?? []), event.data] });
                break;
              case "ready":
                patch(id, { ready: event.data });
                break;
              case "delta":
                patch(id, { text: (current(id)?.text ?? "") + event.data.text });
                break;
              case "done":
                patch(id, { done: event.data, text: event.data.answer, status: "done" });
                break;
              case "error":
                patch(id, { error: event.data, status: "error" });
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
      set({ exchanges: [] });
    },
  };
});
