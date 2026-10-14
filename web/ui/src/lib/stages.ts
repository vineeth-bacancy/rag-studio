/**
 * The steps of RAG, in pipeline order. "How it works" teaches one per section,
 * each with an interactive widget on the user's own documents, and `docs/`
 * has one file per step with the same name.
 *
 * `example` is a real case from these three PDFs, measured in the hand-built
 * project (FINDINGS.md, LEARNING_LOG.md), so every step starts from something
 * that actually went wrong or right here.
 */
export interface Stage {
  slug: string;
  title: string;
  what: string;
  example: string;
  arrives: string;
}

export const STAGES: Stage[] = [
  {
    slug: "parse",
    title: "Extract and parse",
    what: "Read the text, tables and layout out of every PDF page.",
    example: "One GPO page hides its footer under a full-page fill. You can't see it, but a plain extractor reads it anyway.",
    arrives: "M3",
  },
  {
    slug: "clean",
    title: "Clean",
    what: "Repair what extraction broke, in a fixed order, keeping the raw text beside the clean text.",
    example: "The GEP report's font drops the 'h' in 'Th': 'The' reads 'Te', and 15 such words appear 325 times.",
    arrives: "M3",
  },
  {
    slug: "chunk",
    title: "Chunk",
    what: "Cut each document into passages small enough to search and to quote.",
    example: "The fixed-size chunker started passages mid-word: 'n the upside...'.",
    arrives: "M3",
  },
  {
    slug: "context",
    title: "Contextual retrieval",
    what: "Have a model write one sentence placing each passage in its document, before it is indexed.",
    example: "The executive summary's risks paragraph never says 'executive summary', so it ranked 45th.",
    arrives: "M5",
  },
  {
    slug: "embed",
    title: "Embed",
    what: "Turn each passage, and each question, into a vector: a list of numbers that captures meaning.",
    example: "nomic-embed-text expects 'search_document: ' before passages and 'search_query: ' before questions.",
    arrives: "M4",
  },
  {
    slug: "store",
    title: "Vector store",
    what: "Keep the vectors and a word index in LanceDB, so the closest passages are found quickly.",
    example: "At about 1,200 passages, comparing a question with every vector is fast enough that no shortcut is needed.",
    arrives: "M3",
  },
  {
    slug: "search",
    title: "Search lab",
    what: "Find passages by meaning (dense), by words (BM25), or both (hybrid), then rerank the best ones.",
    example: "'Global growth forecast for 2026' found U.S. and euro-area growth first: right topic, wrong subject.",
    arrives: "M3",
  },
  {
    slug: "rewrite",
    title: "Query rewriting and the agentic loop",
    what: "Rephrase or split the question, or let a model decide to search again until the evidence is enough.",
    example: "'What replaced FDsys, and when?' is two questions in one sentence.",
    arrives: "M5",
  },
  {
    slug: "prompt",
    title: "Prompt",
    what: "Number the passages and put them in front of the model, with strict rules about using only them.",
    example: "A whitespace-only change to the prompt flipped a small model between refusing and inventing.",
    arrives: "M6",
  },
  {
    slug: "generate",
    title: "Generate and cite",
    what: "The model writes the answer and cites [n]; code, not the model, turns each [n] into a page.",
    example: "Told to flag section mismatches it couldn't see, qwen3:1.7b said a Chapter 1 fact came from the executive summary.",
    arrives: "M6",
  },
  {
    slug: "figures",
    title: "Figures",
    what: "Read charts as text with a vision model, or match them as images.",
    example: "The iMac chart shows each part of the 445 kg baseline only as a bar's height. No number is printed.",
    arrives: "M7",
  },
  {
    slug: "evaluate",
    title: "Evaluate",
    what: "Score every setup on questions with known answers, so no upgrade is assumed to help.",
    example: "A keyword check passed four answers that contained invented claims.",
    arrives: "M6",
  },
  {
    slug: "safety",
    title: "Safety",
    what: "Treat PDF text as data, never instructions: drop hidden text and flag passages that give orders.",
    example: "Text a reader can't see can still reach the model, unless the parser drops it.",
    arrives: "M7",
  },
];
