/**
 * The steps of RAG, in pipeline order. "How it works" teaches one per section,
 * each with an interactive widget on the user's own documents, and `docs/`
 * has one file per step with the same name.
 *
 * `example` is a real case from these three PDFs, measured in the hand-built
 * project (FINDINGS.md, LEARNING_LOG.md), so every step starts from something
 * that actually went wrong or right here.
 *
 * A step with `explain` has its interactive page at /learn/<slug>: what the
 * step is, why it matters, and what to try (the three-sentence rule).
 */
export interface Stage {
  slug: string;
  title: string;
  what: string;
  example: string;
  arrives: string;
  /** The step's page: three sentences above its widget. Steps without one are not interactive yet. */
  explain?: [string, string, string];
}

export const STAGES: Stage[] = [
  {
    slug: "parse",
    title: "Extract and parse",
    what: "Read the text, tables and layout out of every PDF page.",
    example: "One GPO page hides its footer under a full-page fill. You can't see it, but a plain extractor reads it anyway.",
    arrives: "M3",
    explain: [
      "A PDF stores glyphs at positions, not paragraphs: a parser has to rebuild the text, its reading order and its tables.",
      "What it gets wrong reaches every later step: a flattened table, a hidden footer, a page number that is not the one printed.",
      "Pick a page below to compare PyMuPDF's raw text with Docling's, and see what the page hides.",
    ],
  },
  {
    slug: "clean",
    title: "Clean",
    what: "Repair what extraction broke, in a fixed order, keeping the raw text beside the clean text.",
    example: "The GEP report's font drops the 'h' in 'Th': 'The' reads 'Te', and 15 such words appear 325 times.",
    arrives: "M3",
    explain: [
      "Cleaning repairs what extraction broke, one rule at a time and always in the same order, keeping the raw text beside the clean one.",
      "A broken word cannot match a question: 'Te economy' is never found by 'the economy', and a header on every page boosts the wrong passages.",
      "Pick a document and a rule to see every page it changed, and exactly what it changed.",
    ],
  },
  {
    slug: "chunk",
    title: "Chunk",
    what: "Cut each document into passages small enough to search and to quote.",
    example: "The fixed-size chunker started passages mid-word: 'n the upside...'.",
    arrives: "M3",
    explain: [
      "Chunking cuts each page into passages: the pieces that are searched, ranked and shown to the model.",
      "Where the cuts fall decides what a passage can answer: cut a paragraph in half and neither half states the fact, cut across sections and the passage mixes two topics.",
      "Try each chunker on the same page, change the size and overlap, and watch the boundaries move.",
    ],
  },
  {
    slug: "context",
    title: "Contextual retrieval",
    what: "Have a model write one sentence placing each passage in its document, before it is indexed.",
    example: "The executive summary's risks paragraph never says 'executive summary', so it ranked 45th.",
    arrives: "M5",
    explain: [
      "Contextual retrieval has a model read the part of the document a passage sits in and write one or two sentences placing it there, which are embedded with the passage.",
      "A passage alone often leaves out whose figures, which year or which table it is about, and the question asks exactly that; the line puts it back for search, while the answer model still reads only the passage.",
      "Pick a test question or a passage below to see the line, the exact text embedded with and without it, and where the answer ranks on each index.",
    ],
  },
  {
    slug: "embed",
    title: "Embed",
    what: "Turn each passage, and each question, into a vector: a list of numbers that captures meaning.",
    example: "nomic-embed-text expects 'search_document: ' before passages and 'search_query: ' before questions.",
    arrives: "M4",
    explain: [
      "An embedding model reads a whole passage and returns one vector: a point in a space of hundreds of dimensions where passages that mean similar things sit close together.",
      "Each model builds its own space, with its own prefixes in front of questions and passages, so a vector only means something next to vectors made by the same model the same way.",
      "Below, every passage of an index is a point; ask a question to see where it lands, what survives cutting the vectors shorter, and what happens when two models mix.",
    ],
  },
  {
    slug: "store",
    title: "Vector store",
    what: "Keep the vectors and a word index in LanceDB, so the closest passages are found quickly.",
    example: "At about 1,200 passages, comparing a question with every vector is fast enough that no shortcut is needed.",
    arrives: "M3",
    explain: [
      "LanceDB keeps each index as a table on disk: one row per passage, with its text, its page, its section and its vector, plus a word index for BM25.",
      "An approximate (ANN) index answers faster on millions of rows by not comparing every vector, and pays for it in recall.",
      "Look inside an index below, and measure what an approximate index would lose on your own passages.",
    ],
  },
  {
    slug: "search",
    title: "Search lab",
    what: "Find passages by meaning (dense), by words (BM25), or both (hybrid), then rerank the best ones.",
    example: "'Global growth forecast for 2026' found U.S. and euro-area growth first: right topic, wrong subject.",
    arrives: "M3",
    explain: [
      "Dense search finds passages by meaning, BM25 finds them by the words they share with the question, and hybrid search merges both lists by rank (RRF).",
      "They fail differently: dense search matches the topic but can miss the subject, BM25 matches the exact words but not their meaning.",
      "Ask a question below and follow each passage from list to list.",
    ],
  },
  {
    slug: "rewrite",
    title: "Query rewriting and the agentic loop",
    what: "Rephrase or split the question, or let a model decide to search again until the evidence is enough.",
    example: "'What replaced FDsys, and when?' is two questions in one sentence.",
    arrives: "M5",
    explain: [
      "A question is not always worded the way a report states its answer, and some questions are two in one. Query rewriting has Remote AI ask the question in other words before searching: split into parts, several phrasings, or a drafted answer to search with.",
      "Multi-step search goes further: after each search the model looks at what was found and decides to answer or to search again, with a new query and, if it wants, inside one document. A follow-up question in a conversation is made whole the same way before it is searched.",
      "Pick a test question below to replay what each technique searched for in recorded runs, and where the answer page ranked. With Remote AI connected you can run one live.",
    ],
  },
  {
    slug: "prompt",
    title: "Prompt",
    what: "Number the passages and put them in front of the model, with strict rules about using only them.",
    example: "A whitespace-only change to the prompt flipped a small model between refusing and inventing.",
    arrives: "M6",
    explain: [
      "The prompt is everything the answer model reads: a fixed set of rules, then the numbered passages and the question. It sees nothing else.",
      "Its exact text decides answers: one blank line moved two test questions here, and a prompt longer than the model's window loses its start without any error.",
      "Pick a recorded answer below to see its messages exactly as sent, the context budget they used, and the same passages under prompts v2, v3 and v4.",
    ],
  },
  {
    slug: "generate",
    title: "Generate and cite",
    what: "The model writes the answer and cites [n]; code, not the model, turns each [n] into a page.",
    example: "Told to flag section mismatches it couldn't see, qwen3:1.7b said a Chapter 1 fact came from the executive summary.",
    arrives: "M6",
    explain: [
      "The answer model writes from the passages and points at each one it used with its number, [n]; code, not the model, turns each number into a document and page.",
      "A page number a model types can be invented; a number can only point at a passage it was shown, and one with no passage behind it is reported, never fixed. When nothing states the answer, only the exact refusal counts.",
      "Pick a recorded answer below to follow each [n] to its page, edit the answer and watch the same code read it, and compare the answer models on the same passages.",
    ],
  },
  {
    slug: "figures",
    title: "Figures",
    what: "Read each chart once with a vision model and add its description to the index as a figure passage.",
    example: "The iMac chart shows each part of the 445 kg baseline only as a bar's height. No number is printed.",
    arrives: "M7",
    explain: [
      "A chart's labels are already in the text, but what it shows is not: each bar's size exists only as a drawing, so a vision model describes every chart once and the index adds that description as a figure passage.",
      "Without it, the right page can be found and the question still cannot be answered; with it, part of the answer is a model's reading of a picture, so it is marked as one and checked against the drawing.",
      "Pick a figure below to see what the text already had, what the model wrote, and each value it read beside the value measured from the PDF.",
    ],
  },
  {
    slug: "evaluate",
    title: "Evaluate",
    what: "Score every setup on questions with known answers, so no upgrade is assumed to help.",
    example: "A keyword check passed four answers that contained invented claims.",
    arrives: "M6",
    explain: [
      "An evaluation asks every test question with one setup and scores each answer Pass, Refused or Wrong, beside whether an answer page was found at all.",
      "That is how no upgrade is assumed to help; but 25 questions are few, so a difference of two is a hint to measure more, not proof.",
      "Pick a recorded run below to take its scorecard apart, and another to see which questions changed and whether that could be chance.",
    ],
  },
  {
    slug: "safety",
    title: "Safety",
    what: "Treat PDF text as data, never instructions: drop hidden text and flag passages that give orders.",
    example: "Text a reader can't see can still reach the model, unless the parser drops it.",
    arrives: "M6",
    explain: [
      "Text from a PDF is data, never instructions: the guard drops text a reader cannot see, flags text that tries to give the model orders, and prompt v3 fences passages off as quoted data.",
      "Everything in a passage reaches the model, including a footer hidden under a box or a sentence written to steer it.",
      "Pick the crafted test page or a real page below to see what the guard compared, what it dropped, and how a fenced prompt holds up.",
    ],
  },
];
