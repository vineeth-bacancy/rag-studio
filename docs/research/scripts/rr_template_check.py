"""Compare Qwen3-Reranker ST chat_template rendering vs. the model-card manual format.
Usage: python3 -I rr_template_check.py <tokenizer.json> <chat_template.jinja> <embedding_tokenizer.json>
"""
import sys

import jinja2
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
tpl_src = open(sys.argv[2]).read()
emb_tok = Tokenizer.from_file(sys.argv[3])

print('yes ids:', tok.encode('yes', add_special_tokens=False).ids,
      'no ids:', tok.encode('no', add_special_tokens=False).ids)

task = 'Given a web search query, retrieve relevant passages that answer the query'
q = 'What is the capital of China?'
d = 'The capital of China is Beijing.'

prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
manual = prefix + f'<Instruct>: {task}\n<Query>: {q}\n<Document>: {d}' + suffix

env = jinja2.Environment()  # default: keep_trailing_newline=False, like HF
rendered = env.from_string(tpl_src).render(messages=[
    {'role': 'system', 'content': task},
    {'role': 'query', 'content': q},
    {'role': 'document', 'content': d},
])
print('rendered == manual (string):', rendered == manual)
if rendered != manual:
    print('rendered tail repr:', repr(rendered[-60:]))
    print('manual   tail repr:', repr(manual[-60:]))
ids_r = tok.encode(rendered, add_special_tokens=False).ids
ids_m = tok.encode(manual, add_special_tokens=False).ids
print('token ids equal:', ids_r == ids_m, 'len', len(ids_m))
print('last 6 manual ids:', ids_m[-6:])
# Reranker tokenizer with add_special_tokens=True: does it append anything?
print('reranker tok appends EOS?:', tok.encode('hi', add_special_tokens=True).ids)
print('embedding tok appends EOS?:', emb_tok.encode('hi', add_special_tokens=True).ids)
# token count of the fixed template overhead (prefix + instruct + labels + suffix)
overhead = tok.encode(prefix + f'<Instruct>: {task}\n<Query>: \n<Document>: ' + suffix,
                      add_special_tokens=False).ids
print('template overhead tokens (default instruction, empty q/d):', len(overhead))
