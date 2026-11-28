# Path or direct: a measurement to run yourself

[العربية](README.ar.md)

Tawjih claims that a reviewed path gives better answers than asking the model
the same question. This folder measures that claim on the example project
(quadratic equations), with your own models through Ollama. It needs nothing
but Python and Ollama.

```
ollama pull qwen2.5:3b
python experiments/path-vs-direct/run.py --models qwen2.5:3b
```

A fuller run, with several models (one at a time, in turn):

```
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
python experiments/path-vs-direct/run.py --models qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b \
    --cases 60 --repeat 3
```

It prints how many model calls it will make. A small model on a GPU answers
in a few seconds; on a CPU it can take an hour or more. `--cases 20
--repeat 1` gives a quick first look.

## What is compared

Each equation is asked three ways, `--repeat` times each:

| way | what the model is given |
|---|---|
| `path` | the example's path: the model turns the equation into a, b and c; Python solves it and checks the roots twice; the model explains the computed result |
| `direct` | the question, and one line saying how to write the answer (`x = value`), so both ways are scored by the same test |
| `steps` | the same, asked to work step by step first |

The cases are the example's 20 equations, then generated ones (seed 7):
- larger roots and leading coefficients;
- negative leading terms;
- double roots;
- terms on both sides.

They are half Arabic, half English. Every generated equation is checked in
the tests to have the roots it is scored on.

## What you get

`results/summary.md` has a table for each model:
- the share of right answers, with a 95% interval;
- the cases right every time;
- seconds and tokens per case;
- the difference between the path and each direct way, with its interval;
- the cases only one of them got right, and McNemar's exact test;
- the same, by group (the example or generated, Arabic or English).

`results/results.json` has everything else:
- the settings;
- the computer, the Tawjih version, and each model's digest;
- every case and every answer.

The logs and records of every run are in `results/work/`.

Please send `summary.md` and `results.json` back, or open an issue with
them. Whatever the numbers say, they go into the README and the paper.

## What this does and does not show

- **The path computes.** Its arithmetic is done by Python, so the
  comparison measures "a model plus reviewed code steps" against "a model
  alone". Its result is about decomposition into steps, some of which
  are code. It does not measure the knowledge kept in files (the human
  and reading sides): that needs another example, whose answers depend on
  what the files know rather than on arithmetic.
- **The test is a pattern.** A case is right when the answer says
  `x = r` for every root. An answer that is right but written otherwise
  (`x ∈ {2, 3}`) counts as wrong. The `direct` and `steps` prompts ask for
  the pattern to make this fair. Read some failures in `results.json`
  before trusting a small difference.
- **Small samples.** With 60 cases, a difference of a few points is
  within chance: look at the intervals and the p-value, not only the
  percentages.
- **Equations are easy.** Strong models may solve them all either way. The
  question matters most for small local models, where the gap is expected
  to be largest.
