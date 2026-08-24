# Agent Instructions (Repo-Wide)

## Project Context
- This repo builds lecture materials from the contents of `docs/`.
- The target audience is undergraduate students: write explanations to be easy and intuitive.
- Save authored lecture materials in `lecture/`.
- Current priority: continue creating lecture materials starting from Chapter 06 (`docs/chapter06.md` -> `lecture/chapter06.md`) and onward.

## Default Workflow
- Use `docs/` as the primary source of truth for content.
- Convert/reshape content into teaching-friendly notes (clear structure, simple language, concrete examples when helpful).
- Write outputs to `lecture/` (do not overwrite unrelated files unless asked).

## Next Steps (2026-03-05)
- Goal: Rewrite `lecture/chapter06.md` ~ `lecture/chapter12.md` into undergraduate-friendly, easy-to-skim lecture notes (bullet/outline style), not dense source-style prose.
- Keep: Chapter section numbering/structure aligned with `docs/` (e.g., 6.1~6.9, 7.1~7.6).
- Style targets (match `lecture/chapter02.md` ~ `lecture/chapter05.md`):
  - Start each chapter with: learning goals, 1-line takeaway, preview.
  - Prefer short paragraphs + bullet lists + mini checklists.
  - Add simple analogies and "how to read results" blocks.
  - Link to practice code files in `practice/chapterXX/code/*` where relevant.
  - Avoid over-padding: add content only when it reduces confusion or improves usability.
  - Avoid nested bullets (keep a single bullet level).
- Current status snapshot:
  - Rewritten: `lecture/chapter06.md`, `lecture/chapter07.md`, `lecture/chapter08.md`, `lecture/chapter09.md` (09 is still too short; needs only essential additions).
  - Still source-like: `lecture/chapter10.md`, `lecture/chapter11.md`, `lecture/chapter12.md` (need full rewrite into the target style).
- Suggested execution order:
  - 1) Finish `lecture/chapter09.md` (add only essentials: baselines, multi-step forecast, lookback/horizon, evaluation plots, drift monitoring).
  - 2) Fully rewrite `lecture/chapter10.md` (pipeline/checklists, task-by-task: classification/sentiment/similarity/summarization/topic modeling).
  - 3) Fully rewrite `lecture/chapter11.md` (network basics, centrality interpretation, community detection, diffusion, practical pitfalls).
  - 4) Fully rewrite `lecture/chapter12.md` (complexity intuition, system dynamics, ABM, RL, validation checklists).
  - 5) Final pass: ensure consistent tone/format and remove nested bullets across 6~12.
