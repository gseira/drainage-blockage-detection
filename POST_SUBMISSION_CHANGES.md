# Note on commits after the submission deadline

This repository has a small number of commits dated after the dissertation
submission deadline (see dates below). This note is here so that anyone
reviewing the repository's history — a marker, a supervisor, or an
examiner — can see immediately what those commits are and why they don't
affect the submitted dissertation.

## What changed, and why

| Commit | Date | What it does |
|---|---|---|
| `f49d97b` | 2026-09-08 | Fixes leftover comments and an error-message string that still referred to "Groq" after the code had already been migrated to Cerebras for its LLM explanations *before* submission. Text-only; the LLM provider actually used at runtime did not change. |
| `969c2e6` | 2026-09-08 | Cerebras (the third-party LLM API this project calls) withdrew public access to the `gemma-4-31b` model shortly after submission, moving it behind a paid dedicated-endpoint tier. This broke the live demo's explanation feature outright. The fix points the same code at `qwen-3.8-27b`, a comparable model still on Cerebras's public tier, so the deployed demo keeps working. |
| `33e4c09` | 2026-09-08 | The live demo's monitoring database was filling the hosting server's disk — an infrastructure/ops issue with the deployment, not with the project itself. This tightens how long full-resolution images are kept in that database (while still keeping the historical trend statistics), so the server doesn't run out of disk space. |

## Why none of this affects the submitted work

All three commits are deployment/operations maintenance for keeping the
**live demo** running after a third-party API change and a disk-space
issue on the hosting server. None of them touch:

- the model architecture, weights, or training code,
- the evaluation methodology or reported results,
- the dataset,
- or the written dissertation text.

They exist solely so the publicly-hosted live demo (see README.md) keeps
working for anyone who wants to try it after submission — not because
anything in the submitted dissertation itself needed to change.

Happy to walk through the actual diffs for any of these on request — each
commit message explains the change in full, and the diffs are small
(a handful of comment/string edits, one config-default swap, and one
retention-policy change).
