# CLI Natural-Language Query Plan

## Goal

Support a CLI workflow where a user asks natural-language questions about the soccer forecasting domain and receives a natural-language answer without launching the Streamlit UI.

Representative questions:

- What is Feyenoord's likelihood of finishing top 4 in the league right now?
- What is the worst performing team in the Premier League by Elo change this season?
- What was the most surprising match outcome across the tracked leagues this past weekend?

The intent is to reuse the current Python codebase, local SQLite data, and existing modeling logic rather than building a second forecasting system.

## Feasibility Assessment

This is feasible with the current repository.

The repo already contains the core ingredients needed for a CLI query interface:

- local match/state data in `soccer.db`
- deterministic Elo and probability logic in `elo.py`
- SQLite access in `db.py`
- season context, simulation, diagnostics, and standings logic in `app.py`

The main limitation is architectural, not mathematical. A large amount of reusable logic currently lives inside `app.py` in Streamlit-oriented flow. That means the codebase can answer many of these questions today, but not yet through a clean CLI/query interface.

## What AI Scaffolding Is Needed

The LLM should act as the orchestrator, not the calculator.

Broadly, the CLI stack would need:

- a thin CLI entrypoint that accepts a natural-language question
- a small Python query/runtime layer that exposes reusable football-domain functions
- an LLM orchestration layer that maps question -> query/tool call -> natural-language answer

`codex exec` is viable in this setup, but it should sit on top of deterministic Python functions rather than be responsible for raw calculation.

## Do We Need Codebase Updates First?

Yes, but not a major rewrite.

The minimum likely change is to extract a few reusable, non-Streamlit helpers from `app.py` so they can be called from a CLI script. Without that, a Codex-driven CLI would be possible but brittle because too much of the current computation path is tied to UI execution flow.

## Key Caveats

- “Right now” depends on how fresh `soccer.db` is.
- Date-window questions such as “this past weekend” need clear date interpretation rules.
- Some answers depend on simulation outputs that currently live inside Streamlit flow.
- Questions like “most surprising outcome” require a formal metric, likely based on pregame implied probability of the realized result.
- The CLI should have clear behavior for unsupported or ambiguous questions.

## Broad-Strokes Implementation Direction

1. Extract reusable soccer-query functions from `app.py`.
2. Add a simple CLI script that loads league/season context and returns structured outputs.
3. Put Codex/LLM on top for question interpretation and answer generation only.
4. Start with a small set of canonical query types.
5. Add a fallback path for unsupported questions.

## Bottom Line

This is possible with the current codebase.

What is missing is not core forecasting capability, but a callable query surface that is decoupled from Streamlit enough for a CLI tool to use reliably.

## Potential Messaging Expansion

Once a CLI/query interface exists, it can also serve as the backend for a remote messaging interface such as WhatsApp or another chat channel.

At a high level, the pattern would be:

1. A user sends a message to a messaging endpoint.
2. A webhook/service receives the message.
3. The service passes the question into the same soccer-query runtime used by the CLI.
4. The backend generates a natural-language answer.
5. The answer is sent back through the messaging provider to the user.

This is feasible, but it requires a hosted backend service rather than a local terminal-only workflow.

Broadly, that future expansion would require:

- the same reusable query layer needed for the CLI
- a small web service wrapper around that query layer
- a messaging provider integration (for example WhatsApp Business API or a provider such as Twilio)
- operational handling for message sessions, errors, rate limits, and hosted data freshness

The main architectural point is that messaging should be treated as another interface on top of the same deterministic Python query engine, not as a separate forecasting implementation.
