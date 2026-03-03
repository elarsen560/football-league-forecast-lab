# AGENT NOTE GENERATION TEMPLATE (STRICT)

This file defines the required structure and constraints for all notes created in the `Agent Notes/` directory.

The agent must follow these rules exactly.

## VAULT LOCATION (REFERENCE)

Primary Obsidian Vault path:

/Users/sondehealth/Documents/Obsidian Vault

All scanning for linking must occur within this directory.

Only the following subdirectories may be scanned for links:
- APIs/
- Concepts/
- Projects/

The agent may write ONLY inside:
Agent Notes/

---

## FILE NAME CONVENTION (REQUIRED)

YYYY-MM-DD HHMM AGENT <current-directory-name>

- Use 24-hour time, military format, no colon (not permitted in note names)
- `<current-directory-name>` must be the final directory name only (not full path).
- Example:
  2026-03-05 1842 AGENT CI Adv Bionics Modeling

The note must be created ONLY inside:
Agent Notes/

The agent must NEVER edit existing notes.

## WRITE PERMISSIONS

The agent does not have automatic write access.

File creation permission will be granted manually via inline escalation in the Codex session.

The agent must:
- Prepare the full note content first
- Create exactly one new file in Agent Notes/ using the naming convention
- Never modify or overwrite existing files
- Never write outside Agent Notes/

---

# OUTPUT STRUCTURE (MANDATORY)

# Agent Log – <YYYY-MM-DD HH:MM>
**Source:** <Session context | Git history reconstruction (log/diff) | Hybrid (session + git)>

## Project
<current-directory-name>

## Time Scope
<Explicit natural-language scope provided in prompt>
Examples:
- “Work completed today”
- “Work completed yesterday”
- “Work completed between 2026-03-05 and 2026-03-15”

If Source includes git reconstruction, constrain git queries to this explicit Time Scope (`--since/--until` equivalent).
Use local timezone interpretation unless explicitly specified otherwise in the prompt.

## Summary
Concise, high-signal description of meaningful work.
Focus on:
- Decisions made
- Structural changes
- Modeling adjustments
- Refactors
- Infrastructure changes
- Deployment changes

Do NOT include:
- Minor cosmetic edits
- Trivial formatting changes
- Redundant descriptions

Each bullet must be one line. Maximum 8 bullets.

## Decisions
Only include if actual decisions were made.
Max 5 bullets.
Capture:
- What was decided
- Why (short rationale)

No filler.

## Open Questions / Risks
Optional.
Only include if genuinely unresolved issues exist.
Max 5 bullets.

## Related
List relevant vault notes from:
- APIs/
- Concepts/
- Projects/

Max 8 links total.
Only include links that are directly relevant to today’s work.

Use exact filename match only:
[[Exact Note Title]]

Do NOT:
- Invent note names
- Create new links
- Use fuzzy matches
- Link Daily notes
- Link Agent Notes

If no relevant notes exist, omit this section entirely.

---

# STRICT LINKING RULES

1. Scan only within:
/Users/sondehealth/Documents/Obsidian Vault/

Restricted to these subdirectories:
- APIs/
- Concepts/
- Projects/

2. Only link if exact filename match exists.
3. Maximum 8 links per note.
4. Link only on first occurrence or in Related section.
5. Prefer linking in the "Related" section rather than inline.
6. Do not link excessively.

If unsure about relevance → do not link.

---

# LENGTH CONSTRAINTS

- Target length: 150–300 words total.
- Absolute maximum: 400 words.
- Bullet-driven.
- No narrative paragraphs longer than 3 lines.
- No motivational language.
- No generic summaries.

This is an engineering log, not a diary.

---

# TIME SCOPE RULES

The agent must only summarize work within the time scope explicitly provided in the prompt.

If:
- No work occurred in that time window
- The agent lacks context for that time window
- Context was lost due to compaction
- No git logs/ diffs are available in that time window

Then produce:

## Summary
No recorded activity in the requested time window.

And optionally:

## Context Limitation
Briefly state that no relevant activity is available in current session memory.

Do not fabricate content.

---

# CONTEXT RECOVERY RULES (MANDATORY)

Context reconstruction must follow this priority:

1. Hybrid (session + git) whenever a git repository is available.
   - Always consult git history within the explicit Time Scope (`git log` + `git diff` evidence),
     even if session context is available.
   - Use session context to supplement uncommitted work or reasoning not visible in git.

2. Git history reconstruction only (if session context is unavailable).

3. Session context only (if git repository is unavailable).

4. If evidence is insufficient: use the existing "No recorded activity" fallback.

Source labeling is required:
- `Session context`
- `Git history reconstruction (log/diff)`
- `Hybrid (session + git)`

When using git reconstruction, restrict evidence to commits that touch files under `<current-directory-name>` (path-scoped), and ignore pure formatting/lockfile churn unless it was the main work.

Optional (recommended) when git-only reconstruction is used:

## Confidence Note
Derived from commits/diffs in scope; uncommitted local edits may be missing.

---

# PURPOSE

These notes exist to:
- Preserve decision rationale
- Record structural and modeling changes
- Strengthen Obsidian graph via disciplined linking
- Create a searchable development ledger

They are not:
- Meeting notes
- Journals
- Chat transcripts
- Brain dumps

Be concise. Be precise. Be useful.
