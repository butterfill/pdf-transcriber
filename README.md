# pdf-transcriber

An independently installable `uv` tool for transcribing academic PDFs into Markdown with
**Gemini 3.8 Flash**. Keeps the successful AI Studio prompt, JSON schema, and figure
extraction approach. The CLI is **`pdf-transcriber`**. Use `transcribe` for PDF-only conversion and `go` for both transcription and citation enrichment.

## Install

Requires Python 3.12+ and `uv`. From a local checkout of `programming-experimental`:

```sh
uv tool install ./pdf-transcriber-gemini
```

Or from this private repository using Git SSH authentication:

```sh
uv tool install 'git+ssh://git@github.com/butterfill/programming-experimental.git#subdirectory=pdf-transcriber-gemini'
```

For upgrades, add `--reinstall` to the `uv tool install` command.

Set up a **Gemini Developer API** key in the environment (not a Vertex AI key):

```sh
export GEMINI_API_KEY='your-key'
```

`GOOGLE_API_KEY` is also accepted. The CLI does not read `.env` files implicitly.

## Use

```sh
pdf-transcriber go 'Article title - author2024_keyword.pdf'
pdf-transcriber go --standard 'Article title - author2024_keyword.pdf'
```

The default is **Flex inference** (lower cost, best effort). `--standard` explicitly
requests Standard. Flex never silently falls back to Standard; busy Flex capacity may be
retried and eventually fail. The model is pinned to stable `gemini-3.8-flash` using
Google's supported `google-genai` SDK **2.x** and the new **Interactions API**, not the
legacy `google-generativeai` SDK or the older `generate_content` interface.

Outputs beside the PDF:

```text
Article title - author2024_keyword.pdf
Article title - author2024_keyword.md
author2024_keyword/
    fig_1.png
    fig_2.jpg
```

The image folder is the final ` - ` or `__` segment of the PDF basename. If there are no
substantive figures, no figure folder is created. Existing `.md` or same-named figure
folders are **never overwritten**. The previous AI Studio convention
`<basename>.gemini.json` is not created in the source directory; the temporary working
folder holds `response.json` instead. Any existing `<basename>.json` produced by
`pdf-preprocessor` is untouched.

## PDF upload and output

The tool uploads the original PDF through the **Gemini Files API**, checks whether
processing has completed, and passes the resulting file URI as a `document` input to
`client.interactions.create()`. It requests a structured JSON result using the bundled
schema, then writes Markdown and original/rasterized figures using PyMuPDF. It does **not**
use PDF-native text extraction as the transcription source. Because Google's PDF backend
may expose its embedded text layer to Gemini, the prompt's visual-only instruction is a
quality preference, not a technically enforced guarantee.

Files API uploads expire on Google's servers after 48 hours. After a successful run the
tool also attempts to delete the uploaded file immediately; after a failure it keeps
remote upload metadata for recovery. The model has an output limit (65,536 tokens), so
long PDFs may not fit into one structured JSON response; such runs fail safely without
publishing a partial Markdown document. Figure bounding boxes are model estimates: check
for clipped image edges when reviewing the generated files.

## Manual transcription with Google AI Studio

For difficult PDFs, use the public `aistudio` command to handle the prompt handoff
and local import without a Gemini API key:

```sh
pdf-transcriber aistudio 'Article.pdf'
```

In an interactive terminal, this copies the bundled transcription prompt to the
clipboard (or saves an `Article.aistudio-prompt.txt` file if the clipboard is
unavailable), **reveals the PDF in Finder on macOS using `open -R`**, and opens
Google AI Studio in your browser. Upload the PDF, paste the prompt, and run it.
Copy the returned JSON and press **Enter** in the terminal to import it; or enter
a saved JSON file path (terminal drag-and-drop paths work). If validation fails,
correct the response and retry. Clipboard and browser integration are best effort;
a file path works when the clipboard is unavailable.

If you already saved AI Studio's response, import it immediately:

```sh
pdf-transcriber aistudio 'Article.pdf' response.json
# Equivalent direct-import option:
pdf-transcriber transcribe 'Article.pdf' --from-json response.json
```

To output the prompt without starting the interactive workflow:

```sh
pdf-transcriber aistudio 'Article.pdf' --prompt-only > request.txt
```

Manual import validates the structured JSON (`markdown`, `figures`, `problems`),
extracts the listed figures from the **original PDF**, and writes the Markdown and
figure directory beside it. A fenced JSON code block from AI Studio is accepted.
It does **not** call Gemini, require credentials, or enrich citations. It leaves
the source JSON intact, refuses to overwrite Markdown or existing figure
directories, and cleans scratch files on success or failure. Use
`pdf-transcriber find-citations 'Article.md'` separately if desired.

## Conservative LaTeX escape recovery and validation

The existing JSON transcription prompt and schema are unchanged. Automatic API
conversions (including chunks and resumes), AI Studio imports and `--from-json`
all pass through the same safety checks.

Before JSON decoding, the importer recognizes known LaTeX commands within
`$...$` and `$$...$$` in the top-level `markdown` JSON string and repairs
missing *JSON* backslash escapes. This includes syntactically valid but damaging
sequences such as `\\text` and `\\beta`. It **does not rewrite** the source
JSON file or figure metadata. LaTeX recovery does not modify prose outside
math or unknown commands.

Separately, malformed JSON `\*` immediately followed by Markdown whitespace
in the top-level `markdown` string is recovered as JSON `\\*`, preserving
the escaped literal asterisk. This includes spaces and JSON-encoded newline
(`\n`), tab (`\t`), carriage return (`\r`), form feed (`\f`), and Unicode
whitespace (e.g. `\u2003`). Already-correct `\\*`, `\*` before non-whitespace,
and other JSON fields are left alone. The original JSON file is untouched.
Literal unescaped control characters (such as a raw newline within a JSON
string) remain invalid. If JSON remains invalid, the usual error path applies.

After parsing, warning-only checks identify unexpected control characters,
tabs inside math, suspicious doubled commands, escaped subscripts and
unbalanced argument braces. They do not automatically rewrite mathematical
content. Warnings include the Markdown line and column; automatic repairs
are counted and reported. Review warnings before relying on mathematics
in published transcriptions. Recovery is mechanical, not a correctness
guarantee, and neither JSON recovery nor linting verifies an equation
against the original PDF.

## Recovery

During execution, a hidden `.pdf-transcriber-<random>/` working directory is created
**next to the PDF**. On success it is deleted. On failure it is retained, including:

- `source.json`: PDF path, SHA-256 hash, model and requested tier
- `uploaded.json`: Files API upload name/URI (if uploaded)
- `interaction.json`: response metadata (if model responded)
- `response.json`: complete raw transcription JSON (if model completed)
- `partial-response.txt`: incomplete model text (if generation was interrupted)
- `error.txt`: exception traceback for debugging
- `figures/` and `transcription.md`: staged outputs (if assembly reached that point)

To retry, use the printed command:

```sh
pdf-transcriber go 'Article title - author2024_keyword.pdf' \
  --resume '/path/to/.pdf-transcriber-abcdef12'
```

If `response.json` is valid it is reused **without a second paid API call**. Otherwise,
the tool reuses the existing upload while valid, or uploads again if it has expired. You
can use `--standard` with `--resume` to retry a failed Flex request at the Standard tier;
this is never automatic. A changed PDF is rejected when resuming.

## API compatibility choices (October 2026)

- Model: [`gemini-3.8-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
  (stable / GA).
- SDK: [`google-genai` 2.x](https://pypi.org/project/google-genai/2.29.0/)
  (minimum 2.29.0, released 7 October 2026).
- [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
  for structured output and Flex `service_tier="flex"`.
- [Files API guide](https://ai.google.dev/gemini-api/docs/file-input-methods)
  for PDF upload and file URI inputs.

## Development

```sh
cd pdf-transcriber-gemini
uv sync --extra dev
uv run pytest -q
```

No network/API calls are made in the tests. The prompt and JSON schema are package
data files, which makes them straightforward to evolve independently of the CLI.

## Chunking (default for long PDFs)

`go` automatically splits PDFs **longer than 30 pages** into 10-page PDF chunks,
transcribes each independently, and combines their outputs into one Markdown file and
one figure directory **beside the original PDF**. PDFs of 30 pages or fewer are sent
as one PDF by default. Override this explicitly:

```sh
pdf-transcriber go 'Paper.pdf' --chunk       # Always use 10-page chunks
pdf-transcriber go 'Paper.pdf' --no-chunk    # Never split; may exceed model output limits
```

When merging chunks, the tool translates **local chunk page numbers to original PDF
page numbers**, renames duplicate figure IDs sequentially (`fig_1`, `fig_2`, ...),
extracts the figures from the **original** PDF, and disambiguates repeated Markdown
footnote definitions. It asks Gemini to flag paragraph continuations at chunk
boundaries so those paragraphs can be joined without an artificial break.

Each chunk has a separate `chunks/part-0001/` directory within the temporary
working folder. On failure, successful chunks' responses remain cached: `--resume`
will retry only missing chunks. All temporary PDFs and responses are deleted after
successful publication; on failure they are preserved for manual recovery. A previous
single-file recovery directory remains compatible.

Chunk boundaries can still affect tables, reference-list entries and paragraph joins:
review the result when a chunk falls inside a complex element. Gemini's figure
bounding boxes remain approximate.

## Interactively convert missing PDFs

Discover all PDFs without a corresponding `.md` below `~/syncthing/papers`, review
them in a keyboard-driven list, and **select none by default**:

```sh
pdf-transcriber convert-missing
pdf-transcriber --convert-missing  # Alias
pdf-transcriber convert-missing --papers-dir '/path/to/papers'
pdf-transcriber convert-missing -N 100 --dry-run
pdf-transcriber convert-missing --standard --no-chunk
pdf-transcriber convert-missing --parallel 4
```

This uses the same discovery rule as `pdfmd`: recursively locate `*.pdf` without a
same-basename `.md`, newest modification date first, excluding Syncthing conflict
files. It additionally excludes this tool's temporary recovery PDFs. `-N` limits
the displayed candidate pool to the newest N files.

Keyboard shortcuts in the selection screen:

| Key | Action |
| --- | --- |
| `↑`/`↓` or `k`/`j` | Move the cursor through the PDFs |
| `Space` | Toggle the currently highlighted PDF |
| `a` | Select all candidates |
| `n` | Select none |
| `r` | Select PDFs **created today** (local calendar date) |
| `1`–`9` | Select PDFs created within the previous N×24 hours |
| `Enter` | Finish selection, then review an explicit confirmation |
| `q`, `Esc`, `Ctrl-C` | Cancel without converting anything |

On macOS, selection shortcuts use filesystem **creation time** (`st_birthtime`).
Other filesystems that do not expose creation time fall back to modification time,
with a warning. The display remains sorted by modification time, matching `pdfmd`.

The CLI requires an interactive TTY for this command and **never starts paid API
requests** when the selection is empty, cancelled or not explicitly confirmed.
After selecting files and pressing Enter, you see a **compact review table** with
readable author/year/title and bibliographic key. At this last stage:

- Type `yes` to begin paid conversion; Enter alone always cancels.
- Type `b` to return to the selector, **keeping your previous selection** so you can
  correct mistakes without starting over.
- Type `v` to toggle the **full, untruncated paths** if you need to inspect a name.

`--dry-run` prints the same review table and issues no API requests. Selected PDFs
are processed with up to three parallel conversions by default, and one failure
does not prevent the remaining choices from being attempted.

## Terminal presentation

When a real terminal is available, `go` and `convert-missing` use a compact,
colour-coded live display (provided by Rich):

- **Amber** `● pending`: not started, no API request yet.
- **Blue** `● active`: currently uploading, awaiting Gemini or assembling output.
- **Green** `✓ done`: the Markdown (and figures, where present) was published.
- **Red** `✕ failed`: the PDF could not be converted; a recovery command is retained.

The overall progress bar tracks completed and failed **documents**, while long PDFs
also display completed **chunks** out of the total. Gemini requests have an animated
spinner and elapsed time; there is no invented per-request percentage or ETA.
The active PDF's temporary work-folder name and current stage are visible while
processing. The live display disappears when processing ends, leaving the initial
selection review and a concise, colour-coded **final results table**. Full recovery
paths and commands remain visible on failures. Stray arrow-key presses during a
long request are not echoed into the transcript.

Text redirected to a file or piped to another command stays plain rather than emitting
terminal control sequences. Rich also honours `NO_COLOR` and standard terminal colour
capabilities. Recovery data, billing confirmation, output filenames, chunking and
Gemini request behaviour are unchanged.

## Citation enrichment (compact schema v4)

`pdf-transcriber go 'Paper.pdf'` now performs both stages: PDF transcription
and citation analysis. On success it **replaces** `Paper.md` with its linked
version and writes `Paper.citations.json` (plus any figures). The source
Markdown remains unchanged if analysis or publication fails.

```sh
pdf-transcriber transcribe 'Paper.pdf'        # PDF -> .md + image folder (old go)
pdf-transcriber find-citations 'Paper.md'     # updates .md in place + writes .citations.json
pdf-transcriber find-citations 'Paper.md' --standard
pdf-transcriber _dev prompt transcribe 'Paper.pdf'
pdf-transcriber _dev prompt find-citations 'Paper.md' > Paper.citation-request.txt
pdf-transcriber _dev apply find-citations 'Paper.md'    # reads Paper.citation-proposals.json
```

The `_dev` namespace is hidden from top-level help. For a manual AI Studio run,
save Gemini's JSON result as `Paper.citation-proposals.json` and run
`_dev apply find-citations`; `--json FILE` accepts another path. `--force`
permits replacing an existing `.citations.json` only when the Markdown is
unenriched; it never adds another layer of citation links.

The compact **v4** citation schema uses one `LINE|TEXT` row per distinct
citation string on a source line. Python resolves identities using explicit
printed numeric labels or unique author/year evidence, including coauthors and
year suffixes. It abstains on ambiguous, incomplete or unsupported citations.
Legacy v1–v3 responses remain readable, but their proposed targets are ignored.
**Every identical occurrence on a line** receives its own persistent `cite-NNNN`
ID. Protected spans remain unchanged; missing source text is reported.

`Paper.citation-match-failures.jsonl` records each unresolved occurrence with
its source hash, line, citation text, reason, matching strategy, up to eight
candidate references and a short source excerpt. An empty log is valid.
Reference IDs are opaque document-local `src-ref-NNN` handles; no BibTeX keys
are generated. Existing metadata and failure logs require `--force` to replace.

Gemini extracts bibliographic metadata including authors, editors, year,
title, container, DOI, ISBN, publisher and other fields where printed. The
`Paper.citations.json` sidecar stores this metadata and occurrence IDs. Linking
to the user's wider bibliography is deferred.

Citation requests use the Gemini Interactions API and **Flex** by default;
`--standard` is opt-in. Citation analysis stores a temporary
`.pdf-citations-*` workspace beside the Markdown and cleans it only after a
successful run. Failures preserve the prompt, response (if completed), source
SHA-256 and traceback. Resume without paying for the same completed request:

```sh
pdf-transcriber find-citations 'Paper.md' --resume '/path/to/.pdf-citations-...'
```

`go` reuses an existing `.md` when PDF transcription previously succeeded but
citation analysis did not. `transcribe` retains the old refusal to overwrite
existing Markdown. An existing `.citations.json` blocks a new citation run
unless `find-citations --force` is specified **and the Markdown is unlinked**.
Already-enriched Markdown cannot be processed twice; restore the original
from version control before regenerating citation links. Old `.cited.md` files
block in-place enrichment even with `--force` so they are not silently deleted;
move or remove them manually. `convert-missing` remains transcription-only.
### Malformed individual citation proposals

A completed Gemini response can contain an occasional invalid `LINE|TEXT`
row. The parser **skips only that row** and proceeds with other valid citations.
No unsafe edit is attempted. Skipped proposals, with their original raw
strings and reasons, are recorded under `rejected_mentions` in
`Article.citations.json`, and counted as `mentions_rejected` in the summary.
Source SHA-256 checks, bibliography validation and protection against
ambiguous links remain strict.

When a completed response fails during application, use
`find-citations --resume WORKSPACE` after updating the tool to reuse the
existing response without another Gemini request.

### Fallback for citation text interrupted by comments

The citation linker first searches for the proposed citation text **exactly**
on its reported original source line. Only if no exact match exists does it
retry against a virtual view that removes HTML comments and RMX review-comment
bodies and collapses whitespace. The virtual view is used for locating only:
the updated `.md` wraps the **unchanged original source span**, including
any inline page-break comment and its original spaces. Gemini responses that
contain accidental line breaks within citation text can therefore be recovered
if they correspond to a citation on one original source line.

For example, `Borghi, <!-- pdf page 2 --> 2021` can link when Gemini
returns `Borghi, 2021` or `Borghi, \\n 2021`. This does **not** permit
changes across physical source lines, RMX annotation boundaries, nested links
or code. Complete inline HTML comments are preserved inside citation link
labels; citations requiring an edit across an RMX comment body are skipped.
Conflicting targets for normalized-equivalent citations remain unresolved.
Fallback-derived mentions include `match_method` and `proposed_text` in the
final JSON for easy auditing. The source round-trip check remains exact.


## Comparing Gemini with Codex CLI for citation discovery

Gemini remains the default citation backend. Add `--codex` to run the installed
**Codex CLI** non-interactively, using **GPT-6 Luna** (model ID `gpt-6-luna`)
by default. `--model` overrides the model for whichever citation provider
was selected; it affects **citation discovery only**, not PDF transcription.

```sh
# A previously converted Markdown file (requires Codex CLI, no Gemini key):
pdf-transcriber find-citations 'Article.md' --codex
pdf-transcriber find-citations 'Article.md' --codex --model gpt-6.1-sol

# All-in-one: Gemini transcribes the PDF; Codex analyses the resulting Markdown:
pdf-transcriber go 'Article.pdf' --codex --model gpt-6-luna

# Gemini citation discovery is still the default:
pdf-transcriber find-citations 'Article.md'
pdf-transcriber find-citations 'Article.md' --model gemini-3.8-flash
```

To install Codex CLI (it is **not** a Python package dependency):

```sh
npm install -g @openai/codex
codex login
codex --version
```

Alternatively, on macOS: `brew install --cask codex`. The command must be on
`PATH` and authenticated, and the selected model must be available to that
Codex account. If it is missing, the tool exits with a helpful installation
message *before* creating a citation workspace; `go --codex` also checks it
before starting transcription.

The Codex backend runs `codex exec --ephemeral --sandbox read-only
--skip-git-repo-check --model MODEL --output-last-message FILE -`, feeding
the same line-numbered article prompt on standard input. It includes the
compact schema as text in the Codex prompt and validates the final response
with the same deterministic citation parser used by Gemini. It intentionally
does not request Codex strict `--output-schema`: that would require all of
our otherwise-optional bibliography fields, increasing output tokens.
Codex may not return valid JSON, in which case the run fails safely and
preserves its response and diagnostic logs for inspection.

The temporary `.pdf-citations-*` folder records the provider and model,
plus Codex stdout/stderr and final message when applicable. On a completed
run the final `Article.citations.json` now includes a `generation` field
with `provider` and `model`; this helps compare runs. Both backends overwrite `Article.md` and write `Article.citations.json`,
never `Article.cited.md`. For model comparisons, restore the original unlinked
Markdown from version control between runs, or use separate directories.
`--force` cannot safely reprocess already-enriched Markdown.

The `--standard` flag applies to Gemini only. On
`find-citations --codex`, it is rejected because Codex CLI manages its own
account and service. With `go --standard --codex`, it applies to the
**Gemini PDF transcription** stage, not Codex citation discovery.

Resume a failed Codex citation run with
`pdf-transcriber find-citations 'Article.md' --resume .pdf-citations-XXXX`.
The persisted provider/model are reused; the Codex CLI (and its credentials)
are **not** required when the complete cached `response.json` is available.
The citation sidecar is produced with the identical reference IDs and
protected-edit rules as Gemini's output.


### Safe in-place publication

Citation discovery and the hidden developer apply command now produce `Article.md` (updated in place), `Article.citations.json` and
`Article.citation-match-failures.jsonl`. Before any
overwrite, the linker validates source hashes, reference targets, and a
round-trip reconstruction of the unmodified source. All three outputs are staged
in temporary files; the sidecars are installed first, and the Markdown
replacement happens last. Temporary backups permit rollback on ordinary
file-write failures. A failed model call or invalid response never overwrites
the Markdown.

The previous unlinked Markdown is deliberately not retained as a second
permanent file on successful enrichment. Commit it to Git beforehand if
you might need to compare providers or return to the original. Existing
legacy `.cited.md` files are not deleted automatically.

### Parallel batch transcriptions

`convert-missing` processes **up to three PDFs concurrently by default**. Change
this with `--parallel N` (short `-j N`, supported range 1–16); `-j 1`
retains the old sequential behavior, including its immediate Ctrl-C handling.
For example:

```sh
pdf-transcriber convert-missing                # Default: 3 simultaneous PDFs
pdf-transcriber convert-missing --parallel 4   # Four concurrent PDFs
pdf-transcriber convert-missing -j 1           # Original serial mode
pdf-transcriber convert-missing -N 40 -j 3     # 40 newest candidates; 3 workers
```

Selection remains empty by default, and the user must still approve paid
requests explicitly before any PDF is submitted. The dashboard tracks active,
completed and failed PDFs separately: worker threads relay their progress to
the main UI thread, preventing progress from being mixed across documents.
PDF preflight, chunk creation and figure extraction also run on the main thread
because PyMuPDF does not support threaded use; Gemini requests run concurrently.
Only `N` selected PDFs are scheduled at a time; individual failures do not
block others and each PDF keeps its original crash-recovery folder and command.
After Ctrl-C in parallel mode, no new PDFs are submitted, while already-running
requests finish to avoid discarding paid work. PDFs not yet submitted remain
pending in the summary.

The default of **3** is a conservative starting point for a **Gemini Tier 1**
paid account, not a promise of quota headroom. Rate limits are determined per
project/model and include requests/minute, input tokens/minute and requests/day;
Tier 1 can also face a $10 rolling ten-minute spend-rate cap. The Gemini SDK's
existing temporary-error retry/backoff remains in effect. If 429 errors recur,
try `-j 2` or `-j 1` and consult the account's limits in AI Studio.
The free input-only token counter is not enabled because it cannot estimate
output/thinking tokens or guarantee quota acceptance, and using it would add
additional API requests to every PDF. Parallelism applies only to
`convert-missing`, which is **transcription-only** as before.
