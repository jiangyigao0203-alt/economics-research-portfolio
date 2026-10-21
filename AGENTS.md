# Public research-code conventions

This repository is intended for readers who have not seen the author's local workspace or chat history.

- Give new source files descriptive snake_case names that identify the task, method, or output. Avoid numbered versions, personal names, and names such as `main`, `start`, or `ultimate` when a domain-specific name is available.
- Begin each analysis script with its research purpose, required input schema, important configuration, outputs, working-directory assumptions, and any import-time execution. Document substantive functions' inputs, outputs, units, and statistical assumptions where the implementation is not obvious.
- Write comments for an outside reader. Explain why a step is necessary and distinguish methods with different statistical meanings. Remove conversational references such as "as requested" and obsolete run commands.
- Keep presentation edits separate from changes to estimation, sample construction, seeds, thresholds, and transforms. Verify that documentation-only edits preserve executable logic. Disclose and validate any intentional behavior change.
- Use portable paths and document required external data access. Keep secrets, raw licensed data, binaries, build directories, generated panels, and bulk simulation output out of Git.
- Prefer the maintained workflow over overlapping historical experiments. Record selection decisions, rename mappings, and source provenance. Update all README links and run commands when filenames change.
- Describe validation accurately. Archived result tables are not proof that the current code reproduces them; distinguish syntax, source-equivalence, build, smoke-run, and empirical-replication checks.
