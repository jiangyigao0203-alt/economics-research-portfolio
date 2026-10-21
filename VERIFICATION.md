# Verification notes

Packaging checks:

- All 18 included Python files parsed successfully with Python's AST parser.
- Included source files match their original local files byte for byte; SHA-256 hashes are recorded in SOURCE_MANIFEST.md.
- Five selected aggregate CSV files and one regression-summary text file parsed successfully and contain data rows.
- A scan for common hard-coded password, token, and private-key patterns found no matches. This is a limited pattern check, not a security guarantee.
- Original project folders were left untouched.

The full empirical pipelines, dependency installation, data downloads, numerical estimates, statistical assumptions, and exact correspondence between code and saved outputs were not validated. No claim of successful replication is made.

- File selection was reviewed to omit duplicate implementations, disconnected helpers, bulky outputs, and unrelated code; see FILE_SELECTION.md.
