# Changelog

## Unreleased

- Added explicit CSV/historian observation loading with timestamp, unit,
  pressure-basis, quality-mask, quiet-window, initialization/blind-split, and
  fixed-parameter comparison contracts.
- Added private asset-parameter provenance auditing for JSON/YAML configuration
  files. Facility values remain outside the public package.
- Added deterministic adaptive time stepping for `HomogeneousTank`, including
  event-time alignment, EOS-failure retry, change-limit retry, and structured
  failure diagnostics.
- Added a GUI-neutral `NetworkDraft` JSON contract for icon, port, position, and
  link authoring.
- Extended CI to Python 3.13 and current Node.js action runtimes.
