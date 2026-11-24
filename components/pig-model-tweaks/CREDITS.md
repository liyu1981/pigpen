# Credits

`extensions/pig-model-tweaks` is a Go port of the three model extensions of
**pi-tweaks**, https://github.com/liyu1981/pi-tweaks, by **Yu Li**
(MIT, Copyright (c) 2026 Yu Li).

- Original: `@liyu1981/pi-tweaks` 0.1.1, commit `65e6ccfbb77956aa15ce34e7ca9735e1d476c292`
- The unmodified original (the three extensions and the shared `src/` they use)
  is kept under [`port/oracle/`](port/oracle) as the equivalence oracle, with its
  license at [`port/oracle/LICENSE`](port/oracle/LICENSE).
- Ported: `pt-remember-model` → `pmt-remember-model`,
  `pt-openrouter-lock-provider` → `pmt-openrouter-lock-provider`,
  `pt-model-guard` → `pmt-model-guard-pref`. Names take the `pmt-` prefix and the
  settings file is `pmt-settings.json` where the original used `pt-` and
  `pi-tweaks-settings.json`.
- Not ported: `pt-subagent`, the original's fourth extension (added after the
  pinned release), which spawns isolated child `pi` processes; it is a separate
  concern from model selection.

The Go code and its tests were written for this Package by Yu Li. Behavior,
wording and defaults follow the original; the deliberate differences are listed
in the Package README. No file of the original is modified; the port is a
separate implementation.

## Free-model provider and refresh commands

`free_catalog.go`, `free_provider.go`, `free_stream.go`, `free_chat.go`,
`free_responses.go`, `refresh_commands.go` and their tests are independent MIT
implementations written for this Package. The idea of a keyless provider for the
free models on OpenCode Zen, and the behavior of its discovery and streaming, are
described by [`pi-opencode-free`](../pi-opencode-free) by Pedro Alexis
(GPL-3.0-or-later, commit `0f4c16bade33106847ae4abd8d2be2d1b0058e36`). No source
from that GPL-licensed Package was copied into this MIT-licensed Package, and the
GPL-licensed Package is kept unchanged as a separate component. Thanks are due to
its author for documenting the Zen free tier.
