# Manuscript

`main.tex` is an anonymous draft using the latest official ICLR style currently available (ICLR
2026). The official ICLR 2027 template was not available at bootstrap; replace the five bundled
support files when it is released.

Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

Overleaf main document: `paper/main.tex` when the full repository is imported.

Linked project: <https://www.overleaf.com/project/6a6cdb522d6aa17eed95038d>. It is connected to
`lilywchen/moe-sparse-adaptation` through Overleaf's GitHub integration. After manuscript changes,
push the tested commit to GitHub, pull it into Overleaf, recompile, and record the successful build
in `PROGRESS.md`.
