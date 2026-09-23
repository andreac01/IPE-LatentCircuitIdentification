# Figures for `tree_search.md`

Regenerate the three data figures with:

```
python3 scripts/make_tree_search_figures.py
```

They are built from the CSVs in `experiments/faithfulness_completeness_v2/` only — no model, no GPU.

| file | used in | source |
|---|---|---|
| `f1_vs_threshold.png` | §6.4 | `faithfulness_completeness_v2.csv` |
| `gap_and_incompleteness.png` | §6.5 | `faithfulness_completeness_v2.csv` |
| `completeness_by_class.png` | §6.6 | `incompleteness_by_class_v2.csv` |
| `v2_notebook_overview.png` | — | as emitted by the notebook (superseded by the three above) |
| `v2_notebook_scatter.png` | — | as emitted by the notebook; has the random-`K` clouds the §6.6 version drops |
| **`demo_tree.png`** | §7.2 | **to add** — screenshot, tree search w/ limited level width |
| **`demo_acdc.png`** | §7.2 | **to add** — screenshot, ACDC on the same prompts |

Palette is Okabe–Ito, assigned in fixed order; every series carries a distinct marker and dash as
well as a colour, so the figures survive greyscale and colour-vision deficiency.

The random-`K` scatter clouds are computed inline in the notebook and not persisted to CSV, which is
why `completeness_by_class.png` shows only the ground-truth per-class points. Persisting them would
make that figure fully reproducible without a GPU too.
