# RNApolis

A Python library and CLI utilities for RNA bioinformatics — parsing PDB/mmCIF structures, annotating secondary structure, and clustering 3D conformations.

- **Documentation:** https://tzok.github.io/rnapolis-py/
- **Notebooks:** https://github.com/tzok/rnapolis-py/tree/main/notebooks

## Development

This repository uses `uv` for environment management, locking, and builds.

Install the runtime environment:

```bash
uv sync --locked
```

Install the development tools and run tests:

```bash
uv sync --locked --group dev
uv run --no-sync python -m pytest
```

Install the documentation toolchain and build the site:

```bash
uv sync --locked --group docs
uv run --no-sync mkdocs build --site-dir site
```

Build source and wheel distributions:

```bash
uv build
```

## Utilities

### `annotator`

Extracts and classifies RNA secondary structure from 3D coordinates, detecting base pairs (Leontis-Westof + Saenger), stacking, base-ribose, and base-phosphate interactions. Prints dot-bracket to stdout (pseudoknot-ordered; `--extended` encodes non-canonical pairs) and optionally writes BPSEQ (`--bpseq`), CSV (`--csv`), JSON (`--json`), or GraphViz DOT (`--dot`). `--find-gaps` splits chains with missing segments (>2.4 Å gaps).

### `adapter`

Converts base-interaction output from external tools into RNApolis' secondary-structure format, reusing `annotator`'s output options. Supports FR3D, DSSR, RNAView, BPNet, MAXIT, BARNABA, MC-Annotate, and DNATCO; the source tool is auto-detected from file patterns (override with `--tool`). Usage: `adapter <structure> [external files...]`.

### `aligner`

Aligns two PDB/mmCIF structures with PyMOL and writes trimmed copies containing only matching residues. **Requires PyMOL.** Usage: `aligner -o <out_dir> [-f PDB|mmCIF|keep] <file1> <file2>`.

### `clashfinder`

Detects atomic clashes in PDB/mmCIF structures. Flags: `--ignore-occupancy`, `--nucleic-acid-only`, `--ignore-autoclashes`, `--require-same-atom-name`, `--enable-molprobity-mode` (include weak clashes), `--csv`.

### `distiller`

Clusters RNA 3D structures by geometric similarity. Two modes: `approximate` (default; PCA-reduced feature distances, fast) and `exact` (all-vs-all nRMSD). Four methods: `hierarchical` (default), `affinity-propagation`, `facility-location`, `radius-graph`. Accepts file paths as args or via stdin. Outputs JSON (`--output-json`) and dendrogram/MDS plots (`--visualize`).

### `metareader`

Extracts mmCIF categories into JSON for quick inspection with tools like `jq`. `--category`/`-c` selects categories (default `struct`); `--list-categories`/`-l` lists available ones.

```
$ metareader -c refine 8af0.cif.gz | jq -r .refine[0].ls_d_res_high
2.43
```

### `molecule-filter`

Filters an mmCIF file by entity type (default `polyribonucleotide`), preserving all interdependent mmCIF categories for the selected chains. `--type` selects the entity type.

### `motif-extractor`

Reads a secondary structure in BPSEQ (`--bpseq`) or dot-bracket (`--dbn`) format and lists its structural elements (single strands, stems, loops, hairpins).

### `transformer`

Copies one column to another within an mmCIF category (e.g. set `auth_asym_id` from `label_asym_id`). Usage: `transformer <input> <output> --category atom_site --copy-from label_asym_id --copy-to auth_asym_id`.

### `rfam-folder`

Folds consensus secondary structures for RNA sequence(s) — a single sequence or a FASTA file. `--family` targets a specific Rfam family (otherwise the whole DB is searched); `--no-fold` skips RNAfold; `--count` limits structures per sequence. **Requires [Infernal](http://eddylab.org/infernal/).**

### `unifier`

Normalizes a set of PDB/mmCIF files so they share identical residues: standardizes canonical-nucleotide atom names/order, unifies residue identifiers (chain/number/icode), and drops residues whose atom counts differ across files. Usage: `unifier -o <out_dir> [-f PDB|mmCIF|keep] <files...>`.

### `splitter`

Splits a multi-model PDB or mmCIF file (e.g. NMR ensembles) into one file per model. Usage: `splitter -o <out_dir> [-f PDB|mmCIF|keep] <file>`.

### `quick-filter`

Filters atoms from a PDB/mmCIF file while preserving non-atom records (headers, ANISOU, etc.). `--mode` picks `nucleic-acid` (default) or `protein`; `--keep-ligands`/`--keep-waters`/`--keep-ions` retain those classes; `--altloc`, `--chains`, and `--model` further restrict output. Prints filtered content to stdout.
