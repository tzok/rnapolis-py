# RNApolis

A Python library and utilities containing RNA-related bioinformatics functions and classes.

## Utilities

### `annotator`

**Annotator** is a cutting-edge command-line tool engineered for the extraction and classification of secondary structures from biomolecular coordinates. This tool stands out in its ability to identify and categorize various base interactions, including base pairs, stacking, base-ribose, and base-phosphate interactions.

#### Comprehensive Classification:

- **Base Pair Analysis**: Utilizes the Leontis-Westhof and Saenger methodologies for precise base pair classification.
- **Stacking and Interactions**: Efficiently finds and classifies stacking, base-ribose, and base-phosphate interactions.

#### Flexible Output Formats:

- **Dot-Bracket Notation**: Produces results in dot-bracket format, optimized for pseudoknot-order assignment. With the `--extended` option, it also encodes non-canonical base pairs in a unique way.
- **BPSEQ Format**: Option to output results in BPSEQ format, detailing base interactions and classifications.
- **CSV Output**: Generates a CSV file containing detailed classifications of all base interactions.
- **JSON Export**: When enabled with `--json`, it outputs a JSON file detailing all identified base interactions and structural elements, including single strands, stems, and loops.
- **GraphViz Integration**: The `--dot` option enables the creation of a GraphViz dot file, illustrating the relationships between various structural elements.

#### Enhanced Features:

- **Gap Detection**: Capable of detecting and handling gaps in the PDB chain, breaking it into multiple strands for gaps larger than 2.4 Å.

#### Usage:

Execute `annotator.py` with your file path to initiate analysis. Customize your analysis and outputs using these options:

- `--bpseq [PATH]`: Specify path for BPSEQ file output.
- `--csv [PATH]`: Set path for CSV file output.
- `--json [PATH]`: Define path for JSON file output.
- `--extended`: Enable detailed secondary structure analysis.
- `--find-gaps`: Activate gap detection and handling.
- `--dot [PATH]`: Choose path for GraphViz DOT file output.

For help and more information, use `-h` or `--help`.

#### Examples

```
$ annotator --csv 8af0.csv --json 8af0.json --extended 8af0.cif.gz
    >strand_C
seq GCCCGCTUGUTACGCAGGC
cWW ((((((.((.((.((((((
cWH .........(.........
    >strand_D
seq GCCCGCTUGUTACGCAGGC
cWW )))))).)).)).))))))
cWH .........).........
```

<details>
<summary>Click here to see the output CSV</summary>

| nt1   | nt2   | type      | classification-1 | classification-2 |
| ----- | ----- | --------- | ---------------- | ---------------- |
| C.G26 | D.C44 | base pair | cWW              | XIX              |
| C.C27 | D.G43 | base pair | cWW              | XIX              |
| C.C28 | D.G42 | base pair | cWW              | XIX              |
| C.C29 | D.G41 | base pair | cWW              |                  |
| C.G30 | D.C40 | base pair | cWW              | XIX              |
| C.C31 | D.G39 | base pair | cWW              | XIX              |
| C.U33 | D.A37 | base pair | cWW              | XX               |
| C.G34 | D.C36 | base pair | cWW              | XXVIII           |
| C.U35 | D.U35 | base pair | cWH              |                  |
| C.C36 | D.G34 | base pair | cWW              | XXVIII           |
| C.A37 | D.U33 | base pair | cWW              | XX               |
| C.G39 | D.C31 | base pair | cWW              | XIX              |
| C.C40 | D.G30 | base pair | cWW              | XIX              |
| C.G41 | D.C29 | base pair | cWW              |                  |
| C.G42 | D.C28 | base pair | cWW              | XIX              |
| C.G43 | D.C27 | base pair | cWW              | XIX              |
| C.C44 | D.G26 | base pair | cWW              | XIX              |
| C.G26 | C.C27 | stacking  | upward           |                  |
| C.C27 | C.C28 | stacking  | upward           |                  |
| C.C28 | C.C29 | stacking  | upward           |                  |
| C.C28 | D.G43 | stacking  | inward           |                  |
| C.C29 | C.G30 | stacking  | upward           |                  |
| C.G30 | C.C31 | stacking  | upward           |                  |
| C.G30 | D.G41 | stacking  | inward           |                  |
| C.C31 | C.C32 | stacking  | upward           |                  |
| C.C32 | C.U33 | stacking  | upward           |                  |
| C.U33 | C.G34 | stacking  | upward           |                  |
| C.G34 | C.U35 | stacking  | upward           |                  |
| C.G34 | D.A37 | stacking  | inward           |                  |
| C.U35 | C.C36 | stacking  | upward           |                  |
| C.A37 | C.C38 | stacking  | upward           |                  |
| C.A37 | D.G34 | stacking  | inward           |                  |
| C.C38 | C.G39 | stacking  | upward           |                  |
| C.G39 | C.C40 | stacking  | upward           |                  |
| C.G41 | C.G42 | stacking  | upward           |                  |
| C.G41 | D.G30 | stacking  | inward           |                  |
| C.G42 | C.G43 | stacking  | upward           |                  |
| C.G42 | D.C29 | stacking  | inward           |                  |
| C.G43 | C.C44 | stacking  | upward           |                  |
| C.G43 | D.C28 | stacking  | inward           |                  |
| D.G26 | D.C27 | stacking  | upward           |                  |
| D.C27 | D.C28 | stacking  | upward           |                  |
| D.C28 | D.C29 | stacking  | upward           |                  |
| D.G30 | D.C31 | stacking  | upward           |                  |
| D.C31 | D.C32 | stacking  | upward           |                  |
| D.G34 | D.U35 | stacking  | inward           |                  |
| D.U35 | D.C36 | stacking  | inward           |                  |
| D.A37 | D.C38 | stacking  | upward           |                  |
| D.C38 | D.G39 | stacking  | upward           |                  |
| D.G39 | D.C40 | stacking  | upward           |                  |
| D.C40 | D.G41 | stacking  | upward           |                  |
| D.G41 | D.G42 | stacking  | upward           |                  |
| D.G42 | D.G43 | stacking  | upward           |                  |
| D.G43 | D.C44 | stacking  | upward           |                  |

</details>

<details>
<summary>Click here to see the output JSON</summary>

```json
{
  "baseInteractions": {
    "basePairs": [
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 1,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 26,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 19,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 44,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": null
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 15,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 40,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 8,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 33,
            "icode": null,
            "name": "U"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "D",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "lw": "cWW",
        "saenger": "XX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 11,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 36,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XXVIII"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "D",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "lw": "cWH",
        "saenger": null
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 11,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 36,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XXVIII"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "C",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 8,
            "name": "U"
          },
          "auth": {
            "chain": "D",
            "number": 33,
            "icode": null,
            "name": "U"
          }
        },
        "lw": "cWW",
        "saenger": "XX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 15,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 40,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": null
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 19,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 44,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 1,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 26,
            "icode": null,
            "name": "G"
          }
        },
        "lw": "cWW",
        "saenger": "XIX"
      }
    ],
    "stackings": [
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 1,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 26,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 7,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 32,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 7,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 32,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 8,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 33,
            "icode": null,
            "name": "U"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 8,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 33,
            "icode": null,
            "name": "U"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "D",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "C",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 11,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 36,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "C",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 13,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 38,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "C",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 13,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 38,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 15,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 40,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "A",
            "number": 19,
            "name": "C"
          },
          "auth": {
            "chain": "C",
            "number": 44,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "A",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "C",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 1,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 26,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 2,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 27,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 3,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 28,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 4,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 29,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 5,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 30,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 6,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 31,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 7,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 32,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 9,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 34,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "D",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 10,
            "name": "U"
          },
          "auth": {
            "chain": "D",
            "number": 35,
            "icode": null,
            "name": "U"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 11,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 36,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "inward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 12,
            "name": "A"
          },
          "auth": {
            "chain": "D",
            "number": 37,
            "icode": null,
            "name": "A"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 13,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 38,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 13,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 38,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 14,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 39,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 15,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 40,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 15,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 40,
            "icode": null,
            "name": "C"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 16,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 41,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 17,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 42,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "topology": "upward"
      },
      {
        "nt1": {
          "label": {
            "chain": "B",
            "number": 18,
            "name": "G"
          },
          "auth": {
            "chain": "D",
            "number": 43,
            "icode": null,
            "name": "G"
          }
        },
        "nt2": {
          "label": {
            "chain": "B",
            "number": 19,
            "name": "C"
          },
          "auth": {
            "chain": "D",
            "number": 44,
            "icode": null,
            "name": "C"
          }
        },
        "topology": "upward"
      }
    ],
    "baseRiboseInteractions": [],
    "basePhosphateInteractions": [],
    "otherInteractions": []
  },
  "bpseq": "1 G 38\n2 C 37\n3 C 36\n4 C 0\n5 G 34\n6 C 33\n7 T 0\n8 U 31\n9 G 30\n10 U 0\n11 T 28\n12 A 27\n13 C 0\n14 G 25\n15 C 24\n16 A 0\n17 G 22\n18 G 21\n19 C 20\n20 G 19\n21 C 18\n22 C 17\n23 C 0\n24 G 15\n25 C 14\n26 T 0\n27 U 12\n28 G 11\n29 U 0\n30 T 9\n31 A 8\n32 C 0\n33 G 6\n34 C 5\n35 A 0\n36 G 3\n37 G 2\n38 C 1",
  "dotBracket": ">strand_C\nGCCCGCTUGUTACGCAGGC\n(((.((.((.((.((.(((\n>strand_D\nGCCCGCTUGUTACGCAGGC\n))).)).)).)).)).)))",
  "extendedDotBracket": "    >strand_C\nseq GCCCGCTUGUTACGCAGGC\ncWW ((((((.((.((.((((((\ncWH .........(.........\n    >strand_D\nseq GCCCGCTUGUTACGCAGGC\ncWW )))))).)).)).))))))\ncWH .........).........",
  "stems": [
    {
      "strand5p": {
        "first": 1,
        "last": 3,
        "sequence": "GCC",
        "structure": "((("
      },
      "strand3p": {
        "first": 36,
        "last": 38,
        "sequence": "GGC",
        "structure": ")))"
      },
      "description": "Stem 1 3 GCC ((( 36 38 GGC )))"
    },
    {
      "strand5p": {
        "first": 5,
        "last": 6,
        "sequence": "GC",
        "structure": "(("
      },
      "strand3p": {
        "first": 33,
        "last": 34,
        "sequence": "GC",
        "structure": "))"
      },
      "description": "Stem 5 6 GC (( 33 34 GC ))"
    },
    {
      "strand5p": {
        "first": 8,
        "last": 9,
        "sequence": "UG",
        "structure": "(("
      },
      "strand3p": {
        "first": 30,
        "last": 31,
        "sequence": "TA",
        "structure": "))"
      },
      "description": "Stem 8 9 UG (( 30 31 TA ))"
    },
    {
      "strand5p": {
        "first": 11,
        "last": 12,
        "sequence": "TA",
        "structure": "(("
      },
      "strand3p": {
        "first": 27,
        "last": 28,
        "sequence": "UG",
        "structure": "))"
      },
      "description": "Stem 11 12 TA (( 27 28 UG ))"
    },
    {
      "strand5p": {
        "first": 14,
        "last": 15,
        "sequence": "GC",
        "structure": "(("
      },
      "strand3p": {
        "first": 24,
        "last": 25,
        "sequence": "GC",
        "structure": "))"
      },
      "description": "Stem 14 15 GC (( 24 25 GC ))"
    },
    {
      "strand5p": {
        "first": 17,
        "last": 19,
        "sequence": "GGC",
        "structure": "((("
      },
      "strand3p": {
        "first": 20,
        "last": 22,
        "sequence": "GCC",
        "structure": ")))"
      },
      "description": "Stem 17 19 GGC ((( 20 22 GCC )))"
    }
  ],
  "singleStrands": [
    {
      "strand": {
        "first": 5,
        "last": 6,
        "sequence": "GC",
        "structure": "(("
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 5 6 GC (("
    },
    {
      "strand": {
        "first": 8,
        "last": 9,
        "sequence": "UG",
        "structure": "(("
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 8 9 UG (("
    },
    {
      "strand": {
        "first": 11,
        "last": 12,
        "sequence": "TA",
        "structure": "(("
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 11 12 TA (("
    },
    {
      "strand": {
        "first": 14,
        "last": 15,
        "sequence": "GC",
        "structure": "(("
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 14 15 GC (("
    },
    {
      "strand": {
        "first": 24,
        "last": 25,
        "sequence": "GC",
        "structure": "))"
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 24 25 GC ))"
    },
    {
      "strand": {
        "first": 27,
        "last": 28,
        "sequence": "UG",
        "structure": "))"
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 27 28 UG ))"
    },
    {
      "strand": {
        "first": 30,
        "last": 31,
        "sequence": "TA",
        "structure": "))"
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 30 31 TA ))"
    },
    {
      "strand": {
        "first": 33,
        "last": 34,
        "sequence": "GC",
        "structure": "))"
      },
      "is5p": false,
      "is3p": false,
      "description": "SingleStrand 33 34 GC ))"
    }
  ],
  "hairpins": [
    {
      "strand": {
        "first": 19,
        "last": 20,
        "sequence": "CG",
        "structure": "()"
      },
      "description": "Hairpin 19 20 CG ()"
    }
  ],
  "loops": [
    {
      "strands": [
        {
          "first": 3,
          "last": 5,
          "sequence": "CCG",
          "structure": "(.("
        },
        {
          "first": 34,
          "last": 36,
          "sequence": "CAG",
          "structure": ").)"
        }
      ],
      "description": "Loop 3 5 CCG (.( 34 36 CAG ).)"
    },
    {
      "strands": [
        {
          "first": 6,
          "last": 8,
          "sequence": "CTU",
          "structure": "(.("
        },
        {
          "first": 31,
          "last": 33,
          "sequence": "ACG",
          "structure": ").)"
        }
      ],
      "description": "Loop 6 8 CTU (.( 31 33 ACG ).)"
    },
    {
      "strands": [
        {
          "first": 9,
          "last": 11,
          "sequence": "GUT",
          "structure": "(.("
        },
        {
          "first": 28,
          "last": 30,
          "sequence": "GUT",
          "structure": ").)"
        }
      ],
      "description": "Loop 9 11 GUT (.( 28 30 GUT ).)"
    },
    {
      "strands": [
        {
          "first": 12,
          "last": 14,
          "sequence": "ACG",
          "structure": "(.("
        },
        {
          "first": 25,
          "last": 27,
          "sequence": "CTU",
          "structure": ").)"
        }
      ],
      "description": "Loop 12 14 ACG (.( 25 27 CTU ).)"
    },
    {
      "strands": [
        {
          "first": 15,
          "last": 17,
          "sequence": "CAG",
          "structure": "(.("
        },
        {
          "first": 22,
          "last": 24,
          "sequence": "CCG",
          "structure": ").)"
        }
      ],
      "description": "Loop 15 17 CAG (.( 22 24 CCG ).)"
    }
  ]
}
```

</details>

### `clashfinder`

**ClashFinder** is a robust command-line utility designed for detecting atomic clashes in biomolecular structures provided in PDB or mmCIF formats. This tool is essential for researchers needing detailed insights into the spatial arrangement of atoms within molecular structures.

#### Key Features:

- **Versatile File Support**: Compatible with both PDB and mmCIF file formats.
- **Occupancy-Based Filtering**: Clashes are identified considering atoms' occupancies, with an option (`--ignore-occupancy`) to report clashes regardless of occupancy values.
- **Nucleic Acid Focused Analysis**: While capable of detecting all clash types, it can be set to focus exclusively on nucleic acid chains (`--nucleic-acid-only`).
- **Autoclash Handling**: By default, autoclashes (within the same residue) are reported, but this can be disabled (`--ignore-autoclashes`).
- **Atom Name Specific Clashes**: The option `--require-same-atom-name` allows for reporting clashes only between atoms with identical names.
- **MolProbity Mode**: Activating `--enable-molprobity-mode` adjusts the detection criteria to include 'weak' clashes, akin to the standards used in MolProbity.
- **CSV Output**: Results can be stored in a CSV format for easy analysis and sharing (`--csv [PATH]`).

#### Usage:

Run `clashfinder.py` with your desired file path to start the clash detection process. Tailor the clash detection to your specific needs using the following flags:

- `--ignore-occupancy`: Ignore occupancy checks in clash reporting.
- `--nucleic-acid-only`: Limit detection to nucleic acid chains only.
- `--ignore-autoclashes`: Exclude clashes within the same residue.
- `--require-same-atom-name`: Report only clashes involving atoms with the same name.
- `--enable-molprobity-mode`: Include 'weak' clashes in the detection.
- `--csv [PATH]`: Specify path for CSV file output.

For additional information, use `-h` or `--help`.

### `metareader`

**metareader** is a powerful command-line utility designed for extracting and transforming metadata from PDBx/mmCIF files into a JSON format. It stands out for its ability to handle any mmCIF category, including the `atom_site`, making it an indispensable tool for quick and efficient analysis of mmCIF files using JSON-capable software like jq or Python.

#### Key Features:

- **Flexible Data Extraction**: Transforms mmCIF categories into an easily manageable JSON format.
- **Custom Category Selection**: Allows users to specify one or multiple mmCIF categories for extraction (`--category` or `-c`).
- **Comprehensive Category Listing**: Provides an option to list all available categories in a given mmCIF file (`--list-categories` or `-l`), aiding in targeted data analysis.

#### Usage:

Run `metareader.py` with the path to your mmCIF file. Customize your data extraction using these options:

- `--category CATEGORY, -c CATEGORY`: Specify one or more mmCIF categories to extract (default is 'struct').
- `--list-categories, -l`: List all categories available in the specified mmCIF file.

For further assistance, use `-h` or `--help`.

#### Examples

To check X-ray structure resolution:

```
$ metareader --category refine 8af0.cif.gz | jq -r .refine[0].ls_d_res_high
2.43
```

### `molecule-filter`

**molecule-filter** is a specialized command-line tool designed to filter PDBx/mmCIF files based on specific molecular entity types. By default, it focuses on polyribonucleotide entities. Its unique feature is the embedded map of interdependencies between mmCIF categories, ensuring the output file contains comprehensive information related to the selected molecular chains, including data on missing atoms and modified residues.

#### Key Features:

- **Selective Entity Filtering**: Filters information for specific entity types within PDBx/mmCIF files.
- **Default Focus on Polyribonucleotide**: The tool defaults to polyribonucleotide entities but offers flexibility for other types.
- **Comprehensive Data Retention**: Maintains interrelated data from different mmCIF categories, ensuring a thorough and detailed output file.

#### Usage:

Execute `molecule_filter.py` with the path to a PDBx/mmCIF file. Specify the molecule type to filter using the following option:

- `--type {cyclic-pseudo-peptide,other,peptide nucleic acid,polydeoxyribonucleotide,polydeoxyribonucleotide/polyribonucleotide hybrid,polypeptide(D),polypeptide(L),polyribonucleotide}`: Choose a molecule type to filter (default: polyribonucleotide).

For additional help, use `-h` or `--help`.

### `motif-extractor`

**motif-extractor** is an efficient command-line utility designed for analyzing molecular structures in BPSEQ or dot-bracket formats. It expertly generates a comprehensive list of structural elements, including single strands, stems, and loops, facilitating detailed insights into molecular configurations.

#### Key Features:

- **Support for Multiple Formats**: Capable of processing inputs in both BPSEQ and dot-bracket formats.
- **Detailed Structural Analysis**: Identifies and lists key structural elements within the molecular structure, such as single strands, stems, and loops.

#### Usage:

Run `motif_extractor.py` and specify the path to your input file in either BPSEQ or dot-bracket format. Use the following options to define your input file type:

- `--dbn DBN`: Specify the path to a DotBracket format file.
- `--bpseq BPSEQ`: Provide the path to a BPSEQ format file.

For more information, use `-h` or `--help`.

### `transformer`

**transformer** is a highly functional command-line tool designed for processing and modifying PDBx/mmCIF files. Its primary functionality involves taking a specific category within an mmCIF file, such as `atom_site`, and copying data from one column to another. This feature is particularly useful for resolving issues with mmCIF "label" and "auth" naming schemes, allowing users to create a modified mmCIF file where both columns have identical values.

#### Key Features:

- **Targeted Column Copying**: Facilitates the copying of data between columns within a specified mmCIF file category.
- **Naming Scheme Resolution**: Aids in harmonizing "label" and "auth" naming schemes in mmCIF files, ensuring compatibility with various analytical tools.

#### Usage:

To use `transformer.py`, specify the path to your input mmCIF file and the desired output file path. Customize the data manipulation using these options:

- `--category CATEGORY`: Define the category to work on (e.g., `atom_site`).
- `--copy-from COPY_FROM`: Specify the column name to copy data from (e.g., `label_asym_id`).
- `--copy-to COPY_TO`: Indicate the column name to copy data to (e.g., `auth_asym_id`).

For additional guidance, use `-h` or `--help`.
