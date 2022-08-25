# RNApolis

A Python library containing RNA-related bioinformatics functions and classes.

## `rnapolis/common.py`

Main classes:

- `ResidueLabel` and `ResidueAuth`: to address residues in both PDB and mmCIF formats
- `Residue`: to represent a single nucleotide by its label/auth

Main enums:

- `Molecule`
- `GlycosidicBond`
- `LeontisWesthof`
- `Saenger`
- `StackingTopology`
- `BR`
- `BPh`

## `rnapolis/secondary.py`

Main classes:

- `Interaction`, `BasePair`, `BasePhosphate`, `BaseRibose`, `OtherInteraction`: to represent different kinds of residue-residue interactions
- `Structure2D`: to represent RNA secondary structure

## `rnapolis/tertiary.py`

Main classes:

- `Atom`: to represent a single atom
- `Residue3D`, `BasePair3D`, `Stacking3D`: to extend classes from `rnapolis/secondary.py` with data from 3D (atoms, torsion angles, etc.)
- `Structure3D`: to represent RNA tertiary structure

Main functions:

- `read_3d_structure`: parses a PDB or an mmCIF file into a `Structure3D` object

## `rnapolis/utils.py`

Main functions:

- `torsion_angle`: calculates value of a torsion angle
- `angle_between_vectors`: calculates value of an angle between two vectors

## `rnapolis/annotator.py`

Main functions:

- `extract_secondary_structure`: to find all interactions in a `Structure3D` object to create a `Structure2D` one
