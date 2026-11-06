"""Structural protocols shared by the parser, annotator and adapters."""

from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy
import numpy.typing

from rnapolis.common import MissingResidue, ResidueAuth, ResidueLabel


@runtime_checkable
class AtomLike(Protocol):
    """Minimal atom interface required by geometry-aware code."""

    entity_id: Optional[str]
    name: str

    @property
    def coordinates(self) -> numpy.typing.NDArray[numpy.floating]:
        """Return Cartesian coordinates as a NumPy vector."""
        ...


@runtime_checkable
class ResidueLike(Protocol):
    """Residue interface required by the adapter conversion pipeline."""

    label: Optional[ResidueLabel]
    auth: Optional[ResidueAuth]
    model: int
    one_letter_name: str

    @property
    def chain(self) -> Optional[str]:
        """Return the chain identifier."""
        ...

    @property
    def number(self) -> Optional[int]:
        """Return the residue number."""
        ...

    @property
    def icode(self) -> Optional[str]:
        """Return the insertion code."""
        ...

    @property
    def full_name(self) -> Optional[str]:
        """Return a human-readable residue identifier."""
        ...

    @property
    def entity_id(self) -> Optional[str]:
        """Return the entity identifier, if known."""
        ...

    @property
    def is_nucleotide(self) -> bool:
        """Return True if the residue is a nucleotide."""
        ...

    def __lt__(self, other: "ResidueLike") -> bool:
        """Order residues consistently across implementations."""
        ...

    def __hash__(self) -> int:
        """Hash residues consistently across implementations."""
        ...


@runtime_checkable
class GapAwareResidueLike(ResidueLike, Protocol):
    """Residue interface required by gap detection."""

    atoms: Sequence[AtomLike]

    def find_atom(self, atom_name: str) -> Optional[AtomLike]:
        """Return an atom by name, or None."""
        ...

    def is_connected(self, next_residue_candidate: "GapAwareResidueLike") -> bool:
        """Return True when this residue is connected to the next one."""
        ...


@runtime_checkable
class StructureLike(Protocol):
    """Structure interface required by the adapter conversion pipeline."""

    residues: Sequence[ResidueLike]

    def find_residue(
        self,
        label: Optional[ResidueLabel],
        auth: Optional[ResidueAuth],
    ) -> Optional[ResidueLike]:
        """Find a residue by label or author-style identifier."""
        ...


@runtime_checkable
class GapAwareStructureLike(StructureLike, Protocol):
    """Structure interface required by gap-aware secondary structure mapping."""

    missing_residues: Sequence[MissingResidue]
    sequence_by_entity: Mapping[str, str]
