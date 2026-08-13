"""Canonical 167-class (residue, heavy-atom) vocabulary for protein models.

The 20 standard amino acids contribute exactly 167 heavy-atom types
(hydrogens, OXT, and non-standard residues are excluded). Preprocessing
filters hydrogens first, then drops atoms outside this vocabulary and
reports the heavy-atom drop rate.
"""

from __future__ import annotations

BACKBONE = ("N", "CA", "C", "O")

SIDECHAINS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

VOCAB: dict[tuple[str, str], int] = {
    (residue, atom): index
    for index, (residue, atom) in enumerate(
        (residue, atom)
        for residue in sorted(SIDECHAINS)
        for atom in BACKBONE + SIDECHAINS[residue]
    )
}
FEATURE_DIM = len(VOCAB)
assert FEATURE_DIM == 167, FEATURE_DIM

# Standard heavy-atom names start with their element letter, so each of the
# 167 classes maps to one of C/N/O/S. Used for the element-only feature mode
# that matches the ATOM3D/GVP baseline featurization (hydrogens excluded).
ELEMENTS = ("C", "N", "O", "S")
ELEMENT_OF_CLASS = tuple(
    ELEMENTS.index(atom[0]) for (_, atom), _ in sorted(VOCAB.items(), key=lambda x: x[1])
)
ELEMENT_DIM = len(ELEMENTS)

LABEL_NAMES = ("rmsd", "gdt_ts", "gdt_ha", "tm")
