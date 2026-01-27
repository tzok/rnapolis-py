import configparser
import importlib.resources
import numpy as np
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

from collections import Counter

from rnapolis.tertiary_v2 import Atom, Residue

# --- Atom Definitions ---
# Purines (A, G, DA, DG)
PURINE_CORE_ATOMS = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}
# Pyrimidines (C, T, U, DC, DT)
PYRIMIDINE_CORE_ATOMS = {"N1", "C2", "N3", "C4", "C5", "C6"}

# Atoms used for defining the canonical face orientation (Right Hand Rule)
# Purine: N9 -> C4 -> C5 (Cross product of N9-C4 and N9-C5 vectors)
PURINE_FACE_ATOMS = ("N9", "C4", "C5")
# Pyrimidine: N1 -> C6 -> C5 (Cross product of N1-C6 and N1-C5 vectors)
PYRIMIDINE_FACE_ATOMS = ("N1", "C6", "C5")

# --- Hydrogen Bond Definitions ---
BASE_ACCEPTORS = {
    "A": ["N1", "N3", "N7"],
    "G": ["N3", "O6", "N7"],
    "C": ["O2", "N3"],
    "U": ["O2", "O4"],
    "T": ["O2", "O4"],
}

PHOSPHATE_ACCEPTORS = ["OP1", "OP2", "O3'", "O5'"]

SUGAR_ACCEPTORS = ["O2'", "O4'"]


# antecedent: The reference atom (defines the angle).
# donor: The interacting Donor atom.
class AtomPair(NamedTuple):
    antecedent: str
    donor: str


DONORS: Dict[str, List[AtomPair]] = {
    "A": [
        AtomPair("C6", "N6"),  # Exocyclic
        AtomPair("N1", "C2"),  # Weak C-H
        AtomPair("N7", "C8"),  # Weak C-H
        AtomPair("C2'", "O2'"),  # Hydroxyl (RNA only)
    ],
    "G": [
        AtomPair("C2", "N1"),  # Ring N-H
        AtomPair("C2", "N2"),  # Exocyclic
        AtomPair("N7", "C8"),  # Weak C-H
        AtomPair("C2'", "O2'"),  # Hydroxyl (RNA only)
    ],
    "C": [
        AtomPair("C4", "N4"),  # Exocyclic
        AtomPair("C4", "C5"),  # Weak C-H
        AtomPair("N1", "C6"),  # Weak C-H
        AtomPair("C2'", "O2'"),  # Hydroxyl (RNA only)
    ],
    "U": [
        AtomPair("C2", "N3"),  # Ring N-H
        AtomPair("C4", "C5"),  # Weak C-H
        AtomPair("N1", "C6"),  # Weak C-H
        AtomPair("C2'", "O2'"),  # Hydroxyl
    ],
    "T": [
        AtomPair("C2", "N3"),  # Ring N-H
        AtomPair("N1", "C6"),  # Weak C-H
        AtomPair("C5", "C7"),  # Methyl Group (Very weak/Variable)
    ],
}

BASE_EDGES = {
    "A": {
        "Watson-Crick": ["N1", "C2", "N6"],
        "Hoogsteen": ["N6", "N7", "C8"],
        "Sugar": ["C2", "N3", "O2'"],
    },
    "G": {
        "Watson-Crick": ["N1", "N2", "O6"],
        "Hoogsteen": ["O6", "N7", "C8"],
        "Sugar": ["N2", "N3", "O2'"],
    },
    "C": {
        "Watson-Crick": ["O2", "N4", "N3"],
        "Hoogsteen": ["N4", "C5", "C6"],
        "Sugar": ["O2", "O2'"],
    },
    "U": {
        "Watson-Crick": ["O2", "N3", "O4"],
        "Hoogsteen": ["O4", "C5", "C6"],
        "Sugar": ["O2", "O2'"],
    },
    "T": {
        "Watson-Crick": ["O2", "N3", "O4"],
        "Hoogsteen": ["O4", "C6", "C7"],
        "Sugar": ["O2"],
    },
}


def read_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    """
    Reads configuration parameters from a file. If config_path is None,
    the default configuration file is loaded.
    """
    config = configparser.ConfigParser()

    if config_path is None:
        # Load default configuration from package resources
        try:
            with importlib.resources.path("rnapolis", "default_config.ini") as path:
                config.read(path)
        except FileNotFoundError:
            # Fallback if path context manager fails (e.g., in zip file)
            # Read content directly
            content = importlib.resources.read_text("rnapolis", "default_config.ini")
            config.read_string(content)
    else:
        config.read(config_path)

    return config


def get_c1p_c1p_distance(res1: Residue, res2: Residue) -> Optional[float]:
    """
    Calculates the distance between the C1' atoms of two residues.
    """
    c1p_1 = res1.find_atom("C1'")
    c1p_2 = res2.find_atom("C1'")

    if c1p_1 is None or c1p_2 is None:
        return None

    dist = np.linalg.norm(c1p_1.coordinates - c1p_2.coordinates).item()
    return dist


def is_c1p_c1p_distance_valid(
    distance: Optional[float], config: configparser.ConfigParser
) -> bool:
    """
    Checks if the C1'-C1' distance falls within the configured thresholds.
    """
    if distance is None:
        return False

    section = config["general"]

    # Helper to safely get float from config
    def get_float(key: str, fallback: float) -> float:
        return section.getfloat(key, fallback=fallback)

    min_dist = get_float("c1p_c1p_distance_min", 8.0)
    max_dist = get_float("c1p_c1p_distance_max", 15.0)

    return min_dist <= distance <= max_dist


def _get_base_atoms_coords(residue: Residue) -> Optional[np.ndarray]:
    """Extracts coordinates of core base atoms for a residue.

    Uses `Residue.one_letter_name` so modified residues (e.g. PSU/2MG)
    can still be treated as a canonical base when possible.
    """
    res_name = residue.one_letter_name.upper()
    is_purine = res_name in {"A", "G"}
    is_pyrimidine = res_name in {"C", "U", "T"}

    if not (is_purine or is_pyrimidine):
        return None

    target_atoms = PURINE_CORE_ATOMS if is_purine else PYRIMIDINE_CORE_ATOMS

    coords_list: List[np.ndarray] = []
    for atom_name in target_atoms:
        atom = residue.find_atom(atom_name)
        if atom is not None:
            coords_list.append(atom.coordinates)

    if not coords_list:
        return None

    return np.array(coords_list)


def _calculate_mean_plane(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the centroid and the best-fit plane normal vector using SVD.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Centroid, Normal Vector)
    """
    # 1. Calculate Centroid
    centroid: np.ndarray = np.mean(coords, axis=0)

    # 2. Center coordinates
    centered_coords = coords - centroid

    # 3. Compute Normal Vector using SVD
    # SVD of the centered coordinates matrix A = U * S * Vh
    # The normal vector is the last row of Vh (or last column of V)
    _, _, vh = np.linalg.svd(centered_coords)
    normal = vh[-1, :]

    # Ensure normal is a unit vector (SVD guarantees this)
    return centroid, normal


def _unify_normal(residue: Residue, normal: np.ndarray) -> Optional[np.ndarray]:
    """Ensures the normal vector points in the canonical direction.

    Uses `Residue.one_letter_name` so modified residues can be handled.
    """
    res_name = residue.one_letter_name.upper()
    is_purine = res_name in {"A", "G"}
    is_pyrimidine = res_name in {"C", "U", "T"}

    if not (is_purine or is_pyrimidine):
        return None

    if is_purine:
        a1_name, a2_name, a3_name = PURINE_FACE_ATOMS
    else:
        a1_name, a2_name, a3_name = PYRIMIDINE_FACE_ATOMS

    a1 = residue.find_atom(a1_name)
    a2 = residue.find_atom(a2_name)
    a3 = residue.find_atom(a3_name)

    if a1 is None or a2 is None or a3 is None:
        # Cannot determine canonical face orientation
        return normal

    # Calculate the canonical normal using cross product (Right Hand Rule)
    # Vector 1: a1 -> a2
    v1 = a2.coordinates - a1.coordinates
    # Vector 2: a1 -> a3
    v2 = a3.coordinates - a1.coordinates

    canonical_normal = np.cross(v1, v2)

    # Normalize the canonical normal
    norm_canonical = np.linalg.norm(canonical_normal)
    if norm_canonical < 1e-6:
        return normal  # Cannot normalize, return original SVD normal

    canonical_normal /= norm_canonical

    # Check if the SVD normal aligns with the canonical normal
    if np.dot(normal, canonical_normal) < 0:
        # Flip the SVD normal if it points the wrong way
        return -normal

    return normal


def _calculate_inter_planar_distance(
    c1: np.ndarray, n1: np.ndarray, c2: np.ndarray
) -> float:
    """
    Calculates the vertical separation distance between the centroid of base 2 (C2)
    and the plane defined by base 1 (C1, N1).

    Distance = |(C2 - C1) . N1|
    """
    vector_c1_c2 = c2 - c1
    # Project the vector onto the normal and take the magnitude
    distance = np.abs(np.dot(vector_c1_c2, n1))
    return distance


def _calculate_dihedral_angle(n1: np.ndarray, n2: np.ndarray) -> float:
    """
    Calculates the angle between two normal vectors in degrees.
    Angle = arccos(|n1 . n2|)
    """
    # Dot product of unit vectors is the cosine of the angle
    dot_product = np.dot(n1, n2)

    # We only care about the acute angle between the planes, so use absolute value
    # Clip to [-1, 1] to avoid floating point errors outside arccos domain
    cos_angle = np.clip(np.abs(dot_product), -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def _get_glycosidic_atom(residue: Residue) -> Optional[Atom]:
    """
    Returns the glycosidic bond atom (N1 for pyrimidines, N9 for purines).
    """
    res_name = residue.one_letter_name.upper()
    is_purine = res_name in {"A", "G"}
    is_pyrimidine = res_name in {"C", "U", "T"}

    if is_purine:
        return residue.find_atom("N9")
    elif is_pyrimidine:
        return residue.find_atom("N1")
    return None


def _calculate_lateral_displacement(
    c1: np.ndarray, n1: np.ndarray, c2: np.ndarray
) -> float:
    """
    Calculates the horizontal offset (shift/slide) between the two centroids
    relative to the plane of base 1.
    """
    vector_c1_c2 = c2 - c1

    # 1. Calculate the vertical component (projection of C1C2 onto N1)
    # Projection vector = (C1C2 . N1) * N1
    vertical_magnitude = np.dot(vector_c1_c2, n1)
    vertical_vector = vertical_magnitude * n1

    # 2. Calculate the horizontal component
    horizontal_vector = vector_c1_c2 - vertical_vector

    # 3. Lateral displacement is the magnitude of the horizontal vector
    displacement = np.linalg.norm(horizontal_vector)
    return float(displacement)


def _calculate_overlap_area(
    coords1: np.ndarray, n1: np.ndarray, c1: np.ndarray, coords2: np.ndarray
) -> float:
    """
    Calculates the overlap area by projecting both bases onto the plane of base 1.

    NOTE: This requires a robust 2D polygon intersection library like shapely.
    Since shapely is not guaranteed, this function only implements the projection
    and returns 0.0, serving as a placeholder.
    """
    # 1. Define a 2D coordinate system on the plane of Base 1
    # We need two orthogonal unit vectors (u, v) perpendicular to n1

    # Find a vector 'a' not parallel to n1
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, n1)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    # u is perpendicular to n1 and a
    u = np.cross(n1, a)
    u /= np.linalg.norm(u)

    # v is perpendicular to n1 and u
    v = np.cross(n1, u)
    v /= np.linalg.norm(v)

    # 2. Projection function: P_2D = [(P - C1) . u, (P - C1) . v]
    def project_to_2d(coords: np.ndarray) -> np.ndarray:
        coords_centered = coords - c1
        x_coords = np.dot(coords_centered, u)
        y_coords = np.dot(coords_centered, v)
        return np.stack([x_coords, y_coords], axis=1)

    # Project both sets of coordinates
    coords1_2d = project_to_2d(coords1)
    coords2_2d = project_to_2d(coords2)

    # 3. Calculate Convex Hull (or use all points if base is planar)
    # For simplicity and robustness, we use the convex hull of the base atoms
    # to define the polygon boundary.

    try:
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon
    except ImportError:
        # If shapely/scipy is not available, we cannot calculate overlap area robustly.
        return 0.0

    # Calculate Convex Hull for both projected sets
    hull1 = ConvexHull(coords1_2d)
    hull2 = ConvexHull(coords2_2d)

    # Create Polygons from the hull vertices
    polygon1 = Polygon(coords1_2d[hull1.vertices])
    polygon2 = Polygon(coords2_2d[hull2.vertices])

    # Calculate intersection area
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0

    intersection = polygon1.intersection(polygon2)

    return intersection.area


def get_stacking_parameters(
    res1: Residue, res2: Residue
) -> Dict[str, Union[float, str]]:
    """
    Calculates stacking parameters between two nucleotide residues.

    Parameters:
    -----------
    res1 : Residue
        The first residue (upstream/5')
    res2 : Residue
        The second residue (downstream/3')

    Returns:
    --------
    Dict[str, Union[float, str]]
        Dictionary containing calculated stacking parameters:
        - 'inter_planar_distance': Vertical separation (Angstroms)
        - 'dihedral_angle': Angle between planes (Degrees)
        - 'lateral_displacement': Horizontal offset (Angstroms)
        - 'overlap_area': Area of overlap (Angstroms^2) - Requires shapely/scipy
        - 'face_orientation': 'Same' or 'Opposite'
    """
    coords1 = _get_base_atoms_coords(res1)
    coords2 = _get_base_atoms_coords(res2)

    if coords1 is None or coords2 is None:
        return {
            "error": "One or both residues are not recognized nucleotides or lack core atoms."
        }

    # 1. Calculate Mean Plane and Centroid
    c1, n1_svd = _calculate_mean_plane(coords1)
    c2, n2_svd = _calculate_mean_plane(coords2)

    # 2. Unify Normal Vectors (ensure they point canonically 'up')
    n1 = _unify_normal(res1, n1_svd)
    n2 = _unify_normal(res2, n2_svd)

    if n1 is None or n2 is None:
        return {
            "error": "Could not determine canonical face orientation for one or both residues."
        }

    # 3. Inter-planar Distance (using plane of res1)
    inter_planar_distance = _calculate_inter_planar_distance(c1, n1, c2)

    # 4. Dihedral Angle
    dihedral_angle = _calculate_dihedral_angle(n1, n2)

    # 5. Lateral Displacement
    lateral_displacement = _calculate_lateral_displacement(c1, n1, c2)

    # 6. Overlap Area (Requires shapely/scipy)
    overlap_area = _calculate_overlap_area(coords1, n1, c1, coords2)

    # 7. Face Orientation
    # Check if the unified normals point in the same direction
    dot_product_unified = np.dot(n1, n2)
    face_orientation = "Same" if dot_product_unified > 0 else "Opposite"

    return {
        "inter_planar_distance": inter_planar_distance,
        "dihedral_angle": dihedral_angle,
        "lateral_displacement": lateral_displacement,
        "overlap_area": overlap_area,
        "face_orientation": face_orientation,
    }


def are_bases_coplanar(
    residues: Iterable[Residue],
    *,
    max_distance: float = 1.0,
    max_angle_deg: float = 30.0,
) -> bool:
    """Check if nucleobases of residues are co-planar.

    The criterion is a *single* global best-fit plane computed from the core
    ring atoms of all residues.

    A residue's nucleobase is considered compatible if:
    - its base centroid lies close to the global plane (`max_distance`)
    - the residue's own base plane normal is close to the global normal
      (`max_angle_deg`)

    Notes:
    - If `len(residues) < 2`, returns `True`.
    - Raises `ValueError` if any residue is not a recognizable nucleotide or
      lacks base core atoms.

    Parameters
    ----------
    residues:
        Iterable of `rnapolis.tertiary_v2.Residue`.
    max_distance:
        Maximum allowed atom-to-plane distance in Angstroms.
    max_angle_deg:
        Maximum allowed angle between a residue base normal and the global
        normal, in degrees.

    Returns
    -------
    bool
        True if all residues satisfy both criteria.
    """

    residues_list = list(residues)
    if len(residues_list) < 2:
        return True

    per_res_coords: List[np.ndarray] = []
    all_coords: List[np.ndarray] = []

    for residue in residues_list:
        coords = _get_base_atoms_coords(residue)
        if coords is None or coords.shape[0] < 3:
            raise ValueError(
                f"Cannot compute base plane for residue {residue}; missing core atoms"
            )
        per_res_coords.append(coords)
        all_coords.append(coords)

    stacked_coords = np.vstack(all_coords)
    centroid, global_normal = _calculate_mean_plane(stacked_coords)

    # Ensure a stable reference direction for angle checks.
    ref_centroid, ref_normal = _calculate_mean_plane(per_res_coords[0])
    ref_normal = _unify_normal(residues_list[0], ref_normal)
    if ref_normal is None:
        raise ValueError(
            f"Cannot determine canonical base normal for residue {residues_list[0]}"
        )
    if np.dot(global_normal, ref_normal) < 0:
        global_normal = -global_normal

    # 1) Distance-to-plane check using per-residue base centroids.
    # Using centroids is more robust to small local puckers / atom naming
    # differences, and matches typical "base plane" usage.
    base_centroids = np.array([np.mean(coords, axis=0) for coords in per_res_coords])
    centroid_dists = np.abs((base_centroids - centroid) @ global_normal)
    if float(np.max(centroid_dists)) > max_distance:
        return False

    # 2) Per-residue normal alignment check.
    for residue, coords in zip(residues_list, per_res_coords):
        _, n_svd = _calculate_mean_plane(coords)
        n = _unify_normal(residue, n_svd)
        if n is None:
            raise ValueError(
                f"Cannot determine canonical base normal for residue {residue}"
            )

        cos_angle = np.clip(np.abs(float(np.dot(n, global_normal))), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg > max_angle_deg:
            return False

    return True


def is_stacking_valid(
    stacking_params: Dict[str, Union[float, str]], config: configparser.ConfigParser
) -> bool:
    """
    Checks if the calculated stacking parameters fall within the configured thresholds.
    """
    if "error" in stacking_params:
        return False

    section = config["stacking"]

    # Helper to safely get float from config
    def get_float(key: str, fallback: float) -> float:
        return section.getfloat(key, fallback=fallback)

    # Check inter_planar_distance
    dist = stacking_params["inter_planar_distance"]
    assert isinstance(dist, float)
    if not (
        get_float("inter_planar_distance_min", 3.0)
        <= dist
        <= get_float("inter_planar_distance_max", 4.5)
    ):
        return False

    # Check dihedral_angle
    angle = stacking_params["dihedral_angle"]
    assert isinstance(angle, float)
    if not (
        get_float("dihedral_angle_min", 0.0)
        <= angle
        <= get_float("dihedral_angle_max", 30.0)
    ):
        return False

    # Check lateral_displacement
    disp = stacking_params["lateral_displacement"]
    assert isinstance(disp, float)
    if not (
        get_float("lateral_displacement_min", 0.0)
        <= disp
        <= get_float("lateral_displacement_max", 3.0)
    ):
        return False

    # Check overlap_area
    area = stacking_params["overlap_area"]
    assert isinstance(area, float)
    if area > 0.0:  # Only validate if calculation succeeded
        if not (
            get_float("overlap_area_min", 5.0)
            <= area
            <= get_float("overlap_area_max", 100.0)
        ):
            return False

    return True


# Helper function to find all potential donor triplets (antecedent, donor)
def _get_potential_donors(residue: Residue) -> List[Tuple[Atom, Atom]]:
    potential_donors: List[Tuple[Atom, Atom]] = []
    res_name = residue.one_letter_name.upper()

    # Check base donors
    if res_name in DONORS:
        for pair in DONORS[res_name]:
            antecedent = residue.find_atom(pair.antecedent)
            donor = residue.find_atom(pair.donor)
            if antecedent is not None and donor is not None:
                potential_donors.append((antecedent, donor))

    return potential_donors


# Helper function to find all potential acceptor atoms
def _get_potential_acceptors(residue: Residue) -> List[Atom]:
    potential_acceptors: List[Atom] = []
    res_name = residue.one_letter_name.upper()

    # Check base acceptors
    if res_name in BASE_ACCEPTORS:
        for name in BASE_ACCEPTORS[res_name]:
            atom = residue.find_atom(name)
            if atom is not None:
                potential_acceptors.append(atom)

    # Check phosphate acceptors (P, OP1, OP2)
    for name in PHOSPHATE_ACCEPTORS:
        atom = residue.find_atom(name)
        if atom is not None:
            potential_acceptors.append(atom)

    # Check sugar acceptors (O2', O4')
    for name in SUGAR_ACCEPTORS:
        atom = residue.find_atom(name)
        if atom is not None:
            # Avoid double counting O2' if it was already added as a base acceptor
            if atom not in potential_acceptors:
                potential_acceptors.append(atom)

    return potential_acceptors


def find_hbond_pairs(res1: Residue, res2: Residue) -> List[Tuple[Atom, Atom, Atom]]:
    """
    Finds all possible hydrogen bond donor-acceptor pairs between two residues.

    The search is performed in both directions (res1 -> res2 and res2 -> res1).
    It uses the static definitions (DONORS, BASE_ACCEPTORS, etc.) and the
    one_letter_name property of the residues.

    Parameters:
    -----------
    res1 : Residue
        The first residue.
    res2 : Residue
        The second residue.

    Returns:
    --------
    List[Tuple[Atom, Atom, Atom]]
        A list of potential hydrogen bond triplets: (antecedent, donor, acceptor).
    """
    hbond_pairs: List[Tuple[Atom, Atom, Atom]] = []

    # --- Search Direction 1: res1 (Donor) -> res2 (Acceptor) ---
    donors1 = _get_potential_donors(res1)
    acceptors2 = _get_potential_acceptors(res2)

    for antecedent, donor in donors1:
        for acceptor in acceptors2:
            hbond_pairs.append((antecedent, donor, acceptor))

    # --- Search Direction 2: res2 (Donor) -> res1 (Acceptor) ---
    donors2 = _get_potential_donors(res2)
    acceptors1 = _get_potential_acceptors(res1)

    for antecedent, donor in donors2:
        for acceptor in acceptors1:
            hbond_pairs.append((antecedent, donor, acceptor))

    return hbond_pairs


def get_hbond_parameters(
    antecedent: Atom, donor: Atom, acceptor: Atom
) -> Dict[str, float]:
    """
    Calculates geometric parameters for a potential hydrogen bond involving three atoms.

    Parameters:
    -----------
    antecedent : Atom
        The atom covalently bonded to the donor (e.g., C in C-H...O).
    donor : Atom
        The hydrogen bond donor atom (e.g., H in N-H...O, or N/O if H is implicit).
    acceptor : Atom
        The hydrogen bond acceptor atom (e.g., O in N-H...O).

    Returns:
    --------
    Dict[str, float]
        Dictionary containing:
        - 'antecedent_donor_distance': Distance between antecedent and donor (Angstroms).
        - 'donor_acceptor_distance': Distance between donor and acceptor (Angstroms).
        - 'antecedent_donor_acceptor_angle': Angle A-D-A (Degrees).
    """
    # Get coordinates
    coord_a = antecedent.coordinates
    coord_d = donor.coordinates
    coord_c = acceptor.coordinates

    # 1. Antecedent-Donor distance (A-D)
    dist_ad = np.linalg.norm(coord_a - coord_d).item()

    # 2. Donor-Acceptor distance (D-C)
    dist_dc = np.linalg.norm(coord_d - coord_c).item()

    # 3. Antecedent-Donor-Acceptor angle (A-D-C)
    # Vectors DA and DC
    vec_da = coord_a - coord_d
    vec_dc = coord_c - coord_d

    # Calculate angle using dot product formula: cos(theta) = (v1 . v2) / (|v1| * |v2|)
    dot_product = np.dot(vec_da, vec_dc)
    norm_da = np.linalg.norm(vec_da)
    norm_dc = np.linalg.norm(vec_dc)

    if norm_da == 0 or norm_dc == 0:
        # Should not happen with valid Atom objects, but as a safeguard
        angle_degrees = np.nan
    else:
        cos_theta = dot_product / (norm_da * norm_dc)
        # Clip to [-1, 1] to avoid floating point errors outside arccos domain
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_radians = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle_radians)

    return {
        "antecedent_donor_distance": dist_ad,
        "donor_acceptor_distance": dist_dc,
        "antecedent_donor_acceptor_angle": angle_degrees,
    }


def is_hbond_valid(
    hbond_params: Dict[str, float], config: configparser.ConfigParser
) -> bool:
    """
    Checks if the calculated hydrogen bond parameters fall within the configured thresholds.
    """
    section = config["hbonds"]

    # Helper to safely get float from config
    def get_float(key: str, fallback: float) -> float:
        return section.getfloat(key, fallback=fallback)

    # Check donor_acceptor_distance
    dist = hbond_params["donor_acceptor_distance"]
    assert isinstance(dist, float)
    if not (
        get_float("donor_acceptor_distance_min", 2.5)
        <= dist
        <= get_float("donor_acceptor_distance_max", 3.5)
    ):
        return False

    # Check antecedent_donor_acceptor_angle
    angle = hbond_params["antecedent_donor_acceptor_angle"]
    assert isinstance(angle, float)
    if not (
        get_float("antecedent_donor_acceptor_angle_min", 100.0)
        <= angle
        <= get_float("antecedent_donor_acceptor_angle_max", 180.0)
    ):
        return False

    return True


def _get_edges_for_atom(residue: Residue, atom_name: str) -> List[str]:
    """
    Determines which base edges (WC, Hoogsteen, Sugar) an atom belongs to.
    """
    res_name = residue.one_letter_name.upper()
    if res_name not in BASE_EDGES:
        return []

    edges: List[str] = []
    for edge, atoms in BASE_EDGES[res_name].items():
        if atom_name in atoms:
            edges.append(edge)
    return edges


def get_hbond_edge_counts(
    res1: Residue, res2: Residue, config: configparser.ConfigParser
) -> Dict[Residue, Counter[str]]:
    """
    Finds valid hydrogen bonds between two residues and returns a dictionary
    containing the counts of involved base edges for each residue.

    Parameters:
    -----------
    res1 : Residue
        The first residue.
    res2 : Residue
        The second residue.
    config : configparser.ConfigParser
        Configuration object containing HBond validation thresholds.

    Returns:
    --------
    Dict[Residue, Counter]
        A dictionary mapping each residue to a Counter of involved base edges.
    """
    all_pairs = find_hbond_pairs(res1, res2)
    valid_hbonds: List[Tuple[Atom, Atom, Atom]] = []

    for antecedent, donor, acceptor in all_pairs:
        hbond_params = get_hbond_parameters(antecedent, donor, acceptor)
        if is_hbond_valid(hbond_params, config):
            valid_hbonds.append((antecedent, donor, acceptor))

    edge_counts: Dict[Residue, Counter[str]] = {res1: Counter(), res2: Counter()}

    for antecedent, donor, acceptor in valid_hbonds:
        # Determine which residue is the donor and which is the acceptor
        # The donor atom (donor) and acceptor atom (acceptor) belong to different residues.

        # Check if donor is in res1 (meaning acceptor is in res2)
        if donor.data.name in res1.atoms["name"].values:
            donor_res = res1
            acceptor_res = res2
            donor_atom = donor
            acceptor_atom = acceptor
        # Check if donor is in res2 (meaning acceptor is in res1)
        elif donor.data.name in res2.atoms["name"].values:
            donor_res = res2
            acceptor_res = res1
            donor_atom = donor
            acceptor_atom = acceptor
        else:
            # Should not happen if find_hbond_pairs works correctly
            continue

        # Tally edges for donor residue
        donor_edges = _get_edges_for_atom(donor_res, donor_atom.name)
        for edge in donor_edges:
            edge_counts[donor_res][edge] += 1

        # Tally edges for acceptor residue
        acceptor_edges = _get_edges_for_atom(acceptor_res, acceptor_atom.name)
        for edge in acceptor_edges:
            edge_counts[acceptor_res][edge] += 1

    return edge_counts


def get_inter_base_parameters(
    res1: Residue, res2: Residue
) -> Dict[str, Union[float, str]]:
    """
    Calculates geometric parameters between two bases, including inter-base angle,
    planar displacement, and a specific torsion angle.

    Parameters:
    -----------
    res1 : Residue
        The first residue.
    res2 : Residue
        The second residue.

    Returns:
    --------
    Dict[str, Union[float, str]]
        Dictionary containing calculated parameters:
        - 'inter_base_angle': Angle between base planes (Degrees).
        - 'planar_displacement': Vertical separation (Angstroms).
        - 'torsion_angle_C1N_NC1': Torsion angle C1'(1)-N1/N9(1)-N1/N9(2)-C1'(2) (Degrees).
    """
    # 1. Calculate Mean Plane and Centroid for both residues
    coords1 = _get_base_atoms_coords(res1)
    coords2 = _get_base_atoms_coords(res2)

    if coords1 is None or coords2 is None:
        return {
            "error": "One or both residues are not recognized nucleotides or lack core atoms."
        }

    c1, n1_svd = _calculate_mean_plane(coords1)
    c2, n2_svd = _calculate_mean_plane(coords2)

    # 2. Unify Normal Vectors (ensure they point canonically 'up')
    n1 = _unify_normal(res1, n1_svd)
    n2 = _unify_normal(res2, n2_svd)

    if n1 is None or n2 is None:
        return {
            "error": "Could not determine canonical face orientation for one or both residues."
        }

    # --- 1. Inter-base angle (Dihedral Angle) ---
    inter_base_angle = _calculate_dihedral_angle(n1, n2)

    # --- 2. Planar Displacement (Vertical Separation) ---
    # Project the vector connecting centers onto the Normal of res1.
    planar_displacement = _calculate_inter_planar_distance(c1, n1, c2)

    # --- 3. Torsion Angle C1'(1)-N1/N9(1)-N1/N9(2)-C1'(2) ---
    c1_prime_1 = res1.find_atom("C1'")
    c1_prime_2 = res2.find_atom("C1'")
    n_glycosidic_1 = _get_glycosidic_atom(res1)
    n_glycosidic_2 = _get_glycosidic_atom(res2)

    torsion_angle_C1N_NC1 = np.nan

    if c1_prime_1 and n_glycosidic_1 and n_glycosidic_2 and c1_prime_2:
        a1 = c1_prime_1.coordinates
        a2 = n_glycosidic_1.coordinates
        a3 = n_glycosidic_2.coordinates
        a4 = c1_prime_2.coordinates

        # Calculate vectors between points
        v1 = a2 - a1
        v2 = a3 - a2
        v3 = a4 - a3

        # Calculate normal vectors
        n_1 = np.cross(v1, v2)
        n_2 = np.cross(v2, v3)

        # Normalize normal vectors
        n1_norm = np.linalg.norm(n_1)
        n2_norm = np.linalg.norm(n_2)

        if n1_norm > 1e-6 and n2_norm > 1e-6:
            n_1 = n_1 / n1_norm
            n_2 = n_2 / n2_norm

            # Calculate the angle using dot product
            m1 = np.cross(n_1, v2 / np.linalg.norm(v2))
            x = np.dot(n_1, n_2)
            y = np.dot(m1, n_2)

            # Return angle in degrees
            torsion_angle_C1N_NC1 = np.degrees(np.arctan2(y, x))

    return {
        "inter_base_angle": inter_base_angle,
        "planar_displacement": planar_displacement,
        "torsion_angle_C1N_NC1": torsion_angle_C1N_NC1,
    }


def is_basepair_valid(
    inter_base_params: Dict[str, Union[float, str]], config: configparser.ConfigParser
) -> bool:
    """
    Checks if the calculated inter-base parameters fall within the configured thresholds
    for a valid base pair.
    """
    if "error" in inter_base_params:
        return False

    section = config["basepair"]

    # Helper to safely get float from config
    def get_float(key: str, fallback: float) -> float:
        return section.getfloat(key, fallback=fallback)

    # Check inter_base_angle
    angle = inter_base_params["inter_base_angle"]
    assert isinstance(angle, float)
    if not (
        get_float("inter_base_angle_min", 0.0)
        <= angle
        <= get_float("inter_base_angle_max", 30.0)
    ):
        return False

    # Check planar_displacement
    disp = inter_base_params["planar_displacement"]
    assert isinstance(disp, float)
    if not (
        get_float("planar_displacement_min", 0.0)
        <= disp
        <= get_float("planar_displacement_max", 3.0)
    ):
        return False

    return True


def is_cis(
    inter_base_params: Dict[str, Union[float, str]], config: configparser.ConfigParser
) -> bool:
    """
    Checks if the torsion angle indicates a cis conformation based on config thresholds.
    """
    if "error" in inter_base_params:
        return False

    section = config["basepair"]

    # Helper to safely get float from config
    def get_float(key: str, fallback: float) -> float:
        return section.getfloat(key, fallback=fallback)

    # Check torsion_angle_C1N_NC1
    torsion = inter_base_params["torsion_angle_C1N_NC1"]
    assert isinstance(torsion, float)

    # Note: Torsion angle is in degrees, config values should be in degrees too.
    if not (
        get_float("torsion_angle_cis_min", -30.0)
        <= torsion
        <= get_float("torsion_angle_cis_max", 30.0)
    ):
        return False

    return True
