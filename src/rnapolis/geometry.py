import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Assuming Residue is importable from tertiary_v2
from .tertiary_v2 import Residue

# --- Atom Definitions (Redefined from tertiary_v2 for local use) ---
# Purines (A, G, DA, DG)
PURINE_CORE_ATOMS = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}
# Pyrimidines (C, T, U, DC, DT)
PYRIMIDINE_CORE_ATOMS = {"N1", "C2", "N3", "C4", "C5", "C6"}

# Atoms used for defining the canonical face orientation (Right Hand Rule)
# Purine: N9 -> C4 -> C5 (Cross product of N9-C4 and N9-C5 vectors)
PURINE_FACE_ATOMS = ("N9", "C4", "C5")
# Pyrimidine: N1 -> C6 -> C5 (Cross product of N1-C6 and N1-C5 vectors)
PYRIMIDINE_FACE_ATOMS = ("N1", "C6", "C5")


def _get_base_atoms_coords(residue: Residue) -> Optional[np.ndarray]:
    """
    Extracts coordinates of core base atoms for a given residue.
    """
    res_name = residue.residue_name
    is_purine = res_name in {"A", "G", "DA", "DG"}
    is_pyrimidine = res_name in {"C", "U", "T", "DC", "DT"}

    if not (is_purine or is_pyrimidine):
        return None

    target_atoms = PURINE_CORE_ATOMS if is_purine else PYRIMIDINE_CORE_ATOMS

    coords_list = []
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
    centroid = np.mean(coords, axis=0)

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
    """
    Ensures the normal vector points in the canonical direction (IUPAC standard).
    """
    res_name = residue.residue_name
    is_purine = res_name in {"A", "G", "DA", "DG"}
    is_pyrimidine = res_name in {"C", "U", "T", "DC", "DT"}

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
    return displacement


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
