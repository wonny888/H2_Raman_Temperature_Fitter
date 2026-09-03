from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re

import numpy as np
import pandas as pd

from raman_paths import LIBRARY_DIR


ProgressCallback = Callable[[float, str], None]

DEFAULT_CROP_START = 297.0
DEFAULT_CROP_END = 299.5
DEFAULT_SHIFT_MIN = -0.10
DEFAULT_SHIFT_MAX = 0.10
DEFAULT_SHIFT_STEP = 0.005
DEFAULT_BROADENING_MIN = 0.000
DEFAULT_BROADENING_MAX = 0.15
DEFAULT_BROADENING_STEP = 0.001
DEFAULT_EXCITATION_NM = 266.0
DEFAULT_GRID_START_NM = 296.0
DEFAULT_GRID_END_NM = 300.2
DEFAULT_GRID_STEP_NM = 0.01
ALLOWED_GRID_STEPS_NM = (0.01, 0.001, 0.0001)


@dataclass(frozen=True)
class AnalysisSettings:
    wavelength_column: int = 0
    intensity_column: int = 0
    crop_start: float = DEFAULT_CROP_START
    crop_end: float = DEFAULT_CROP_END
    fit_temp_step: int = 5
    allow_shift: bool = True
    auto_generate_missing: bool = True
    regenerate_library: bool = False
    generation_temp_min: int = 1600
    generation_temp_max: int = 3000
    generation_temp_step: int = 5
    excitation_nm: float = DEFAULT_EXCITATION_NM
    generation_grid_start_nm: float = DEFAULT_GRID_START_NM
    generation_grid_end_nm: float = DEFAULT_GRID_END_NM
    grid_step_nm: float = DEFAULT_GRID_STEP_NM


@dataclass
class AnalysisResult:
    phi: float
    reactants: dict[str, float]
    product_moles: dict[str, float]
    composition: dict[str, float]
    library_path: Path
    temps: np.ndarray
    simulation_wavelengths: np.ndarray
    experimental_wavelengths_raw: np.ndarray
    experimental_intensity_raw: np.ndarray
    theory_wavelengths: np.ndarray
    theory_intensity: np.ndarray
    experimental_wavelengths_crop: np.ndarray
    experimental_intensity_crop: np.ndarray
    best_temperature: int
    best_shift: float
    best_broadening: float
    best_error: float
    best_simulation: np.ndarray
    fit_results: pd.DataFrame


def _report(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def phi_to_tag(phi: float) -> str:
    text = f"{float(phi):g}"
    if "." not in text:
        text += ".0"
    return text.replace(".", "p")


def extract_phi_from_filename(filename: str) -> float | None:
    match = re.search(
        r"phi\s*[_-]?\s*([0-9]+(?:\.[0-9]+)?)",
        filename,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def calculate_ideal_h2_air_composition(phi: float):
    """
    Product mole fractions using a fixed stoichiometric-air basis:
        H2 entering = Phi
        O2 entering = 0.5
        N2 entering = 1.88
    """
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("Phi must be greater than zero.")

    # Fixed stoichiometric-air basis. The incoming H2 amount changes with Phi.
    h2_in = phi
    o2_in = 0.5
    n2_in = 1.88

    # The available oxygen can burn at most 1 mole H2 on this basis.
    h2_burned = min(h2_in, 2.0 * o2_in)

    # Ideal product mole amounts before normalization.
    h2_out = h2_in - h2_burned
    h2o_out = h2_burned
    o2_out = o2_in - 0.5 * h2_burned
    n2_out = n2_in

    total = h2_out + h2o_out + o2_out + n2_out
    if total <= 0:
        raise ValueError("Calculated product total is not positive.")

    reactants = {
        "H2": h2_in,
        "O2": o2_in,
        "N2": n2_in,
    }
    product_moles = {
        "H2": h2_out,
        "N2": n2_out,
        "H2O": h2o_out,
        "O2": o2_out,
    }
    mole_fractions = {
        name: amount / total for name, amount in product_moles.items()
    }
    mole_fractions.update({"CO": 0.0, "CO2": 0.0, "He": 0.0})

    # Eliminate tiny floating-point sum errors that can exceed exactly 1.0.
    main_sum = sum(mole_fractions[name] for name in ["H2", "N2", "H2O", "O2"])
    mole_fractions["N2"] += 1.0 - main_sum

    return reactants, product_moles, mole_fractions


def default_generation_range(phi: float) -> tuple[int, int]:
    phi = float(phi)
    if phi <= 1.5:
        return 1800, 3200
    if phi <= 2.0:
        return 1600, 3000
    if phi <= 2.5:
        return 1200, 2700
    if phi <= 3.0:
        return 900, 2200
    return 700, 2000


def _library_search_folders() -> list[Path]:
    """
    Return the one persistent folder used for generated libraries.

    Python source version:
        Python/LIF_Raman/generated_libraries

    Packaged Windows app:
        %LOCALAPPDATA%/H2_Raman_Temperature_Fitter/generated_libraries
    """
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return [LIBRARY_DIR]


def _npz_scalar(data, key: str, default: float | None = None) -> float | None:
    if key not in data.files:
        return default
    values = np.asarray(data[key], dtype=float).ravel()
    if values.size == 0:
        return default
    value = float(values[0])
    return value if np.isfinite(value) else default


def read_library_metadata(library_path: str | Path) -> dict[str, object]:
    """Read or infer the settings stored in one Raman NPZ library."""
    library_path = Path(library_path).resolve()

    with np.load(library_path, allow_pickle=False) as library:
        required = {"temps", "wavelengths", "spectra"}
        missing = required.difference(library.files)
        if missing:
            raise ValueError(
                f"{library_path.name} is missing: "
                + ", ".join(sorted(missing))
            )

        temps = np.asarray(library["temps"], dtype=float).ravel()
        wavelengths = np.asarray(library["wavelengths"], dtype=float).ravel()
        spectra_shape = tuple(np.asarray(library["spectra"]).shape)

        temps = np.unique(np.sort(temps[np.isfinite(temps)]))
        wavelengths = np.unique(
            np.sort(wavelengths[np.isfinite(wavelengths)])
        )

        if temps.size == 0:
            raise ValueError(f"{library_path.name} has no valid temperatures.")
        if wavelengths.size < 2:
            raise ValueError(
                f"{library_path.name} needs at least two wavelength points."
            )

        temp_differences = np.diff(temps)
        temp_differences = temp_differences[temp_differences > 1e-9]
        inferred_temp_step = (
            float(np.median(temp_differences))
            if temp_differences.size
            else float("nan")
        )

        wavelength_differences = np.diff(wavelengths)
        wavelength_differences = wavelength_differences[
            wavelength_differences > 1e-12
        ]
        inferred_grid_step = float(np.median(wavelength_differences))

        metadata: dict[str, object] = {
            "path": library_path,
            "filename": library_path.name,
            "phi": _npz_scalar(library, "phi"),
            "excitation_nm": _npz_scalar(
                library,
                "excitation_nm",
                DEFAULT_EXCITATION_NM,
            ),
            "grid_start_nm": _npz_scalar(
                library,
                "grid_start_nm",
                float(wavelengths.min()),
            ),
            "grid_end_nm": _npz_scalar(
                library,
                "grid_end_nm",
                float(wavelengths.max()),
            ),
            "grid_step_nm": _npz_scalar(
                library,
                "grid_step_nm",
                inferred_grid_step,
            ),
            "temperature_min_K": _npz_scalar(
                library,
                "temperature_min_K",
                float(temps.min()),
            ),
            "temperature_max_K": _npz_scalar(
                library,
                "temperature_max_K",
                float(temps.max()),
            ),
            "temperature_step_K": _npz_scalar(
                library,
                "temperature_step_K",
                inferred_temp_step,
            ),
            "temperature_count": int(temps.size),
            "wavelength_count": int(wavelengths.size),
            "spectra_shape": spectra_shape,
            "H2conc": _npz_scalar(library, "H2conc"),
            "N2conc": _npz_scalar(library, "N2conc"),
            "H2Oconc": _npz_scalar(library, "H2Oconc"),
            "O2conc": _npz_scalar(library, "O2conc"),
        }

    return metadata


def detect_library_temperature_step(library_path: str | Path) -> float:
    """Return the actual temperature spacing stored in an NPZ library."""
    value = float(read_library_metadata(library_path)["temperature_step_K"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Could not determine the library temperature step.")
    return value


def detect_library_grid_step(library_path: str | Path) -> float:
    """Return the actual wavelength-grid spacing stored in an NPZ library."""
    value = float(read_library_metadata(library_path)["grid_step_nm"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Could not determine the library wavelength step.")
    return value


def _library_matches_composition(
    metadata: dict[str, object],
    composition: dict[str, float] | None,
) -> bool:
    if composition is None:
        return True

    for species, key in [
        ("H2", "H2conc"),
        ("N2", "N2conc"),
        ("H2O", "H2Oconc"),
        ("O2", "O2conc"),
    ]:
        stored = metadata.get(key)
        if stored is None:
            # A library without composition metadata is not considered a safe
            # match. It will be regenerated once in the newer format.
            return False
        if not np.isclose(
            float(stored),
            float(composition[species]),
            rtol=0.0,
            atol=1e-10,
        ):
            return False
    return True


def find_phi_library(
    phi: float,
    base_dir: str | Path,
    *,
    excitation_nm: float = DEFAULT_EXCITATION_NM,
    grid_start_nm: float = DEFAULT_GRID_START_NM,
    grid_end_nm: float = DEFAULT_GRID_END_NM,
    grid_step_nm: float = DEFAULT_GRID_STEP_NM,
    requested_temp_min: float | None = None,
    requested_temp_max: float | None = None,
    requested_temp_step: float | None = None,
    composition: dict[str, float] | None = None,
) -> Path | None:
    """Find the best library matching Phi and the requested generation grid."""
    tag = phi_to_tag(phi)
    decimal_tag = f"{float(phi):g}"

    patterns = [
        f"*phi_{tag}*.npz",
        f"*phi{tag}*.npz",
        f"*Phi_{tag}*.npz",
        f"*Phi{tag}*.npz",
        f"*phi_{decimal_tag}*.npz",
        f"*phi{decimal_tag}*.npz",
    ]

    matches: list[Path] = []
    for folder in _library_search_folders():
        if folder.exists():
            for pattern in patterns:
                matches.extend(folder.glob(pattern))

    if not matches and LIBRARY_DIR.exists():
        for pattern in patterns:
            matches.extend(LIBRARY_DIR.rglob(pattern))

    candidates = list(
        {path.resolve(): path.resolve() for path in matches}.values()
    )

    matching: list[tuple[Path, dict[str, object]]] = []
    for path in candidates:
        try:
            metadata = read_library_metadata(path)
        except Exception:
            continue

        stored_phi = metadata.get("phi")
        if stored_phi is not None and not np.isclose(
            float(stored_phi), float(phi), rtol=0.0, atol=1e-10
        ):
            continue

        if not np.isclose(
            float(metadata["excitation_nm"]),
            float(excitation_nm),
            rtol=0.0,
            atol=1e-9,
        ):
            continue

        stored_grid_start = float(metadata["grid_start_nm"])
        stored_grid_end = float(metadata["grid_end_nm"])
        stored_grid_step = float(metadata["grid_step_nm"])
        requested_grid_start = float(grid_start_nm)
        requested_grid_end = float(grid_end_nm)
        requested_grid_step = float(grid_step_nm)

        # A broader stored wavelength range can satisfy a narrower request.
        if stored_grid_start > requested_grid_start + 1e-8:
            continue
        if stored_grid_end < requested_grid_end - 1e-8:
            continue

        # A finer stored grid can satisfy a coarser request.
        # Example: a 0.0001 nm library can be reused for a 0.001 nm run.
        if stored_grid_step > requested_grid_step + 1e-12:
            continue

        stored_temp_min = float(metadata["temperature_min_K"])
        stored_temp_max = float(metadata["temperature_max_K"])

        if (
            requested_temp_min is not None
            and stored_temp_min > float(requested_temp_min) + 1e-9
        ):
            continue

        if (
            requested_temp_max is not None
            and stored_temp_max < float(requested_temp_max) - 1e-9
        ):
            continue

        if not _library_matches_composition(metadata, composition):
            continue

        matching.append((path, metadata))

    if not matching:
        return None

    def score(item: tuple[Path, dict[str, object]]):
        path, metadata = item
        temp_step = float(metadata["temperature_step_K"])
        stored_min = float(metadata["temperature_min_K"])
        stored_max = float(metadata["temperature_max_K"])
        stored_grid_start = float(metadata["grid_start_nm"])
        stored_grid_end = float(metadata["grid_end_nm"])
        stored_grid_step = float(metadata["grid_step_nm"])

        requested_grid_start = float(grid_start_nm)
        requested_grid_end = float(grid_end_nm)
        requested_grid_step = float(grid_step_nm)

        grid_exact = np.isclose(
            stored_grid_step,
            requested_grid_step,
            rtol=0.0,
            atol=1e-12,
        )
        wavelength_range_exact = (
            np.isclose(
                stored_grid_start,
                requested_grid_start,
                rtol=0.0,
                atol=1e-8,
            )
            and np.isclose(
                stored_grid_end,
                requested_grid_end,
                rtol=0.0,
                atol=1e-8,
            )
        )
        extra_wavelength_span = max(
            0.0, requested_grid_start - stored_grid_start
        ) + max(0.0, stored_grid_end - requested_grid_end)

        requested_step = (
            None if requested_temp_step is None else float(requested_temp_step)
        )
        step_fine_enough = (
            True
            if requested_step is None
            else temp_step <= requested_step + 1e-9
        )
        step_exact = (
            True
            if requested_step is None
            else np.isclose(temp_step, requested_step, rtol=0.0, atol=1e-9)
        )

        range_exact = True
        extra_span = 0.0
        if requested_temp_min is not None:
            requested_min = float(requested_temp_min)
            range_exact = range_exact and np.isclose(
                stored_min, requested_min, rtol=0.0, atol=1e-9
            )
            extra_span += max(0.0, requested_min - stored_min)
        if requested_temp_max is not None:
            requested_max = float(requested_temp_max)
            range_exact = range_exact and np.isclose(
                stored_max, requested_max, rtol=0.0, atol=1e-9
            )
            extra_span += max(0.0, stored_max - requested_max)

        return (
            1 if step_fine_enough else 0,
            # Prefer an exact requested wavelength grid when available.
            1 if grid_exact else 0,
            # Otherwise prefer the coarsest acceptable stored grid.
            stored_grid_step,
            # Prefer an exact wavelength range, then the least extra span.
            1 if wavelength_range_exact else 0,
            -extra_wavelength_span,
            1 if step_exact else 0,
            1 if range_exact else 0,
            -extra_span,
            path.stat().st_mtime,
        )

    return max(matching, key=score)[0]


def _validate_library_arrays(temps, wavelengths, spectra):
    temps = np.asarray(temps, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    spectra = np.asarray(spectra, dtype=float)

    if spectra.shape == (len(wavelengths), len(temps)):
        spectra = spectra.T

    if spectra.shape != (len(temps), len(wavelengths)):
        raise ValueError(
            "Library dimensions do not match: "
            f"temps={temps.shape}, wavelengths={wavelengths.shape}, "
            f"spectra={spectra.shape}."
        )

    temp_order = np.argsort(temps)
    temps = temps[temp_order]
    spectra = spectra[temp_order]

    wave_order = np.argsort(wavelengths)
    wavelengths = wavelengths[wave_order]
    spectra = spectra[:, wave_order]

    return temps, wavelengths, spectra


def load_library(path: str | Path):
    path = Path(path)
    with np.load(path) as data:
        required = {"temps", "wavelengths", "spectra"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("Library is missing: " + ", ".join(sorted(missing)))
        return _validate_library_arrays(
            data["temps"],
            data["wavelengths"],
            data["spectra"],
        )


def resample_library_wavelength_grid(
    wavelengths,
    spectra,
    *,
    target_start_nm: float,
    target_end_nm: float,
    target_step_nm: float,
):
    """
    Prepare a loaded library for the requested analysis wavelength grid.

    A finer and/or broader saved library is reused in memory. The original
    NPZ file is not changed and no second library file is created.
    """
    wavelengths = np.asarray(wavelengths, dtype=float).ravel()
    spectra = np.asarray(spectra, dtype=float)

    target_start_nm = float(target_start_nm)
    target_end_nm = float(target_end_nm)
    target_step_nm = float(target_step_nm)

    if target_step_nm <= 0:
        raise ValueError("Target wavelength step must be greater than zero.")
    if target_end_nm <= target_start_nm:
        raise ValueError(
            "Target wavelength end must be greater than its start."
        )
    if wavelengths.size < 2:
        raise ValueError("Library needs at least two wavelength points.")
    if spectra.ndim != 2 or spectra.shape[1] != wavelengths.size:
        raise ValueError(
            "Library spectra do not match the wavelength dimension."
        )

    source_step_nm = float(np.median(np.diff(wavelengths)))
    if not np.isfinite(source_step_nm) or source_step_nm <= 0:
        raise ValueError("Could not determine the stored wavelength step.")

    if source_step_nm > target_step_nm + 1e-12:
        raise ValueError(
            f"Stored wavelength step {source_step_nm:g} nm is coarser than "
            f"the requested {target_step_nm:g} nm step."
        )

    if wavelengths[0] > target_start_nm + 1e-8:
        raise ValueError("Stored library does not reach the requested start.")
    if wavelengths[-1] < target_end_nm - 1e-8:
        raise ValueError("Stored library does not reach the requested end.")

    same_grid = (
        np.isclose(source_step_nm, target_step_nm, rtol=0.0, atol=1e-12)
        and np.isclose(
            wavelengths[0], target_start_nm, rtol=0.0, atol=1e-8
        )
        and np.isclose(
            wavelengths[-1], target_end_nm, rtol=0.0, atol=1e-8
        )
    )
    if same_grid:
        return wavelengths, spectra

    target_wavelengths = np.arange(
        target_start_nm,
        target_end_nm + 0.5 * target_step_nm,
        target_step_nm,
        dtype=float,
    )
    target_wavelengths = target_wavelengths[
        target_wavelengths <= target_end_nm + 1e-10
    ]

    resampled_spectra = np.empty(
        (spectra.shape[0], target_wavelengths.size),
        dtype=float,
    )
    for row_index, spectrum in enumerate(spectra):
        resampled_spectra[row_index] = np.interp(
            target_wavelengths,
            wavelengths,
            spectrum,
        )

    return target_wavelengths, resampled_spectra

def read_spectrum_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    # Detect the real file type from its contents instead of trusting
    # only the filename extension.
    with path.open("rb") as file:
        signature = file.read(8)

    # Modern Excel .xlsx files are ZIP-based and begin with "PK".
    if signature.startswith(b"PK"):
        return pd.read_excel(
            path,
            header=None,
            engine="openpyxl",
        )

    # Older true .xls files use this binary signature.
    if signature == bytes.fromhex("D0CF11E0A1B11AE1"):
        return pd.read_excel(
            path,
            header=None,
            engine="xlrd",
        )

    # Otherwise, treat the file as CSV, tab-separated, or
    # whitespace-separated text—even if its extension says .xls.
    try:
        return pd.read_csv(
            path,
            sep=None,
            header=None,
            engine="python",
        )
    except Exception:
        return pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
        )

def load_processed_file(
    path: str | Path,
    wavelength_column: int,
    intensity_column: int,
):
    path = Path(path)
    df = read_spectrum_table(path)

    column_count = df.shape[1]

    wavelength_column = int(wavelength_column)
    intensity_column = int(intensity_column)

    # Users count columns starting at 1.
    wavelength_index = wavelength_column - 1
    intensity_index = intensity_column - 1

    if wavelength_column < 1 or wavelength_column > column_count:
        raise ValueError(
            f"Wavelength column {wavelength_column} does not exist. "
            f"The file contains {column_count} columns."
        )

    if intensity_column < 1 or intensity_column > column_count:
        raise ValueError(
            f"Intensity column {intensity_column} does not exist. "
            f"The file contains {column_count} columns."
        )

    if wavelength_index == intensity_index:
        raise ValueError(
            "Wavelength and intensity must use different columns."
        )

    exp_df = pd.DataFrame(
        {
            "exp_wavelength_nm": pd.to_numeric(
                df.iloc[:, wavelength_index],
                errors="coerce",
            ),
            "exp_intensity": pd.to_numeric(
                df.iloc[:, intensity_index],
                errors="coerce",
            ),
        }
    )

    # Removes headers, blank rows, and nonnumeric cells.
    exp_df = exp_df.dropna()

    exp_df = exp_df[
        np.isfinite(exp_df["exp_wavelength_nm"])
        & np.isfinite(exp_df["exp_intensity"])
        & (exp_df["exp_wavelength_nm"] > 0)
    ]

    exp_df = exp_df.sort_values("exp_wavelength_nm")

    if len(exp_df) < 10:
        raise ValueError(
            "Too few valid experimental points were found using "
            f"wavelength column {wavelength_column} and "
            f"intensity column {intensity_column}."
        )

    # The uploaded theory curve is no longer used or displayed.
    theory_df = pd.DataFrame(
        columns=[
            "theory_wavelength_nm",
            "theory_intensity",
        ]
    )

    return theory_df, exp_df


def normalize(y):
    y = np.asarray(y, dtype=float)
    y = y - np.nanmin(y)
    maximum = np.nanmax(y)
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros_like(y)
    return y / maximum


def gaussian_broaden(x, y, fwhm_nm):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if fwhm_nm <= 0:
        return y.copy()

    dx = np.nanmedian(np.diff(x))
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Simulation wavelength grid must be increasing.")

    sigma_points = (fwhm_nm / 2.354820045) / dx
    if sigma_points < 1e-12:
        return y.copy()

    half_width = max(1, int(np.ceil(4 * sigma_points)))
    kernel_x = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_points) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def crop_spectrum(x, y, start, end):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= start) & (x <= end)
    if np.count_nonzero(mask) < 10:
        raise ValueError("The fit region leaves fewer than 10 points.")
    return x[mask], y[mask]


def inclusive_values(start, end, step):
    if step <= 0:
        raise ValueError("Search step must be greater than zero.")
    if end < start:
        raise ValueError("Search maximum must be at least the minimum.")
    return np.arange(start, end + step / 2, step, dtype=float)


def select_temperature_indices(temps, start, end, requested_step):
    in_range = np.where((temps >= start) & (temps <= end))[0]
    if len(in_range) == 0:
        raise ValueError(
            f"No temperatures exist between {start:.0f} K and {end:.0f} K."
        )
    if len(in_range) == 1:
        return in_range

    native_step = float(np.nanmedian(np.diff(temps[in_range])))
    if not np.isfinite(native_step) or native_step <= 0:
        return in_range
    stride = max(1, int(round(requested_step / native_step)))
    return in_range[::stride]



def fit_temperature_shift_and_broadening(
    *,
    exp_x,
    exp_y,
    sim_x,
    spectra,
    temps,
    temp_start,
    temp_end,
    temp_step,
    shift_min,
    shift_max,
    shift_step,
    broadening_min,
    broadening_max,
    broadening_step,
    progress_callback: ProgressCallback | None = None,
):
    exp_target = normalize(exp_y)
    temp_indices = select_temperature_indices(
        temps, temp_start, temp_end, temp_step
    )
    shifts = inclusive_values(shift_min, shift_max, shift_step)
    broadenings = inclusive_values(
        broadening_min, broadening_max, broadening_step
    )

    total_combinations = len(temp_indices) * len(broadenings)
    rows = []
    best = {
        "temperature": None,
        "shift": None,
        "broadening": None,
        "error": np.inf,
        "simulation": None,
    }

    completed = 0
    for broadening in broadenings:
        for temp_index in temp_indices:
            temperature = float(temps[temp_index])
            simulated = gaussian_broaden(
                sim_x, spectra[temp_index], broadening
            )
            simulated = normalize(simulated)

            best_combo_error = np.inf
            best_combo_shift = None
            best_combo_interp = None

            for shift in shifts:
                interpolated = np.interp(
                    exp_x,
                    sim_x + shift,
                    simulated,
                    left=np.nan,
                    right=np.nan,
                )
                valid = np.isfinite(interpolated) & np.isfinite(exp_target)
                if np.count_nonzero(valid) < 10:
                    continue

                exp_valid = normalize(exp_target[valid])
                sim_valid = normalize(interpolated[valid])
                error = float(np.mean((exp_valid - sim_valid) ** 2))

                if error < best_combo_error:
                    best_combo_error = error
                    best_combo_shift = float(shift)
                    best_combo_interp = interpolated.copy()

            if best_combo_shift is not None:
                rows.append(
                    {
                        "Broadening FWHM (nm)": float(broadening),
                        "Temperature (K)": int(round(temperature)),
                        "Best shift (nm)": best_combo_shift,
                        "Error": best_combo_error,
                    }
                )

                if best_combo_error < best["error"]:
                    best = {
                        "temperature": int(round(temperature)),
                        "shift": best_combo_shift,
                        "broadening": float(broadening),
                        "error": best_combo_error,
                        "simulation": best_combo_interp,
                    }

            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total_combinations:
                _report(
                    progress_callback,
                    completed / max(total_combinations, 1),
                    f"Testing {int(round(temperature))} K",
                )

    if best["temperature"] is None:
        raise ValueError("No valid fit was found. Check wavelength overlap.")

    return best, pd.DataFrame(rows)


def _generate_library(
    *,
    phi: float,
    composition: dict[str, float],
    settings: AnalysisSettings,
    progress_callback: ProgressCallback | None,
) -> Path:
    try:
        from raman_library_generator import generate_phi_library
    except Exception as error:
        raise RuntimeError(
            "Could not import raman_library_generator.py. Put it in the same "
            "folder as pyside6_app.py and raman_backend.py. Original error: "
            + str(error)
        ) from error

    return generate_phi_library(
        phi=phi,
        composition=composition,
        temp_min=int(settings.generation_temp_min),
        temp_max=int(settings.generation_temp_max),
        temp_step=min(
            int(settings.generation_temp_step),
            int(settings.fit_temp_step),
        ),
        excitation_nm=float(settings.excitation_nm),
        grid_start_nm=float(settings.generation_grid_start_nm),
        grid_end_nm=float(settings.generation_grid_end_nm),
        grid_step_nm=float(settings.grid_step_nm),
        progress_callback=progress_callback,
    )


def run_complete_analysis(
    *,
    experimental_file: str | Path,
    phi: float,
    settings: AnalysisSettings,
    base_dir: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    base_dir = Path(base_dir).resolve()
    experimental_file = Path(experimental_file).resolve()

    requested_grid_step = float(settings.grid_step_nm)
    if not any(
        np.isclose(
            requested_grid_step,
            allowed,
            rtol=0.0,
            atol=1e-12,
        )
        for allowed in ALLOWED_GRID_STEPS_NM
    ):
        raise ValueError(
            "Wavelength grid step must be 0.01, 0.001, or 0.0001 nm."
        )

    if int(settings.fit_temp_step) <= 0:
        raise ValueError("Temperature step must be greater than zero.")

    if int(settings.generation_temp_max) < int(settings.generation_temp_min):
        raise ValueError(
            "Maximum temperature must be at least the minimum temperature."
        )

    requested_excitation = float(settings.excitation_nm)
    requested_grid_start = float(settings.generation_grid_start_nm)
    requested_grid_end = float(settings.generation_grid_end_nm)

    if not np.isfinite(requested_excitation) or requested_excitation <= 0:
        raise ValueError("Excitation wavelength must be greater than zero.")
    if requested_grid_end <= requested_grid_start:
        raise ValueError(
            "Generation wavelength end must be greater than its start."
        )
    if (
        float(settings.crop_start) < requested_grid_start
        or float(settings.crop_end) > requested_grid_end
    ):
        raise ValueError(
            "The generation wavelength range must fully cover the fit range."
        )

    _report(progress_callback, 0.01, "Reading experimental file")
    theory_df, exp_df = load_processed_file(
        experimental_file,
        wavelength_column=settings.wavelength_column,
        intensity_column=settings.intensity_column,
    )

    exp_x_raw = exp_df["exp_wavelength_nm"].to_numpy(dtype=float)
    exp_y_raw = exp_df["exp_intensity"].to_numpy(dtype=float)
    exp_x_crop, exp_y_crop = crop_spectrum(
        exp_x_raw,
        exp_y_raw,
        float(settings.crop_start),
        float(settings.crop_end),
    )

    reactants, product_moles, composition = calculate_ideal_h2_air_composition(phi)

    library_path = find_phi_library(
        phi,
        base_dir,
        excitation_nm=requested_excitation,
        grid_start_nm=requested_grid_start,
        grid_end_nm=requested_grid_end,
        grid_step_nm=requested_grid_step,
        requested_temp_min=float(settings.generation_temp_min),
        requested_temp_max=float(settings.generation_temp_max),
        requested_temp_step=float(settings.fit_temp_step),
        composition=composition,
    )

    needs_generation = settings.regenerate_library or library_path is None

    # A library coarser than the requested fit step cannot provide that
    # temperature resolution, so create a finer one automatically.
    if library_path is not None and not needs_generation:
        actual_temp_step = detect_library_temperature_step(library_path)
        if actual_temp_step > float(settings.fit_temp_step) + 1e-9:
            needs_generation = True

    if needs_generation:
        if not settings.auto_generate_missing and not settings.regenerate_library:
            raise FileNotFoundError(
                f"No matching library was found for Phi {phi:g}, "
                f"excitation {requested_excitation:g} nm, wavelength range "
                f"{requested_grid_start:g}–{requested_grid_end:g} nm, "
                f"grid {requested_grid_step:g} nm, and the requested "
                "temperature resolution. Enable automatic generation or "
                "place a matching NPZ in generated_libraries/."
            )

        def generation_progress(fraction: float, message: str) -> None:
            _report(progress_callback, 0.03 + 0.42 * fraction, message)

        library_path = _generate_library(
            phi=phi,
            composition=composition,
            settings=settings,
            progress_callback=generation_progress,
        )
    else:
        metadata = read_library_metadata(library_path)
        _report(
            progress_callback,
            0.10,
            f"Using {library_path.name} "
            f"(grid {float(metadata['grid_step_nm']):g} nm, "
            f"T step {float(metadata['temperature_step_K']):g} K)",
        )

    temps, sim_x, spectra = load_library(library_path)

    # Check that the loaded/generated library covers the fit range.
    # Do not regenerate here, because generation has already been handled above.
    tolerance = max(1e-8, 0.5 * requested_grid_step)

    if (
        sim_x.min() > float(settings.crop_start) + tolerance
        or sim_x.max() < float(settings.crop_end) - tolerance
    ):
        raise ValueError(
            "The loaded/generated library does not cover the selected fit range. "
            f"Library range: {sim_x.min():.6f}–{sim_x.max():.6f} nm. "
            f"Fit range: {float(settings.crop_start):.6f}–"
            f"{float(settings.crop_end):.6f} nm. "
            "Widen the generation wavelength range and run again."
        )

    _report(progress_callback, 0.44, "Preparing library for fitting...")

    stored_grid_step = float(np.median(np.diff(sim_x)))
    sim_x, spectra = resample_library_wavelength_grid(
        sim_x,
        spectra,
        target_start_nm=requested_grid_start,
        target_end_nm=requested_grid_end,
        target_step_nm=requested_grid_step,
    )

    if stored_grid_step < requested_grid_step - 1e-12:
        _report(
            progress_callback,
            0.12,
            f"Reusing {library_path.name}: wavelength grid resampled "
            f"from {stored_grid_step:g} nm to {requested_grid_step:g} nm",
        )

    if settings.allow_shift:
        shift_min = DEFAULT_SHIFT_MIN
        shift_max = DEFAULT_SHIFT_MAX
        shift_step = DEFAULT_SHIFT_STEP
    else:
        shift_min = 0.0
        shift_max = 0.0
        shift_step = 1.0

    def fit_progress(fraction: float, message: str) -> None:
        _report(progress_callback, 0.45 + 0.53 * fraction, message)

    best, fit_results = fit_temperature_shift_and_broadening(
        exp_x=exp_x_crop,
        exp_y=exp_y_crop,
        sim_x=sim_x,
        spectra=spectra,
        temps=temps,
        temp_start=float(settings.generation_temp_min),
        temp_end=float(settings.generation_temp_max),
        temp_step=float(settings.fit_temp_step),
        shift_min=shift_min,
        shift_max=shift_max,
        shift_step=shift_step,
        broadening_min=DEFAULT_BROADENING_MIN,
        broadening_max=DEFAULT_BROADENING_MAX,
        broadening_step=DEFAULT_BROADENING_STEP,
        progress_callback=fit_progress,
    )

    _report(progress_callback, 1.0, "Analysis complete")

    return AnalysisResult(
        phi=float(phi),
        reactants=reactants,
        product_moles=product_moles,
        composition=composition,
        library_path=Path(library_path),
        temps=temps,
        simulation_wavelengths=sim_x,
        experimental_wavelengths_raw=exp_x_raw,
        experimental_intensity_raw=exp_y_raw,
        theory_wavelengths=theory_df["theory_wavelength_nm"].to_numpy(dtype=float),
        theory_intensity=theory_df["theory_intensity"].to_numpy(dtype=float),
        experimental_wavelengths_crop=exp_x_crop,
        experimental_intensity_crop=exp_y_crop,
        best_temperature=int(best["temperature"]),
        best_shift=float(best["shift"]),
        best_broadening=float(best["broadening"]),
        best_error=float(best["error"]),
        best_simulation=np.asarray(best["simulation"], dtype=float),
        fit_results=fit_results,
    )
