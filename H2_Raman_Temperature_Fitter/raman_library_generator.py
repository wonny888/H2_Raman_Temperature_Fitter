from __future__ import annotations

from pathlib import Path
from typing import Callable
import os

import numpy as np

from raman_paths import LIBRARY_DIR, RESOURCE_DIR

from Raman_Intensities import H2_intensities
from Analysis_Library import basic_raman_gen_analysis


ProgressCallback = Callable[[float, str], None]


def _number_to_tag(value: float) -> str:
    text = f"{float(value):g}"
    if "." not in text:
        text += ".0"
    return text.replace(".", "p").replace("-", "m")


def _read_numeric_column(path: Path, minimum_count: int) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Required Raman model file was not found: {path}")

    values: list[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            values.append(float(parts[0]))
        except ValueError:
            continue

    if len(values) < minimum_count:
        raise ValueError(
            f"{path.name} contains {len(values)} numeric values; "
            f"at least {minimum_count} are required."
        )
    return values


def _build_molecular_arrays(constants_path: Path, vmax: int, jmax: int):
    values = _read_numeric_column(constants_path, 20)

    (
        we, wexe, weye, weze,
        go,
        be, ba, bb, bc,
        de, da, db,
        he, ha,
        le, la,
        alpha, gamma,
        intermediate_state,
        int_factor,
    ) = values[:20]

    vib = np.zeros(vmax + 1)
    rot = np.zeros((vmax + 1, jmax + 1))
    rovib = np.zeros((vmax + 1, jmax + 1))

    raman_vib = np.zeros(vmax)
    raman_q = np.zeros((vmax, jmax + 1))
    raman_o = np.zeros((vmax, jmax + 1))
    raman_s = np.zeros((vmax, jmax + 1))

    for v in range(vmax + 1):
        vp = v + 0.5
        vib[v] = (
            we * vp
            + wexe * vp**2
            + weye * vp**3
            + weze * vp**4
            - go
        )

        for j in range(jmax + 1):
            jj = j * (j + 1)
            rot[v, j] = (
                jj * (be + ba * vp + bb * vp**2 + bc * vp**3)
                - jj**2 * (de + da * vp + db * vp**2)
                + jj**3 * (he + ha * vp)
                - jj**4 * (le + la * vp)
            )
            rovib[v, j] = rot[v, j] + vib[v]

    for v in range(vmax):
        raman_vib[v] = vib[v + 1] - vib[v]
        for j in range(jmax + 1):
            raman_q[v, j] = rovib[v + 1, j] - rovib[v, j]

        for j in range(jmax - 1):
            raman_s[v, j] = rovib[v + 1, j + 2] - rovib[v, j]

        for j in range(2, jmax + 1):
            raman_o[v, j] = rovib[v + 1, j - 2] - rovib[v, j]

    return (
        vib,
        rovib,
        raman_vib,
        raman_q,
        raman_o,
        raman_s,
        alpha,
        gamma,
        be,
        intermediate_state,
        int_factor,
    )


def _read_base_system_settings(spec_path: Path) -> dict[str, float]:
    values = _read_numeric_column(spec_path, 39)

    names = [
        "equilibrium",
        "pressure",
        "Tmin",
        "Tmax",
        "H2conc",
        "N2conc",
        "O2conc",
        "COconc",
        "H2Oconc",
        "CO2conc",
        "Heconc",
        "lambdaN",
        "linewidthN",
        "lambdaB",
        "linewidthB",
        "lambdaspect",
        "dispersion",
        "slitwidth",
        "grid_start",
        "grid_end",
        "increment",
        "N_lineshape",
        "NGLvalue",
        "VNG",
        "VNL",
        "B_lineshape",
        "BGLvalue",
        "VBG",
        "VBL",
        "filter_flag",
        "printS",
        "analysis",
        "method",
        "mix",
        "variance",
        "T_meas",
        "shots",
        "printM",
        "T_fluid",
    ]
    return dict(zip(names, values[:39]))


def _make_filter_fit(
    base_dir: Path,
    filter_flag: int,
    grid_start: float,
    grid_end: float,
):
    if filter_flag != 0:
        return None

    filter_path = base_dir / "Spectral_Filter.txt"
    if not filter_path.exists():
        raise FileNotFoundError(
            "The specifications request a spectral filter, but "
            f"{filter_path.name} was not found."
        )

    wavelength = []
    absorption = []
    for line in filter_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            wavelength.append(float(parts[0]))
            absorption.append(float(parts[1]))
        except ValueError:
            continue

    wavelength_array = np.asarray(wavelength, dtype=float)
    absorption_array = np.asarray(absorption, dtype=float)

    if len(wavelength_array) < 7:
        raise ValueError("Spectral_Filter.txt does not contain enough points.")

    low = min(grid_start, grid_end)
    high = max(grid_start, grid_end)
    mask = (wavelength_array >= low) & (wavelength_array <= high)

    if np.count_nonzero(mask) < 7:
        raise ValueError(
            "The spectral-filter file does not cover the requested generation grid. "
            "Set filter_flag to 1 in Raman_Input_Specifications.txt if no filter "
            "should be applied to this requested wavelength range."
        )

    coefficients = np.polyfit(
        wavelength_array[mask],
        absorption_array[mask],
        6,
    )
    return coefficients[::-1]


def _npz_scalar(data, key: str) -> float | None:
    if key not in data.files:
        return None
    values = np.asarray(data[key], dtype=float).ravel()
    if values.size == 0:
        return None
    value = float(values[0])
    return value if np.isfinite(value) else None


def _is_exact_duplicate_library(
    candidate: Path,
    *,
    phi: float,
    excitation_nm: float,
    grid_start_nm: float,
    grid_end_nm: float,
    grid_step_nm: float,
    temp_min: int,
    temp_max: int,
    temp_step: int,
    composition: dict[str, float],
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
) -> bool:
    """Return True only when a saved NPZ has the exact same configuration."""
    try:
        with np.load(candidate, allow_pickle=False) as data:
            required = {
                "temps",
                "wavelengths",
                "phi",
                "excitation_nm",
                "grid_start_nm",
                "grid_end_nm",
                "grid_step_nm",
                "temperature_step_K",
                "H2conc",
                "N2conc",
                "H2Oconc",
                "O2conc",
            }
            if not required.issubset(data.files):
                return False

            scalar_checks = [
                (_npz_scalar(data, "phi"), phi, 1e-10),
                (_npz_scalar(data, "excitation_nm"), excitation_nm, 1e-9),
                (_npz_scalar(data, "grid_start_nm"), grid_start_nm, 1e-9),
                (_npz_scalar(data, "grid_end_nm"), grid_end_nm, 1e-9),
                (_npz_scalar(data, "grid_step_nm"), grid_step_nm, 1e-12),
                (_npz_scalar(data, "temperature_step_K"), temp_step, 1e-9),
                (_npz_scalar(data, "H2conc"), composition["H2"], 1e-10),
                (_npz_scalar(data, "N2conc"), composition["N2"], 1e-10),
                (_npz_scalar(data, "H2Oconc"), composition["H2O"], 1e-10),
                (_npz_scalar(data, "O2conc"), composition["O2"], 1e-10),
            ]

            for stored, expected, tolerance in scalar_checks:
                if stored is None or not np.isclose(
                    float(stored),
                    float(expected),
                    rtol=0.0,
                    atol=tolerance,
                ):
                    return False

            candidate_temps = np.asarray(data["temps"], dtype=float).ravel()
            candidate_waves = np.asarray(data["wavelengths"], dtype=float).ravel()

            if candidate_temps.shape != temperatures.shape:
                return False
            if candidate_waves.shape != wavelengths.shape:
                return False
            if not np.allclose(
                candidate_temps,
                temperatures,
                rtol=0.0,
                atol=1e-9,
            ):
                return False
            if not np.allclose(
                candidate_waves,
                wavelengths,
                rtol=0.0,
                atol=1e-10,
            ):
                return False

            # These checks make the intended temperature range explicit even
            # for files created with older filename conventions.
            if not np.isclose(candidate_temps.min(), temp_min, atol=1e-9):
                return False
            if not np.isclose(candidate_temps.max(), temp_max, atol=1e-9):
                return False

            return True
    except Exception:
        # Broken or unrelated files are left untouched.
        return False


def _delete_duplicate_libraries(
    output_dir: Path,
    keep_path: Path,
    *,
    phi: float,
    excitation_nm: float,
    grid_start_nm: float,
    grid_end_nm: float,
    grid_step_nm: float,
    temp_min: int,
    temp_max: int,
    temp_step: int,
    composition: dict[str, float],
    temperatures: np.ndarray,
    wavelengths: np.ndarray,
) -> list[Path]:
    """Delete only exact duplicate configurations, never different libraries."""
    deleted: list[Path] = []
    keep_resolved = keep_path.resolve()

    for candidate in output_dir.glob("*.npz"):
        if candidate.resolve() == keep_resolved:
            continue

        if _is_exact_duplicate_library(
            candidate,
            phi=phi,
            excitation_nm=excitation_nm,
            grid_start_nm=grid_start_nm,
            grid_end_nm=grid_end_nm,
            grid_step_nm=grid_step_nm,
            temp_min=temp_min,
            temp_max=temp_max,
            temp_step=temp_step,
            composition=composition,
            temperatures=temperatures,
            wavelengths=wavelengths,
        ):
            candidate.unlink()
            deleted.append(candidate)

    return deleted


def generate_phi_library(
    *,
    phi: float,
    composition: dict[str, float],
    temp_min: int,
    temp_max: int,
    temp_step: int = 5,
    excitation_nm: float = 266.0,
    grid_start_nm: float = 296.0,
    grid_end_nm: float = 300.2,
    grid_step_nm: float = 0.01,
    base_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """
    Generate a Phi-specific H2 Raman library directly as one NPZ file.

    This bypasses all terminal input prompts and does not create the individual
    temperature CSV files. It reads the instrument/lineshape settings from
    Raman_Input_Specifications.txt, then overrides Phi composition, temperature
    range, excitation wavelength, and wavelength grid.
    """
    phi = float(phi)
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("Phi must be greater than zero.")

    temp_min = int(temp_min)
    temp_max = int(temp_max)
    temp_step = int(temp_step)
    excitation_nm = float(excitation_nm)
    grid_start_nm = float(grid_start_nm)
    grid_end_nm = float(grid_end_nm)
    grid_step_nm = float(grid_step_nm)

    if not np.isfinite(excitation_nm) or excitation_nm <= 0:
        raise ValueError("Excitation wavelength must be greater than zero.")
    if temp_step <= 0:
        raise ValueError("Temperature step must be greater than zero.")
    if temp_max < temp_min:
        raise ValueError("Maximum temperature must be at least the minimum.")

    allowed_grid_steps = (0.01, 0.001, 0.0001)
    if not any(
        np.isclose(grid_step_nm, allowed, rtol=0.0, atol=1e-12)
        for allowed in allowed_grid_steps
    ):
        raise ValueError(
            "Grid step must be 0.01, 0.001, or 0.0001 nm."
        )

    if grid_end_nm <= grid_start_nm:
        raise ValueError("Grid end must be greater than grid start.")

    required_species = ["H2", "N2", "H2O", "O2"]
    missing = [name for name in required_species if name not in composition]
    if missing:
        raise ValueError("Composition is missing: " + ", ".join(missing))

    h2conc = float(composition["H2"])
    n2conc = float(composition["N2"])
    h2oconc = float(composition["H2O"])
    o2conc = float(composition["O2"])
    coconc = float(composition.get("CO", 0.0))
    co2conc = float(composition.get("CO2", 0.0))
    heconc = float(composition.get("He", 0.0))

    concentration_total = (
        h2conc + n2conc + h2oconc + o2conc + coconc + co2conc + heconc
    )
    if not np.isclose(concentration_total, 1.0, atol=1e-8):
        raise ValueError(
            "Generated composition must sum to 1. "
            f"Current total is {concentration_total:.12f}."
        )

    model_dir = RESOURCE_DIR
    constants_path = model_dir / "H2_Molecular_Constants.txt"
    specs_path = model_dir / "Raman_Input_Specifications.txt"

    settings = _read_base_system_settings(specs_path)

    # Preserve the original narrowband/broadband wavelength separation.
    original_delta_nm = float(settings["lambdaB"] - settings["lambdaN"])
    lambda_n = float(excitation_nm)
    lambda_b = float(excitation_nm + original_delta_nm)

    pressure = float(settings["pressure"])
    linewidth_n = float(settings["linewidthN"])
    linewidth_b = float(settings["linewidthB"])
    lambda_spect = float(settings["lambdaspect"])
    dispersion = float(settings["dispersion"])
    slitwidth = float(settings["slitwidth"])
    n_lineshape = int(settings["N_lineshape"])
    ngl_value = float(settings["NGLvalue"])
    vng = float(settings["VNG"])
    vnl = float(settings["VNL"])
    b_lineshape = int(settings["B_lineshape"])
    bgl_value = float(settings["BGLvalue"])
    vbg = float(settings["VBG"])
    vbl = float(settings["VBL"])
    filter_flag = int(settings["filter_flag"])
    mix = float(settings["mix"])
    t_fluid = int(settings["T_fluid"])

    filter_fit = _make_filter_fit(
        model_dir,
        filter_flag,
        float(grid_start_nm),
        float(grid_end_nm),
    )

    vmax = 10
    jmax = 25
    species = 1

    (
        vib,
        rovib,
        raman_vib,
        raman_q,
        raman_o,
        raman_s,
        alpha,
        gamma,
        be,
        intermediate_state,
        int_factor,
    ) = _build_molecular_arrays(constants_path, vmax, jmax)

    nlcm = 1.0 / (lambda_n * 1e-7)
    blcm = 1.0 / (lambda_b * 1e-7)
    num_grid_points = int(round((grid_end_nm - grid_start_nm) / grid_step_nm))

    temperatures = np.arange(
        temp_min,
        temp_max + temp_step,
        temp_step,
        dtype=int,
    )
    temperatures = temperatures[temperatures <= temp_max]

    if len(temperatures) == 0:
        raise ValueError("No temperatures were selected for library generation.")

    spectra_rows: list[np.ndarray] = []
    wavelength_grid: np.ndarray | None = None
    conc = h2conc

    # Model modules use relative paths, so generate while temporarily in model_dir.
    previous_cwd = Path.cwd()
    os.chdir(model_dir)
    try:
        for index, temperature in enumerate(temperatures):
            if progress_callback is not None:
                progress_callback(
                    index / len(temperatures),
                    f"Generating {int(temperature)} K "
                    f"({index + 1}/{len(temperatures)})",
                )

            (
                n_wave,
                n_int,
                num_n_transitions,
                b_wave,
                b_int,
                num_b_transitions,
                rho,
            ) = H2_intensities(
                int(temperature),
                pressure,
                h2conc,
                n2conc,
                h2oconc,
                o2conc,
                coconc,
                co2conc,
                heconc,
                vmax,
                jmax,
                nlcm,
                blcm,
                vib,
                raman_vib,
                rovib,
                raman_q,
                raman_o,
                raman_s,
                alpha,
                gamma,
                be,
                t_fluid,
                intermediate_state,
                conc,
                int_factor,
                float(grid_start_nm),
                float(grid_end_nm),
                filter_flag,
                filter_fit,
            )

            plot_x, n_spec, b_spec, _ = basic_raman_gen_analysis(
                int(temperature),
                num_grid_points,
                num_n_transitions,
                num_b_transitions,
                n_lineshape,
                b_lineshape,
                ngl_value,
                bgl_value,
                n_wave,
                n_int,
                b_wave,
                b_int,
                1,
                species,
                pressure,
                rho,
                jmax,
                h2conc,
                h2oconc,
                n2conc,
                o2conc,
                heconc,
                coconc,
                co2conc,
                linewidth_n,
                linewidth_b,
                lambda_spect,
                dispersion,
                slitwidth,
                vng,
                vnl,
                vbg,
                vbl,
                0,  # analysis
                1,  # method
                1,  # shots
                0,  # T_meas
                0,  # variance
                0,  # num_meas_data
                1,  # printM
                mix,
                np.array([]),
                np.array([]),
                np.array([]),
                float(grid_start_nm),
                float(grid_step_nm),
            )

            predicted = np.asarray(n_spec, dtype=float) * (1.0 - mix) + np.asarray(
                b_spec, dtype=float
            ) * mix

            if wavelength_grid is None:
                wavelength_grid = np.asarray(plot_x, dtype=float)
            elif not np.allclose(wavelength_grid, plot_x, rtol=0.0, atol=1e-10):
                predicted = np.interp(
                    wavelength_grid,
                    np.asarray(plot_x, dtype=float),
                    predicted,
                )

            if not np.all(np.isfinite(predicted)):
                raise ValueError(
                    f"Non-finite simulated intensity was produced at {temperature} K."
                )

            spectra_rows.append(predicted)
    finally:
        os.chdir(previous_cwd)

    if wavelength_grid is None or not spectra_rows:
        raise RuntimeError("The Raman model did not produce any spectra.")

    spectra_array = np.vstack(spectra_rows)
    output_dir = LIBRARY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"phi_{_number_to_tag(phi)}_"
        f"{_number_to_tag(excitation_nm)}nm_"
        f"{_number_to_tag(grid_start_nm)}_{_number_to_tag(grid_end_nm)}_"
        f"grid{_number_to_tag(grid_step_nm)}_"
        f"T{temp_min}_{temp_max}_step{temp_step}.npz"
    )
    output_path = output_dir / filename

    np.savez_compressed(
        output_path,
        temps=temperatures.astype(float),
        wavelengths=wavelength_grid,
        spectra=spectra_array,
        phi=np.asarray(phi),
        H2conc=np.asarray(h2conc),
        N2conc=np.asarray(n2conc),
        H2Oconc=np.asarray(h2oconc),
        O2conc=np.asarray(o2conc),
        excitation_nm=np.asarray(excitation_nm),
        grid_start_nm=np.asarray(grid_start_nm),
        grid_end_nm=np.asarray(grid_end_nm),
        grid_step_nm=np.asarray(grid_step_nm),
        temperature_min_K=np.asarray(temp_min),
        temperature_max_K=np.asarray(temp_max),
        temperature_step_K=np.asarray(temp_step),
        temperature_count=np.asarray(len(temperatures)),
        wavelength_count=np.asarray(len(wavelength_grid)),
        library_format_version=np.asarray(2),
    )

    deleted = _delete_duplicate_libraries(
        output_dir,
        output_path,
        phi=phi,
        excitation_nm=excitation_nm,
        grid_start_nm=grid_start_nm,
        grid_end_nm=grid_end_nm,
        grid_step_nm=grid_step_nm,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_step=temp_step,
        composition={
            "H2": h2conc,
            "N2": n2conc,
            "H2O": h2oconc,
            "O2": o2conc,
        },
        temperatures=temperatures.astype(float),
        wavelengths=np.asarray(wavelength_grid, dtype=float),
    )

    if progress_callback is not None:
        message = f"Saved {output_path.name}"
        if deleted:
            message += f"; removed {len(deleted)} exact duplicate(s)"
        progress_callback(1.0, message)

    return output_path
