from pathlib import Path
import re

import numpy as np
import pandas as pd


# ==================================================
# SETTINGS YOU CHANGE
# ==================================================

PHI_FOR_LIBRARY = 3.5

EXCITATION_NM = 266.0
GRID_START_NM = 296.0
GRID_END_NM = 300.2
GRID_STEP_NM = 0.01

# Folder containing the individual generated spectra CSV files
# Example files: 800_spectrum_convoluted.csv, 801_spectrum_convoluted.csv, etc.
BASE_DIR = Path(__file__).resolve().parent
GENERATED_SPECTRA_DIR = BASE_DIR / "generated_spectra"

# Folder where the final .npz library will be saved
GENERATED_LIBRARIES_DIR = BASE_DIR / "generated_libraries"


# ==================================================
# HELPERS
# ==================================================

def number_to_tag(value):
    """
    Converts 4.0 -> 4p0 and 296.0 -> 296p0 for filenames.
    """
    return str(value).replace(".", "p")


def parse_temperature_from_filename(path):
    """
    Reads temperature from filename.

    Works for names like:
        800_spectrum_convoluted.csv
        2760_spectrum_convoluted.csv
        2760K_spectrum_convoluted.csv
    """
    name = path.stem

    match = re.search(r"(\d+)", name)

    if match is None:
        raise ValueError(f"Could not find temperature in filename: {path.name}")

    return int(match.group(1))


def choose_wavelength_column(df):
    """
    Tries to find the wavelength column automatically.
    """

    possible_names = [
        "wavelength",
        "wavelength_nm",
        "lambda",
        "lambda_nm",
        "x",
        "N_spec",
        "n_spec",
    ]

    lower_cols = {str(c).lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]

    # If no named column works, use first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        raise ValueError("Could not detect wavelength column.")

    return numeric_cols[0]


def choose_intensity_column(df):
    """
    Tries to find the predicted/simulated intensity column automatically.
    """

    possible_names = [
        "Predicted",
        "predicted",
        "B_Spec",
        "b_spec",
        "intensity",
        "Intensity",
        "signal",
        "Signal",
        "y",
    ]

    lower_cols = {str(c).lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]

    # If no named column works, use last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        raise ValueError("Could not detect intensity column.")

    return numeric_cols[-1]


def read_spectrum_csv(path):
    """
    Reads one generated spectrum CSV.
    Returns wavelength array and intensity array.
    """

    df = pd.read_csv(path)

    wave_col = choose_wavelength_column(df)
    intensity_col = choose_intensity_column(df)

    wavelengths = df[wave_col].to_numpy(dtype=float)
    intensity = df[intensity_col].to_numpy(dtype=float)

    # Sort by wavelength just in case
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    intensity = intensity[order]

    return wavelengths, intensity


def build_library():
    """
    Builds temps, wavelengths, spectra arrays from generated_spectra/*.csv
    """

    if not GENERATED_SPECTRA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find generated spectra folder:\n{GENERATED_SPECTRA_DIR}"
        )

    csv_files = sorted(GENERATED_SPECTRA_DIR.glob("*spectrum_convoluted*.csv"))

    if not csv_files:
        csv_files = sorted(GENERATED_SPECTRA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{GENERATED_SPECTRA_DIR}"
        )

    print(f"Found {len(csv_files)} CSV spectra files.")

    rows = []

    reference_wavelengths = None

    for path in csv_files:
        try:
            temp = parse_temperature_from_filename(path)
            wavelengths, intensity = read_spectrum_csv(path)

            if reference_wavelengths is None:
                reference_wavelengths = wavelengths
            else:
                if len(wavelengths) != len(reference_wavelengths):
                    print(f"Skipping {path.name}: wavelength length does not match.")
                    continue

                if not np.allclose(wavelengths, reference_wavelengths, rtol=0, atol=1e-8):
                    print(f"Skipping {path.name}: wavelength grid does not match.")
                    continue

            rows.append((temp, intensity, path.name))

        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not rows:
        raise ValueError("No valid spectra were loaded.")

    # Sort by temperature
    rows.sort(key=lambda x: x[0])

    temps = np.array([r[0] for r in rows], dtype=float)
    spectra = np.vstack([r[1] for r in rows])
    wavelengths = reference_wavelengths

    print()
    print("Library summary:")
    print(f"Temperatures: {temps.min():.0f} K to {temps.max():.0f} K")
    print(f"Number of temperatures: {len(temps)}")
    print(f"Wavelength range: {wavelengths.min():.4f} to {wavelengths.max():.4f} nm")
    print(f"Number of wavelength points: {len(wavelengths)}")
    print(f"Spectra shape: {spectra.shape}")

    return temps, wavelengths, spectra


def save_phi_library(temps, wavelengths, spectra):
    """
    Saves the generated spectra library directly into generated_libraries/
    with a phi-specific filename that app.py can automatically find.
    """

    GENERATED_LIBRARIES_DIR.mkdir(exist_ok=True)

    phi_tag = number_to_tag(PHI_FOR_LIBRARY)
    grid_start_tag = number_to_tag(GRID_START_NM)
    grid_end_tag = number_to_tag(GRID_END_NM)
    grid_step_tag = number_to_tag(GRID_STEP_NM)

    library_filename = (
        f"phi_{phi_tag}"
        f"_{int(EXCITATION_NM)}nm"
        f"_{grid_start_tag}_{grid_end_tag}"
        f"_step{grid_step_tag}"
        f".npz"
    )

    library_path = GENERATED_LIBRARIES_DIR / library_filename

    np.savez_compressed(
        library_path,
        temps=np.asarray(temps),
        wavelengths=np.asarray(wavelengths),
        spectra=np.asarray(spectra),
    )

    print()
    print("===================================")
    print("Saved phi-specific library to:")
    print(library_path)
    print("===================================")
    print()

    return library_path


def main():
    temps, wavelengths, spectra = build_library()
    save_phi_library(temps, wavelengths, spectra)


if __name__ == "__main__":
    main() 