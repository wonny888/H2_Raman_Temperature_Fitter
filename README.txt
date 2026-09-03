# H2 RAMAN TEMPERATURE FITTER

## CONTACT INFORMATION

Email:

wonsun.you@vanderbilt.edu
frances.c.brown@vanderbilt.edu
stefan.conde@vanderbilt.edu


## About

H2 Raman Temperature Fitter estimates gas temperature by comparing a processed experimental H2 Raman spectrum with simulated H2 Raman spectra.

The program can:

- Load wavelength-calibrated experimental spectra
- Let the user choose the wavelength and intensity columns
- Preview the experimental spectrum
- Detect or enter equivalence ratio, Phi
- Calculate the H2-air product gas composition
- Generate or load a matching Raman simulation library
- Compare experimental and simulated spectra
- Report the best-fit temperature, shift, broadening, and fit error
- Display the experimental spectrum with the best-fit simulation

## How to Run

From the Python/LIF_Raman folder, run:

python app.py

Required Python packages:

pip install numpy scipy matplotlib pandas PySide6

## Important Files

The main UI file is:

app.py

The main support files are:

raman_backend.py
raman_library_generator.py
raman_paths.py
phi_calculator.py
build_spectra_library.py
H2_Raman_Model.py

The Raman model also uses supporting files such as:

Generic_Raman_Functions.py
Raman_Intensities.py
Raman_Linewidths.py
Raman_System_Specifications.py
H2_Molecular_Constants.txt
Raman_Input_Specifications.txt

Do not delete these files unless you know they are no longer used.

## Experimental Data Requirements

The program requires a processed experimental spectrum with:

1. A wavelength column in nanometers
2. A background-subtracted intensity column

The wavelength must already be calibrated. The program does not currently convert detector pixels to wavelength.

The experimental background should already be removed before loading the file. Slightly negative intensity values after background subtraction are usually acceptable as long as the H2 Raman peaks are still visible.

Supported file types include:

.csv
.txt
.dat
.xlsx
.xls

The file can be comma-separated, tab-separated, whitespace-separated, or a real Excel workbook.

## Selecting Columns

Column numbering starts at 1, like Excel.

For a normal two-column file:

Wavelength column: 1
Intensity column: 2

## Running an Analysis

Basic steps:

1. Open the app.
2. Select the experimental file.
3. Enter the wavelength and intensity column numbers.
4. Check that the preview looks correct.
5. Enter or verify Phi.
6. Choose the fit wavelength range.
7. Choose the temperature range and temperature step.
8. Choose the generation wavelength range.
9. Choose the wavelength grid increment.
10. Choose whether wavelength shift fitting should be enabled.
11. Run the analysis.

The program then crops and normalizes the experimental spectrum, calculates the gas composition, loads or generates a matching Raman library, compares the spectra, and displays the best-fit result.

## Equivalence Ratio

Phi is the flame equivalence ratio.

Phi < 1: lean
Phi = 1: stoichiometric
Phi > 1: fuel-rich

The program may try to detect Phi from the filename, but the user should always verify it before running the analysis.

## Fit Wavelength Range

The fit range is the part of the experimental spectrum compared with the simulations.

Example:

Fit start: 296.7 nm
Fit end: 299.5 nm

Choose a region that contains useful H2 Raman structure. Avoid unrelated peaks, noisy edge regions, or contamination when possible.

## Generation Wavelength Range

The generation range is the wavelength range used for simulated spectra.

Example:

Generation start: 296.0 nm
Generation end: 300.2 nm

The generation range must fully cover the fit range.

## Temperature Settings

Choose the minimum temperature, maximum temperature, and temperature step.

A smaller temperature step gives finer resolution but takes longer and creates larger libraries.

## Fitting Parameters

During fitting, the program compares the experimental spectrum with each simulated spectrum in the library.

The program tests Gaussian broadening values from:

- `0.000 nm` to `0.150 nm`

When wavelength shift fitting is enabled, the program also tests wavelength shifts from:

- `-0.10 nm` to `+0.10 nm`

For each temperature, broadening, and wavelength-shift combination, the program calculates a fitting error. The lowest-error result is selected as the best-fit gas temperature.

## Wavelength Grid Increment

Common grid increments include:

0.01 nm
0.001 nm
0.0001 nm

A smaller grid increment creates more wavelength points and gives a higher-resolution library, but it also increases generation time, memory use, and file size.

## Generated Libraries

The program saves generated Raman simulation libraries so they can be reused later.

When running from source with:

python app.py

generated libraries are saved in the local generated_libraries folder, usually:

Python/LIF_Raman/generated_libraries

If a compatible library already exists, the program loads it instead of generating a new one.

Compatibility depends on settings such as:

- Phi
- Gas composition
- Excitation wavelength
- Temperature range
- Temperature step
- Generation wavelength range
- Wavelength grid increment

A broader or higher-resolution library may be reused for a narrower or coarser analysis.

## Results

The program reports:

- Best-fit temperature in kelvin
- Best wavelength shift in nanometers
- Best Gaussian broadening in nanometers
- Best fit error
- Simulation library used

The final plot compares the cropped experimental spectrum with the best-fit simulated spectrum.

The reported temperature is a model-based best-fit result.

## Troubleshooting

If the app does not run or gets stuck, close the application and restart it.

### The preview does not appear

Check that:

- An experimental file was selected
- Both column boxes contain numbers
- The selected columns are different
- The selected columns exist
- The columns contain numeric data
- At least 10 valid rows are available

### The preview looks wrong

Check:

- Wavelength column number
- Intensity column number
- Wavelength units
- File format or delimiter
- Whether the data is still in pixels instead of nanometers

### The fit range leaves too few points

Widen the fit wavelength range or check that the experimental wavelength data overlaps the selected range.

### The generation range does not cover the fit range

Change the generation start and end values so the generation range fully contains the fit range.

### Library generation is slow

Library generation becomes slower when using:

- A large temperature range
- A small temperature step
- A wide wavelength range
- A very fine wavelength grid, such as 0.0001 nm

After generation finishes, the saved library should be reused for future compatible analyses.

### The result is at the minimum or maximum temperature

The true best fit may be outside the selected search range. Expand the temperature range and run the analysis again.

### The experiment and simulation do not match well

Possible causes include:

- Wrong wavelength or intensity column
- Wrong Phi
- Poor wavelength calibration
- Background contamination
- Incorrect fitting region
- Insufficient temperature range
- Instrument broadening differences
- Experimental noise
- Incorrect gas composition
- Unrelated spectral features