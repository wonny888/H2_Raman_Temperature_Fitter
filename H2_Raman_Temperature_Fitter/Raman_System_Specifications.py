import numpy as np
import sys
import os


# ------------------------------------------------------------
# Validation Helpers (C++ Ierrorfunction / Derrorfunction)
# ------------------------------------------------------------

def check_int(value, low, high, message):
    if not (low <= value <= high):
        raise ValueError(message)


def check_float(value, low, high, message):
    if not (low <= value <= high):
        raise ValueError(message)


# ------------------------------------------------------------
# Main Raman System Specification Reader
# ------------------------------------------------------------

def raman_system_specifications(species):

    filter_fit = None
    num_meas_data = 0
    wave_meas = None
    int_meas = None
    noise_meas = None
    params = {}

    input_file = int(input("READ RAMAN SPECIFICATIONS FROM FILE (0=yes; 1=no): "))

    # ============================================================
    # READ FROM FILE
    # ============================================================

    if input_file == 0:

        with open("Raman_Input_Specifications.txt", "r") as f:
            lines = f.readlines()

        values = []
        for line in lines:
            try:
                values.append(float(line.strip().split()[0]))
            except:
                continue

        # Assign values in order (same order as C++)
        (
            equilibrium,
            pressure,
            Tmin,
            Tmax,
            H2conc,
            N2conc,
            O2conc,
            COconc,
            H2Oconc,
            CO2conc,
            Heconc,
            lambdaN,
            linewidthN,
            lambdaB,
            linewidthB,
            lambdaspect,
            dispersion,
            slitwidth,
            grid_start,
            grid_end,
            increment,
            N_lineshape,
            NGLvalue,
            VNG,
            VNL,
            B_lineshape,
            BGLvalue,
            VBG,
            VBL,
            filter_flag,
            printS,
            analysis,
            method,
            mix,
            variance,
            T_meas,
            shots,
            printM,
            T_fluid
        ) = values[:39]

        T_meas = int(T_meas)

    # ============================================================
    # MANUAL ENTRY
    # ============================================================

    else:

        equilibrium = int(input("Is system in equilibrium (0=yes;1=no): "))
        if equilibrium == 1:
            raise RuntimeError("System must be thermodynamic equilibrium.")

        pressure = float(input("Enter pressure (Pa): "))
        T_fluid = int(input("Enter source fluid temperature (K): "))
        Tmin = int(input("Minimum temperature (K): "))
        Tmax = int(input("Maximum temperature (K): "))

        if Tmax < Tmin:
            raise ValueError("Tmax must be > Tmin")
        if input_file == 0 and (Tmax - Tmin) > 375:
            raise ValueError("T range must be <= 375 for file input")
        if input_file == 1 and (Tmax - Tmin) > 250:
            raise ValueError("T range must be <= 250 for manual input")

        H2conc = float(input("H2 concentration: "))
        N2conc = float(input("N2 concentration: "))
        H2Oconc = float(input("H2O concentration: "))
        O2conc = float(input("O2 concentration: "))
        CO2conc = float(input("CO2 concentration: "))
        COconc = float(input("CO concentration: "))
        Heconc = float(input("He concentration: "))

        lambdaN = float(input("Narrowband wavelength (nm): "))
        linewidthN = float(input("Narrowband linewidth (nm): "))
        lambdaB = float(input("Broadband wavelength (nm): "))
        linewidthB = float(input("Broadband linewidth (nm): "))

        lambdaspect = float(input("Spectrograph resolution (nm): "))
        dispersion = float(input("Spectrograph dispersion (nm/mm): "))
        slitwidth = float(input("Slitwidth (mm): "))

        grid_start = float(input("Grid start wavelength: "))
        grid_end = float(input("Grid end wavelength: "))
        increment = float(input("Grid increment: "))

        filter_flag = int(input("Apply spectral filter (0=yes;1=no): "))

        N_lineshape = int(input("Narrowband lineshape (1-7): "))
        B_lineshape = int(input("Broadband lineshape (1-7): "))

        NGLvalue = 0
        BGLvalue = 0
        VNG = VNL = VBG = VBL = 0

        printS = int(input("Save generated spectra (0=yes;1=no): "))

        analysis = int(input("Analysis option (0,1,2): "))
        method = 1
        mix = 0
        variance = 0
        T_meas = 0
        shots = 1
        printM = 0

        if analysis in [1, 2]:
            method = int(input("Method (1 or 2): "))
            mix = float(input("Mix value: "))
            variance = int(input("Variance steps: "))
            T_meas = int(input("Measured temperature: "))
            shots = int(input("Number of shots: "))
            printM = int(input("Save best fit (0=yes;1=no): "))

    # ============================================================
    # VALIDATION
    # ============================================================

    check_float(pressure, 0, 1e8, "Pressure must be 0-1e8")
    check_int(T_fluid, 15, 4000, "Fluid T 15-4000")
    check_int(Tmin, 15, 4000, "Tmin 15-4000")
    check_int(Tmax, 15, 4000, "Tmax 15-4000")

    totalconc = H2conc + N2conc + H2Oconc + O2conc + CO2conc + COconc + Heconc
    if totalconc > 1:
        raise ValueError("Total mixture concentration cannot exceed 1")

    check_float(lambdaN, 240.0, 270.0, "UV wavelength required")
    check_float(lambdaB, 240.0, 270.0, "UV wavelength required")

    check_float(linewidthN, 0.001, 1.0, "Linewidth 0.001-1.0 nm")
    check_float(linewidthB, 0.001, 1.0, "Linewidth 0.001-1.0 nm")

    num_grid_points = int((grid_end - grid_start) / increment)

    if num_grid_points > 65001:
        raise ValueError("Too many grid points")

    NLcm = 1 / (lambdaN * 1e-7)
    BLcm = 1 / (lambdaB * 1e-7)

    # ============================================================
    # STORE PARAMETERS
    # ============================================================

    params.update(locals())


    filter_fit = None

    if filter_flag == 0:

        wavelength = []
        absorption = []

        with open("Spectral_Filter.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    wavelength.append(float(parts[0]))
                    absorption.append(float(parts[1]))

        wavelength = np.array(wavelength)
        absorption = np.array(absorption)

        # Find closest indices to grid_start and grid_end
        start = np.argmin(np.abs(wavelength - grid_start))
        end = np.argmin(np.abs(wavelength - grid_end))

        # Ensure proper ordering
        if start < end:
            start, end = end, start

        wavelength_slice = wavelength[end:start+1][::-1]
        absorption_slice = absorption[end:start+1][::-1]

        # 6th order polynomial fit
        filter_fit = np.polyfit(wavelength_slice, absorption_slice, 6)
        filter_fit = filter_fit[::-1]
        # Save coefficients
        with open("Filter_Coefficients.txt", "w") as f:
            f.write("Constant\tValue\n")
            for i, coef in enumerate(filter_fit):
                f.write(f"{i}\t{coef:.15e}\n")

    wave_meas = []
    int_meas = []
    noise_meas = []

    if analysis in (1, 2):

        # =========================
        # AVERAGE spectrum (shots == 1)
        # =========================
        if shots == 1:

            filename = f"{T_meas}_Avg_Measured.txt"

            with open(filename, "r") as f:
                lines = f.readlines()[2:]  # skip headers

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    wave_meas.append(float(parts[0]))
                    int_meas.append(float(parts[1]))
                    noise_meas.append(float(parts[2]))

            wave_meas = np.array(wave_meas)
            int_meas = np.array(int_meas)
            noise_meas = np.array(noise_meas)

            num_meas_data = len(wave_meas)

        # =========================
        # SINGLE SHOT (shots > 1)
        # =========================
        if shots > 1:

            # Read wavelengths
            filename_meas = f"{T_meas}_SS_Measured.txt"

            with open(filename_meas, "r") as f:
                lines = f.readlines()[1:]  # skip header

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    wave_meas.append(float(parts[0]))

            wave_meas = np.array(wave_meas)
            num_meas_data = len(wave_meas)

            # Read S/N file
            filename_noise = f"{T_meas}_SS_Noise.txt"

            wave_noise = []

            with open(filename_noise, "r") as f:
                lines = f.readlines()[1:]  # skip header

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    wave_noise.append(float(parts[0]))

            wave_noise = np.array(wave_noise)

            if len(wave_noise) != num_meas_data:
                raise ValueError(
                    "The number of data in the SS and S/N files are not equal"
                )

    return equilibrium,pressure,Tmin,Tmax,H2conc,N2conc,O2conc,COconc,H2Oconc,CO2conc,Heconc,lambdaN,linewidthN,lambdaB,linewidthB,lambdaspect,dispersion,slitwidth,grid_start,grid_end,increment,N_lineshape,NGLvalue,VNG,VNL,B_lineshape,BGLvalue,VBG,VBL,filter_flag,printS,analysis,method,mix,variance,T_meas,shots,printM,T_fluid, NLcm, BLcm, filter_fit, num_grid_points, num_meas_data, wave_meas, int_meas, noise_meas