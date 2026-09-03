""""
In C++ omega_function opens the file every single time it is called. The Python does the same. This works but is very inefficient since it's called 7 times per 
Dicke_linewidth call which itself is called repeatedly. Consider loading the table once at startup and passing it in, but this is a performance issue not a correctness issue.
"""
import math
import numpy as np
from raman_paths import RESOURCE_DIR


# ============================================================
# Doppler Linewidth
# ============================================================

def Doppler_linewidth(T, species):

    if species == 1:
        return 0.03867732672 * math.sqrt(T / 2)
    elif species == 2:
        return 0.03867732672 * math.sqrt(T / 28)
    elif species == 3:
        return 0.03867732672 * math.sqrt(T / 36)
    elif species == 4:
        return 0.03867732672 * math.sqrt(T / 28)
    elif species == 5:
        return 0.03867732672 * math.sqrt(T / 18)
    elif species == 6:
        return 0.03867732672 * math.sqrt(T / 48)

    return 0.0


# ============================================================
# Omega Function (reads omega function.txt)
# ============================================================

def omega_function(value):

    x_tab = []
    y_tab = []

    omega_path = RESOURCE_DIR / "omega function.txt"

    if not omega_path.exists():
        raise FileNotFoundError(
            f"Required Raman file was not found: {omega_path}"
        )

    with omega_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[1:]

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                x_tab.append(float(parts[0]))
                y_tab.append(float(parts[1]))

    x_tab = np.array(x_tab)
    y_tab = np.array(y_tab)

    # bounds
    if value <= 0.3:
        return 2.662

    if value >= 100:
        return 0.517

    # exact match
    idx = np.where(x_tab == value)
    if len(idx[0]) > 0:
        return y_tab[idx[0][0]]

    # interpolate
    i = np.searchsorted(x_tab, value)

    x_min = x_tab[i-1]
    x_max = x_tab[i]
    y_min = y_tab[i-1]
    y_max = y_tab[i]

    y = y_max - (y_max - y_min) * ((x_max - value) / (x_max - x_min))

    return y


# ============================================================
# Dicke Linewidth
# ============================================================

def Dicke_linewidth(
        T, pressure, rho,
        H2conc, H2Oconc, N2conc, O2conc,
        Heconc, COconc, CO2conc):

    P = pressure / 101325
    Temp = T / 273.16

    kteH2  = T / math.sqrt(33.3 * 33.3)
    kteHe  = T / math.sqrt(10.22 * 33.3)
    kteN2  = T / math.sqrt(91.5 * 33.3)
    kteO2  = T / math.sqrt(113 * 33.3)
    kteH2O = T / math.sqrt(356 * 33.3)
    kteCO  = T / math.sqrt(110 * 33.3)
    kteCO2 = T / math.sqrt(190 * 33.3)

    omegaH2  = omega_function(kteH2)
    omegaHe  = omega_function(kteHe)
    omegaN2  = omega_function(kteN2)
    omegaO2  = omega_function(kteO2)
    omegaH2O = omega_function(kteH2O)
    omegaCO  = omega_function(kteCO)
    omegaCO2 = omega_function(kteCO2)

    DH2  = (H2conc  * 0.0018583 * math.sqrt(T**3)) / (omegaH2  * P * 2.968**2)
    DHe  = (Heconc  * 0.0018583 * math.sqrt(0.75 * T**3)) / (omegaHe * P * 2.772**2)
    DN2  = (N2conc  * 0.0018583 * math.sqrt(0.535714285714 * T**3)) / (omegaN2 * P * 3.3245**2)
    DO2  = (O2conc  * 0.0018583 * math.sqrt(0.53125 * T**3)) / (omegaO2 * P * 3.2005**2)
    DH2O = (H2Oconc * 0.0018583 * math.sqrt(0.5555555556 * T**3)) / (omegaH2O * P * 2.8085**2)
    DCO  = (COconc  * 0.0018583 * math.sqrt(0.535714285714 * T**3)) / (omegaCO * P * 3.279**2)
    DCO2 = (CO2conc * 0.0018583 * math.sqrt(0.52272727272 * T**3)) / (omegaCO2 * P * 3.482**2)

    Dtotal = DH2 + DHe + DN2 + DO2 + DH2O + DCO + DCO2

    Doptical = (P * Dtotal) / Temp

    Dicke = (1.38258736286 * Doptical) / rho

    return Dicke


# ============================================================
# Natural Linewidths (non-H2 species)
# ============================================================

def natural_linewidths(
        T, species, pressure, rho,
        H2conc, H2Oconc, N2conc, O2conc,
        Heconc, COconc, CO2conc):

    Doppler = Doppler_linewidth(T, species)

    Dicke = Dicke_linewidth(
        T, pressure, rho,
        H2conc, H2Oconc, N2conc, O2conc,
        Heconc, COconc, CO2conc)

    if Doppler <= Dicke:
        return Doppler * 0.0076729
    else:
        return Dicke * 0.0076729


# ============================================================
# H2 Natural Linewidths
# ============================================================

def H2_natural_linewidths(T, species, pressure, rho, Jmax,
                         H2conc, H2Oconc, N2conc, O2conc,
                         Heconc, COconc, CO2conc,
                         ):

    # Doppler and Dicke linewidths
    Doppler = Doppler_linewidth(T, species)

    Dicke = Dicke_linewidth(
        T, pressure, rho,
        H2conc, H2Oconc, N2conc,
        O2conc, Heconc, COconc, CO2conc
    )

    broaden = np.zeros(Jmax+1)

    # Pressure-collisional broadening
    for j in range(Jmax+1):

        if j == 0:
            broaden[j] = 2*rho * (
                H2conc*(0.00000908*T - 0.00138) +
                N2conc*(0.0000139*T - 0.0011) +
                Heconc*(0.00001171*T - 0.00163) +
                H2Oconc*(0.0000977*T - 0.01484)
            )

        elif j == 1:
            broaden[j] = 2*rho * (
                H2conc*(0.0000050*T - 0.00061) +
                N2conc*(0.0000062*T + 0.0002) +
                Heconc*(0.00000716*T - 0.00039) +
                H2Oconc*(0.000131*T + 0.071 - 0.0056*np.sqrt(T))
            )

        elif j == 2:
            broaden[j] = 2*rho * (
                H2conc*(0.00000548*T - 0.0002) +
                N2conc*(0.0000086*T - 0.0003) +
                Heconc*(0.00000855*T - 0.00031) +
                H2Oconc*(0.0000286*T + 0.0062)
            )

        elif j == 3:
            broaden[j] = 2*rho * (
                H2conc*(0.00000396*T - 0.00098) +
                N2conc*(0.0000079*T - 0.0001) +
                Heconc*(0.00000813*T - 0.00023) +
                H2Oconc*(0.000088*T + 0.0047 - 0.0035*np.sqrt(T))
            )

        elif j == 4:
            broaden[j] = 2*rho * (
                H2conc*(0.00000425*T - 0.0004) +
                N2conc*(0.0000066*T - 0.0004) +
                Heconc*(0.00000816*T - 0.00074) +
                H2Oconc*(0.0000434*T + 0.00177 - 0.00145*np.sqrt(T))
            )

        elif j == 5:
            broaden[j] = 2*rho * (
                H2conc*(0.00000438*T - 0.0006) +
                N2conc*(0.0000065*T - 0.0006) +
                Heconc*(0.00000724*T - 0.00043) +
                H2Oconc*(0.0000623*T + 0.0417 - 0.00295*np.sqrt(T))
            )

        elif j == 6:
            broaden[j] = 2*rho * (
                H2conc*(0.00000438*T - 0.0006) +
                N2conc*(0.000008*T - 0.001) +
                Heconc*(0.00000724*T - 0.00043) +
                H2Oconc*(0.0000623*T + 0.0417 - 0.00295*np.sqrt(T))
            )

        else:
            broaden[j] = 2*rho * (
                H2conc*(0.00000438*T - 0.0006) +
                H2Oconc*(0.0000623*T + 0.0417 - 0.00295*np.sqrt(T))
            )

    broaden_linewidth = np.max(broaden)

    # Determine natural linewidth
    if (broaden_linewidth + Dicke) > Doppler and Dicke > broaden_linewidth:
        natural_linewidth = Doppler * 0.0076729
    else:
        natural_linewidth = (broaden_linewidth + Dicke) * 0.0076729

    return natural_linewidth
# ============================================================
# Raman Linewidth (MAIN FUNCTION USED BY SPECTRUM)
# ============================================================

def Raman_linewidth(
        T, pressure, rho, Jmax, species,
        H2conc, H2Oconc, N2conc, O2conc,
        Heconc, COconc, CO2conc,
        linewidthN, linewidthB,
        lambdaspect, dispersion, slitwidth):

    if species == 1:

        natural_linewidth = H2_natural_linewidths(
            T, species, pressure, rho, Jmax,
            H2conc, H2Oconc, N2conc, O2conc,
            Heconc, COconc, CO2conc)

    else:

        natural_linewidth = natural_linewidths(
            T, species, pressure, rho,
            H2conc, H2Oconc, N2conc, O2conc,
            Heconc, COconc, CO2conc)

    narrow_instrument = math.sqrt(
        lambdaspect**2 +
        linewidthN**2 +
        (dispersion*slitwidth)**2)

    broad_instrument = math.sqrt(
        lambdaspect**2 +
        linewidthB**2 +
        (dispersion*slitwidth)**2)

    if natural_linewidth > narrow_instrument:

        total_narrow = math.sqrt(
            narrow_instrument**2 +
            natural_linewidth**2)

    else:

        total_narrow = narrow_instrument


    if natural_linewidth > broad_instrument:

        total_broad = math.sqrt(
            broad_instrument**2 +
            natural_linewidth**2)

    else:

        total_broad = broad_instrument


    return total_narrow, total_broad, natural_linewidth

def convolution_function(
    T, num_grid_points,num_N_transitions,num_B_transitions,N_lineshape,B_lineshape,
    NGLvalue,BGLvalue,N_resized_wave,N_resized_int,B_resized_wave,B_resized_int,
    plot_x,species,pressure,rho,Jmax,H2conc,H2Oconc,N2conc,O2conc,Heconc,COconc,
    CO2conc,linewidthN,linewidthB,lambdaspect,dispersion,slitwidth,VNG,VNL,VBG,VBL
):

    plot_x = np.asarray(plot_x)

    N_Int_Spectrum = np.zeros_like(plot_x)
    B_Int_Spectrum = np.zeros_like(plot_x)

    # Get linewidths
    total_narrow_linewidth, total_broad_linewidth, natural_linewidth = Raman_linewidth(
        T, pressure, rho, Jmax, species,
        H2conc, H2Oconc, N2conc, O2conc, Heconc, COconc, CO2conc,
        linewidthN, linewidthB, lambdaspect, dispersion, slitwidth
    )

    # Voigt constants
    A = np.array([-1.215, -1.3509, -1.215, -1.3509])
    B = np.array([1.2359, 0.3786, -1.2359, -0.3786])
    C = np.array([-0.3085, 0.5906, -0.3085, 0.5906])
    D = np.array([0.021, -1.1858, -0.0210, 1.1858])

    # ============================
    # Narrowband Convolution
    # ============================

    for i in range(num_grid_points + 1):

        for j in range(num_N_transitions + 1):

            dx = plot_x[i] - N_resized_wave[j]

            # Gaussian
            if N_lineshape == 1:
                N_Int_Spectrum[i] += (
                    N_resized_int[j]
                    * np.exp(-2 * (dx / total_narrow_linewidth) ** 2)
                )

            # Lorentzian
            elif N_lineshape == 2:
                N_Int_Spectrum[i] += (
                    N_resized_int[j]
                    * (total_narrow_linewidth / 2) ** 2
                    / (dx**2 + (total_narrow_linewidth / 2) ** 2)
                )

            # Gaussian-Lorentzian Product
            elif N_lineshape == 3:
                N_Int_Spectrum[i] += (
                    NGLvalue * N_resized_int[j]
                    * np.exp(-2 * (dx / total_narrow_linewidth) ** 2)
                    * (1 - NGLvalue) * N_resized_int[j]
                    * (total_narrow_linewidth / 2) ** 2
                    / (dx**2 + (total_narrow_linewidth / 2) ** 2)
                )

            # Gaussian-Lorentzian Sum
            elif N_lineshape == 4:
                gaussian = np.exp(-2 * (dx / total_narrow_linewidth) ** 2)
                lorentz = (
                    (total_narrow_linewidth / 2) ** 2
                    / (dx**2 + (total_narrow_linewidth / 2) ** 2)
                )

                N_Int_Spectrum[i] += (
                    NGLvalue * N_resized_int[j] * gaussian
                    + (1 - NGLvalue) * N_resized_int[j] * lorentz
                )

            # Voigt
            elif N_lineshape == 5:

                Y = (VNL / VNG) * np.sqrt(np.log(2.0))
                I = (2 / VNG) * np.sqrt(np.log(2.0) / np.pi)

                X = (2 * np.sqrt(np.log(2.0)) * dx) / VNG

                V = 0.0
                for k in range(4):
                    V += (
                        C[k] * (Y - A[k]) + D[k] * (X - B[k])
                    ) / ((Y - A[k]) ** 2 + (X - B[k]) ** 2)

                N_Int_Spectrum[i] += V * N_resized_int[j] * I

            # Natural Lorentzian
            elif N_lineshape == 6:

                N_Int_Spectrum[i] += (
                    N_resized_int[j]
                    * 0.677661253505
                    * (natural_linewidth / 2) ** 2
                    / (dx**2 + (natural_linewidth / 2) ** 2)
                )

            # Natural Voigt
            elif N_lineshape == 7:

                Y = np.sqrt(np.log(2.0))
                I = (2 / natural_linewidth) * np.sqrt(np.log(2.0) / np.pi)

                X = (2 * np.sqrt(np.log(2.0)) * dx) / natural_linewidth

                V = 0.0
                for k in range(4):
                    V += (
                        C[k] * (Y - A[k]) + D[k] * (X - B[k])
                    ) / ((Y - A[k]) ** 2 + (X - B[k]) ** 2)

                N_Int_Spectrum[i] += V * N_resized_int[j] * I


    # ============================
    # Broadband Convolution
    # ============================

    for i in range(num_grid_points + 1):

        for j in range(num_B_transitions + 1):

            dx = plot_x[i] - B_resized_wave[j]

            if B_lineshape == 1:

                B_Int_Spectrum[i] += (
                    B_resized_int[j]
                    * np.exp(-2 * (dx / total_broad_linewidth) ** 2)
                )

            elif B_lineshape == 2:
                B_Int_Spectrum[i] += (
                    B_resized_int[j]
                    * 0.677661253505
                    * (total_broad_linewidth / 2) ** 2
                    / (dx**2 + (total_broad_linewidth / 2) ** 2)
                )

            elif B_lineshape == 3:

                B_Int_Spectrum[i] += (
                    BGLvalue * B_resized_int[j]
                    * np.exp(-2 * (dx / total_broad_linewidth) ** 2)
                    * (1 - BGLvalue) * B_resized_int[j]
                    * (total_broad_linewidth / 2) ** 2
                    / (dx**2 + (total_broad_linewidth / 2) ** 2)
                )

            elif B_lineshape == 4:

                gaussian = np.exp(-2 * (dx / total_broad_linewidth) ** 2)
                lorentz = (0.677661253505 * (total_broad_linewidth/2)**2
                        / (dx**2 + (total_broad_linewidth/2)**2))
                + (1 - BGLvalue) * B_resized_int[j] * lorentz
                #Check This

                B_Int_Spectrum[i] += (
                    BGLvalue * B_resized_int[j] * gaussian
                    + (1 - BGLvalue) * B_resized_int[j] * lorentz
                )

            elif B_lineshape == 5:

                Y = (VBL / VBG) * np.sqrt(np.log(2.0))
                I = (2 / VBG) * np.sqrt(np.log(2.0) / np.pi)

                X = (2 * np.sqrt(np.log(2.0)) * dx) / VBG

                V = 0.0
                for k in range(4):
                    V += (
                        C[k] * (Y - A[k]) + D[k] * (X - B[k])
                    ) / ((Y - A[k]) ** 2 + (X - B[k]) ** 2)

                B_Int_Spectrum[i] += V * B_resized_int[j] * I

            elif B_lineshape == 6:

                B_Int_Spectrum[i] += (
                    B_resized_int[j]
                    * 0.677661253505
                    * (natural_linewidth / 2) ** 2
                    / (dx**2 + (natural_linewidth / 2) ** 2)
                )

            elif B_lineshape == 7:

                Y = np.sqrt(np.log(2.0))
                I = (2 / natural_linewidth) * np.sqrt(np.log(2.0) / np.pi)

                X = (2 * np.sqrt(np.log(2.0)) * dx) / natural_linewidth

                V = 0.0
                for k in range(4):
                    V += (
                        C[k] * (Y - A[k]) + D[k] * (X - B[k])
                    ) / ((Y - A[k]) ** 2 + (X - B[k]) ** 2)

                B_Int_Spectrum[i] += V * B_resized_int[j] * I


    # ============================
    # Normalize
    # ============================

    N_Int_Spectrum /= np.max(N_Int_Spectrum)
    B_Int_Spectrum /= np.max(B_Int_Spectrum)


    print("N spectrum max:", np.max(N_Int_Spectrum))
    print("B spectrum max:", np.max(B_Int_Spectrum))
    return N_Int_Spectrum, B_Int_Spectrum
'''
    if printS == 0:
        filename = f"{int(T)}_{species_name}_spectrum.txt"
        with open(filename, "w") as f:
            f.write("Narrowband\tNarrowband\tBroadband\n")
            f.write("Wavelength\tIntensity\tIntensity\n")
            f.write("  (nm)   \t   (-)   \t   (-)   \n")
            for j in range(num_grid_points + 1):
                f.write(f"{plot_x[j]:.10f}\t{N_Int_Spectrum[j]:.10e}\t{B_Int_Spectrum[j]:.10e}\n")
'''

