import numpy as np
from scipy.constants import c, h, k
from numpy.polynomial.polynomial import polyfit, polyval

#Integer and Double Error Functions 
''' Removed for now likely not needed
def ierrorfunction(variable, a, b, name="variable"):
    if not (a <= variable <= b):
        raise ValueError(f"{name} out of range ({a}-{b})")
    return variable


def derrorfunction(variable, a, b, name="variable"):
    if not (a <= variable <= b):
        raise ValueError(f"{name} out of range ({a}-{b})")
    return variable
'''
#Polyfit might not be needed but here for now to match C++ code structure CAn be replaced with just a call to np.polyfit in the main code
def poly_fit(y, x, order=6): #may need to play around with order
    coeffs = np.polyfit(x, y, order)
    return coeffs[::-1]  # match C++ ascending order *** could also be changed

#Dont need background_pixels (error function)
#Dont need background_wave (error function)
#Calculate the Pixel Wavelengths
#Function will convert from pixel number to wavelength using calibration file a nd the corrects for air to vacuum wavelength
#Does not write out to file might need to add
def pixel_wavelength_conversion(pixel, calibration_file): # Caibration file = Spectrograph_Calibration.txt <-- need to be adaoted
    data = np.loadtxt(calibration_file, skiprows=2)
    x = data[:, 0]
    y = data[:, 1]

    coeffs = np.polyfit(x, y, 3) #polynomial fit to calibration data

    wave = np.polyval(coeffs, pixel) #converts measured pixels into wavelengths

    # Index of refraction correction (air to vacuum wavelengths)
    index_n = 1 + 1e-7 * (
        2726.43 +
        (12.288 / ((1e-8) * (wave * 10)**2)) +
        (0.3555 / ((1e-16) * (wave * 10)**4))
    )

    return wave * index_n

#Background Subtraction Function
#Assumes raman peaks are localized and background is smooth -> use this to create a straight line for locations with no raman peaks and subtract from entire spectrum
def background_subtraction(pixel, intensity,
                           left_start, left_end,
                           right_start, right_end):

    mask = ((pixel >= left_start) & (pixel <= left_end)) | \
           ((pixel >= right_start) & (pixel <= right_end))

    bg_pixels = pixel[mask]
    bg_intensity = intensity[mask]

    coeffs = np.polyfit(bg_pixels, bg_intensity, 1)
    background = np.polyval(coeffs, pixel)

    ints_meas = intensity - np.abs(background)
    ints_meas[ints_meas < 0] = 0

    return ints_meas

#uncertainty_bckg_subtraction? Skipped for now will have to add later as uncertainty is better understood

	# //*************************************************************************************
	# //the equation of state is given by the Redlich-Kwong method
	# //P-((8314*T)/(v-mixtureB))+(mixtureA/(v*(v+mixtureB)*T^0.5))=0
	# //the coefficients for mixture A and B are calculated from Redlich-Kwong data
	# //A=(0.42748*(R^2*Tc^2.5)/Pc)^0.5 and B=(0.08664*(R*Tc)/Pc)
	# //the state equation in reduced form is equivalent to a 3rd order polynomial
	# //((P*T^0.5)*v^3)-((8314*T^1.5)*v^2)-((8314*T^1.5)*b+((P*T^0.5)*b^2)-a)*v)-a*b)=0
	# // or (A*v^3)-(B*v^2)-C*v-D=0 in cubic form is (v^3)-(E*v^2)-F*v-G=0 (Mathematica)
	# //*************************************************************************************

def real_gas(T, pressure, H2, N2, H2O, O2, He, CO, CO2):
    r = 83144606.62  # GSL_CONST_CGS_MOLAR_GAS in erg/(mol·K)
    v = 22413.996    # GSL_CONST_CGS_STANDARD_GAS_VOLUME in cm³/mol
    R = r / 10000    # = 8314.46
    V = 1 / (v / 1000)  # = 44.62

    mixtureA = (379.94341684*H2 + 1312.26978933*O2 + 1246.32499774*N2 +
                3776.17465168*H2O + 89.0056178002*He +
                1311.187*CO + 2037.444*CO2) ** 2

    mixtureB = (0.018396*H2 + 0.021966*O2 + 0.026773*N2 +
                0.021108*H2O + 0.016286*He +
                0.02773533*CO + 0.029683363*CO2)

    A = (R * T) / pressure
    B = ((pressure * mixtureB**2 * T**0.5) +
         (R * mixtureB * T**1.5) - mixtureA) / (pressure * T**0.5)
    C = (mixtureA * mixtureB) / (pressure * T**0.5)

    D = 3*B + A**2
    E = D**3
    F = 9*A*B + 27*C + 2*A**3
    G = F**2
    H_val = (max(G - 4*E, 0))**0.5

    real_V = (A/3
              + (1.25992104989 * D) / (3 * (F + H_val)**0.333333333333)
              + 0.264566841995 * (F + H_val)**0.333333333333)

    amagats = (1 / real_V) / V
    return amagats

def Q_vib(T, vib):
    hck = (h * c) / k
    vib = np.array(vib)
    return np.sum(np.exp(-(hck * vib) / T))

#Qrot and Q_H2 rot joined into one 
def Q_rot(T, Vmax, Jmax, rovib, g_odd, g_even):
    hck = (h * c) / k
    Q = 0.0
    for v in range(Vmax + 1):
        for j in range(Jmax + 1):
            g = g_even if j % 2 == 0 else g_odd
            Q += g * (2*j + 1) * np.exp(-(hck * rovib[v, j]) / T)
    return Q

#changed to Filter instead of apply filter
def Filter(filter_fit,N_resized_wave,N_resized_int,B_resized_wave,B_resized_int
):
    """
    Python equivalent of C++ Filter() function.
    Applies spectral filter correction to narrowband and broadband spectra.
    """

    # Convert to numpy arrays (safe if already arrays)
    N_resized_wave = np.array(N_resized_wave, dtype=float)
    N_resized_int = np.array(N_resized_int, dtype=float)

    B_resized_wave = np.array(B_resized_wave, dtype=float)
    B_resized_int = np.array(B_resized_int, dtype=float)

    # ----------------------------
    # NARROWBAND FILTER
    # ----------------------------

    # Evaluate polynomial manually (matches C++ structure exactly)
    filter_N = (
        filter_fit[6] * N_resized_wave**6 +
        filter_fit[5] * N_resized_wave**5 +
        filter_fit[4] * N_resized_wave**4 +
        filter_fit[3] * N_resized_wave**3 +
        filter_fit[2] * N_resized_wave**2 +
        filter_fit[1] * N_resized_wave +
        filter_fit[0]
    )

    # Normalize by minimum absorbance
    min_N_absorbance = np.min(filter_N)
    filter_N = filter_N / min_N_absorbance

    # Apply correction
    N_resized_int = N_resized_int / filter_N

    # Renormalize intensity to max
    max_N = np.max(N_resized_int)
    if max_N != 0:
        N_resized_int = N_resized_int / max_N

    # ----------------------------
    # BROADBAND FILTER
    # ----------------------------

    filter_B = (
        filter_fit[6] * B_resized_wave**6 +
        filter_fit[5] * B_resized_wave**5 +
        filter_fit[4] * B_resized_wave**4 +
        filter_fit[3] * B_resized_wave**3 +
        filter_fit[2] * B_resized_wave**2 +
        filter_fit[1] * B_resized_wave +
        filter_fit[0]
    )

    min_B_absorbance = np.min(filter_B)
    filter_B = filter_B / min_B_absorbance

    B_resized_int = B_resized_int / filter_B

    max_B = np.max(B_resized_int)
    if max_B != 0:
        B_resized_int = B_resized_int / max_B

    #return N_resized_int, B_resized_int
    return N_resized_wave, N_resized_int, B_resized_wave, B_resized_int

def pibub(x, y):
    if len(x) == 0:
        return np.array([]), np.array([])
    pairs = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*pairs)
    return np.array(x_sorted), np.array(y_sorted)
