import numpy as np

from Generic_Raman_Functions import real_gas
from Generic_Raman_Functions import Q_vib
from Generic_Raman_Functions import Q_rot
from Generic_Raman_Functions import Filter
from Generic_Raman_Functions import pibub

C_CGS = 2.99792458e10        # cm/s
H_CGS = 6.62607015e-27       # erg*s
K_CGS = 1.380649e-16         # erg/K
HCK = (H_CGS * C_CGS) / K_CGS

# Why does everytong retunr rho and it all calls the same function will have to check that when done!!!
# also organize and remove chatgpt comments


def H2_intensities(
    T, pressure,
    H2conc, N2conc, H2Oconc, O2conc, COconc, CO2conc, Heconc,
    Vmax, Jmax,
    NLcm, BLcm,
    vib, Ramanvib,
    rovib, RamanQ, RamanO, RamanS,
    alpha, gamma, Be,
    T_fluid, intermediate_state,
    conc, int_factor,
    grid_start, grid_end,
    filter_on,
    filter_fit):

    # -----------------------------
    # Real gas density
    # -----------------------------
    rho = real_gas(
        T, pressure,
        H2conc, N2conc, H2Oconc,
        O2conc, Heconc, COconc, CO2conc
    )

    # -----------------------------
    # Collisional shifts
    # -----------------------------
    vibcollision_shift = np.zeros(Vmax+1)
    rotcollision_shift = np.zeros(Jmax+1)

    for v in range(Vmax+1):
        vibcollision_shift[v] = 0.001 * rho * (
            H2conc*((0.7074*np.sqrt(T))-14.44) +
            N2conc*((0.8*np.sqrt(T))-23.2) +
            H2Oconc*((1.08*np.sqrt(T))-65.0) +
            Heconc*((0.63*np.sqrt(T))-1.52)
        )

    ##Add cases 1-7...!!!! check the values # added

    for j in range(Jmax+1):
        if j == 0:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.7074*np.sqrt(T))-14.44))
                                                +(N2conc*((0.8*np.sqrt(T))-23.2))
                                                +(H2Oconc*((1.08*np.sqrt(T))-65.0))
                                                +(Heconc*((0.63*np.sqrt(T))-1.52)))
        if j == 1:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.8044*np.sqrt(T))-17.04))
                                                +(N2conc*((1.08*np.sqrt(T))-27.2))
                                                +(H2Oconc*((0.79*np.sqrt(T))-52.4))
                                                +(Heconc*((0.8*np.sqrt(T))-4.05)))
        if j == 2:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.7615*np.sqrt(T))-15.07))
                                                +(N2conc*((1.12*np.sqrt(T))-27.4))
                                                +(H2Oconc*((0.357*np.sqrt(T))-30.1))
                                                +(Heconc*((0.82*np.sqrt(T))-4.24)))
        if j == 3:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.7414*np.sqrt(T))-14.73))
                                                +(N2conc*((1.14*np.sqrt(T))-27.4))
                                                +(H2Oconc*((0.374*np.sqrt(T))+28.95))
                                                +(Heconc*((0.86*np.sqrt(T))-4.87)))
        if j == 4:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.7775*np.sqrt(T))-14.96))
                                                +(N2conc*((1.23*np.sqrt(T))-28.9))
                                                +(H2Oconc*((0.49*np.sqrt(T))+32.0))
                                                +(Heconc*((0.9*np.sqrt(T))-5.33)))
        if j == 5:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.7797*np.sqrt(T))-14.87))
                                                +(N2conc*((1.27*np.sqrt(T))-29.8))
                                                +(H2Oconc*((0.75*np.sqrt(T))+39.1))
                                                +(Heconc*((0.93*np.sqrt(T))-5.8)))
        if j == 6:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.78*np.sqrt(T))-14.87))
                                                +(N2conc*((1.27*np.sqrt(T))-29.9))
                                                +(H2Oconc*((0.75*np.sqrt(T))+39.1))
                                                +(Heconc*((0.93*np.sqrt(T))-5.8)))
        if j >= 7:
            rotcollision_shift[j] = 0.001*rho*((H2conc*((0.78*np.sqrt(T))-14.87))
                                                +(N2conc*((1.3*np.sqrt(T))-29.9))
                                                +(H2Oconc*((0.75*np.sqrt(T))+39.1))
                                                +(Heconc*((0.93*np.sqrt(T))-5.8)))

    # -----------------------------
    # Raman shifts with collisions
    # -----------------------------
    VIB_Raman_shift = Ramanvib[:Vmax] + vibcollision_shift[:Vmax]

    Raman_shift_Q = RamanQ[:Vmax, :Jmax+1] + rotcollision_shift
    Raman_shift_O = np.zeros((Vmax, Jmax+1))
    Raman_shift_S = np.zeros((Vmax, Jmax+1))

    # O branch valid for j = 2 → Jmax
    Raman_shift_O[:, 2:Jmax+1] = (
        RamanO[:Vmax, 2:Jmax+1]
        + rotcollision_shift[2:Jmax+1]
    )

    # S branch valid for j = 0 → Jmax-2
    Raman_shift_S[:, :Jmax-1] = (
        RamanS[:Vmax, :Jmax-1]
        + rotcollision_shift[:Jmax-1]
    )
        # -----------------------------
    # Frequencies and wavelengths
    # -----------------------------

   # added O and S cases 
    NRamanPositionQ = np.zeros((Vmax, Jmax+1))
    NRamanPositionO = np.zeros((Vmax, Jmax+1))
    NRamanPositionS = np.zeros((Vmax, Jmax+1))

    BRamanPositionQ = np.zeros((Vmax, Jmax+1))
    BRamanPositionO = np.zeros((Vmax, Jmax+1))
    BRamanPositionS = np.zeros((Vmax, Jmax+1))

    N_RamanQ_Wave = np.zeros((Vmax, Jmax+1))
    N_RamanO_Wave = np.zeros((Vmax, Jmax+1))
    N_RamanS_Wave = np.zeros((Vmax, Jmax+1))

    B_RamanQ_Wave = np.zeros((Vmax, Jmax+1))
    B_RamanO_Wave = np.zeros((Vmax, Jmax+1))
    B_RamanS_Wave = np.zeros((Vmax, Jmax+1))
    
    for v in range(Vmax):
        # ------------------
        # Q BRANCH (j = 0 → Jmax)
        # ------------------
        for j in range(Jmax+1):
            NRamanPositionQ[v][j] = NLcm - Raman_shift_Q[v][j]
            N_RamanQ_Wave[v][j] = (1.0 / NRamanPositionQ[v][j]) * 1e7

            BRamanPositionQ[v][j] = BLcm - Raman_shift_Q[v][j]
            B_RamanQ_Wave[v][j] = (1.0 / BRamanPositionQ[v][j]) * 1e7


        # ------------------
        # O BRANCH (j = 2 → Jmax)
        # ------------------

        for j in range(2, Jmax+1):
            NRamanPositionO[v][j] = NLcm - Raman_shift_O[v][j]
            N_RamanO_Wave[v][j] = (1.0 / NRamanPositionO[v][j]) * 1e7

            BRamanPositionO[v][j] = BLcm - Raman_shift_O[v][j]
            B_RamanO_Wave[v][j] = (1.0 / BRamanPositionO[v][j]) * 1e7


        # ------------------
        # S BRANCH (j = 0 → Jmax-2)
        # ------------------
        for j in range(Jmax-1):
            NRamanPositionS[v][j] = NLcm - Raman_shift_S[v][j]
            N_RamanS_Wave[v][j] = (1.0 / NRamanPositionS[v][j]) * 1e7

            BRamanPositionS[v][j] = BLcm - Raman_shift_S[v][j]
            B_RamanS_Wave[v][j] = (1.0 / BRamanPositionS[v][j]) * 1e7

    # -----------------------------
    # Plazcek-Teller coefficients
    # -----------------------------
    PQN = np.zeros(Jmax+1)
    PQB = np.zeros(Jmax+1)
    PON = np.zeros(Jmax+1)
    POB = np.zeros(Jmax+1)
    PSN = np.zeros(Jmax+1)
    PSB = np.zeros(Jmax+1)

    for j in range(Jmax+1):
        PQN[j] = alpha + (7*(j*(j+1))*gamma)/(45*(2*j-1)*(2*j+3))
        PQB[j] = 0.5*alpha + (13*(j*(j+1))*gamma)/(90*(2*j-1)*(2*j+3))

        if j == 0:
            PQN[j] = 2.1508e-33
            PQB[j] = 0.5 * 2.1508e-33
        if j >= 2:
            PON[j] = (7*3*j*(j-1)*gamma)/(45*2*(2*j+1)*(2*j-1))
            POB[j] = (13*3*j*(j-1)*gamma)/(90*2*(2*j+1)*(2*j-1))

        PSN[j] = (7*3*(j+1)*(j+2)*gamma)/(45*2*(2*j+1)*(2*j+3))
        PSB[j] = (13*3*(j+1)*(j+2)*gamma)/(90*2*(2*j+1)*(2*j+3))

    # -----------------------------
    # Nuclear spin statistics
    # -----------------------------
    h = 6.62607015e-27
    c = 2.99792458e10
    k = 1.380649e-16
    hck = (h*c)/k
    # appears twice
    theta_r = (Be*hck)/T_fluid

    H2_para = 0.0
    H2_ortho = 0.0

    #original calls on loop for V not sure why
    for j in range(Jmax+1):
        if j % 2 == 0:
            H2_para += (2*j+1)*np.exp(-theta_r*j*(j+1))
        else:
            H2_ortho += 3*(2*j+1)*np.exp(-theta_r*j*(j+1))

    spin_ratio = H2_ortho/H2_para
    if T_fluid >= 250:
        spin_ratio = 3.0

    g_even = 1.0
    g_odd = spin_ratio

    # -----------------------------
    # Partition function
    # -----------------------------
    #Originally calls Q_H2_rot but this was joined with Q_rot in Generic_Raman_Functions.py so just call Q_rot hopefully fine
    Qrovib_tot =Q_rot(T, Vmax, Jmax, rovib, g_odd, g_even) #removing for now Need to Fix something here and in generic raman function 
    Qvib_tot=Q_vib(T,vib) #added this but not called 
    Q_equilibrium = Qrovib_tot

    # -----------------------------
    # Raman intensities
    # -----------------------------
    N_Intensity_Q = np.zeros((Vmax, Jmax+1))
    B_Intensity_Q = np.zeros((Vmax, Jmax+1))
    N_Intensity_O = np.zeros((Vmax, Jmax+1))
    B_Intensity_O = np.zeros((Vmax, Jmax+1))
    N_Intensity_S = np.zeros((Vmax, Jmax+1))
    B_Intensity_S = np.zeros((Vmax, Jmax+1))

    for v in range(Vmax):

        # Q branch: lower state j -> upper state j
        for j in range(Jmax+1):
            boltz = np.exp(-(HCK * rovib[v][j]) / T)
            g = g_even if j % 2 == 0 else g_odd

            N_Intensity_Q[v, j] = (
                int_factor * conc *
                NRamanPositionQ[v, j]**4 *
                (v + 1) * PQN[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_Q[v, j] = (
                int_factor * conc *
                BRamanPositionQ[v, j]**4 *
                (v + 1) * PQB[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

        # O branch: lower state j -> upper state j-2
        for j in range(2, Jmax+1):
            boltz = np.exp(-(HCK * rovib[v][j]) / T)
            g = g_even if j % 2 == 0 else g_odd

            N_Intensity_O[v, j] = (
                int_factor * conc *
                NRamanPositionO[v, j]**4 *
                (v + 1) * PON[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_O[v, j] = (
                int_factor * conc *
                BRamanPositionO[v, j]**4 *
                (v + 1) * POB[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

        # S branch: lower state j -> upper state j+2
        for j in range(Jmax-1):
            boltz = np.exp(-(HCK * rovib[v][j]) / T)
            g = g_even if j % 2 == 0 else g_odd

            N_Intensity_S[v, j] = (
                int_factor * conc *
                NRamanPositionS[v, j]**4 *
                (v + 1) * PSN[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_S[v, j] = (
                int_factor * conc *
                BRamanPositionS[v, j]**4 *
                (v + 1) * PSB[j] * g *
                (2 * j + 1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])


        # -----------------------------
        # Normalize by max Q branch
        # -----------------------------
    max_N = np.max(N_Intensity_Q)
    max_B = np.max(B_Intensity_Q)


# ""_Intensity_Q in original
    N_norm_Q = N_Intensity_Q / max_N
    B_norm_Q = B_Intensity_Q / max_B
# Added for O and S branches!!!!!!!
    N_norm_O = N_Intensity_O / max_N
    B_norm_O = B_Intensity_O / max_B

    N_norm_S = N_Intensity_S / max_N
    B_norm_S = B_Intensity_S / max_B

    # -----------------------------
    # Flatten + limit transitions
    # -----------------------------
    # ============================
    # NARROWBAND
    # ============================

    N_wave = []
    N_int  = []

    # --- O branch ---
    for v in range(Vmax):
        for j in range(2, Jmax+1):
            w = N_RamanO_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                N_wave.append(w)
                N_int.append(N_norm_O[v][j])

    # --- Q branch ---
    for v in range(Vmax):
        for j in range(Jmax+1):
            w = N_RamanQ_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                N_wave.append(w)
                N_int.append(N_norm_Q[v][j])

    # --- S branch ---
    for v in range(Vmax):
        for j in range(Jmax-1):
            w = N_RamanS_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                N_wave.append(w)
                N_int.append(N_norm_S[v][j])

    # Convert to numpy
    N_wave = np.array(N_wave)
    N_int  = np.array(N_int)

    # --- Limit weak transitions ---
    mask = N_int >= 1e-6
    N_wave = N_wave[mask]
    N_int  = N_int[mask]

    # --- Sort (replaces pibub) ---
    idx = np.argsort(N_wave)
    N_wave = N_wave[idx]
    N_int  = N_int[idx]

    num_N_transitions = len(N_wave) - 1


    # ============================
    # BROADBAND
    # ============================

    B_wave = []
    B_int  = []

    # --- O branch ---
    for v in range(Vmax):
        for j in range(2, Jmax+1):
            w = B_RamanO_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                B_wave.append(w)
                B_int.append(B_norm_O[v][j])

    # --- Q branch ---
    for v in range(Vmax):
        for j in range(Jmax+1):
            w = B_RamanQ_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                B_wave.append(w)
                B_int.append(B_norm_Q[v][j])

    # --- S branch ---
    for v in range(Vmax):
        for j in range(Jmax-1):
            w = B_RamanS_Wave[v][j]
            if (grid_start-0.3) < w < (grid_end+0.3):
                B_wave.append(w)
                B_int.append(B_norm_S[v][j])

    B_wave = np.array(B_wave)
    B_int  = np.array(B_int)

    # --- Limit weak transitions ---
    mask = B_int >= 1e-6
    B_wave = B_wave[mask]
    B_int  = B_int[mask]

    # --- Sort ---
    idx = np.argsort(B_wave)
    B_wave = B_wave[idx]
    B_int  = B_int[idx]

    num_B_transitions = len(B_wave) - 1


    # ============================
    # Spectral Filter
    # ============================

    if filter_on == 0:
        N_wave, N_int, B_wave, B_int = Filter(
            filter_fit,N_wave, N_int,B_wave, B_int
        )

    return N_wave, N_int, num_N_transitions, B_wave, B_int, num_B_transitions,rho



def N2_O2_CO_Int(T, Vmax, Jmax, NLcm, BLcm,
                  Raman_shift_Q, Raman_shift_O, Raman_shift_S,
                  alpha, gamma, species,
                  Be, T_fluid, vib, rovib,
                  intermediate_state, conc, int_factor,
                  VIB_Raman_shift,
                  grid_start, grid_end,
                  pressure, H2conc, N2conc, O2conc,
                  COconc, CO2conc, H2Oconc, Heconc,
                  filter_on, filter_fit): #for some reason set to =false???? wierd error/ added filter fit need to check this
    rho = real_gas(T, pressure, H2conc, N2conc,
                   H2Oconc, O2conc, Heconc,
                   COconc, CO2conc)
 # Not sure why this stuff was not inncluded
    NRamanPositionQ = np.zeros((Vmax, Jmax+1))
    NRamanPositionO = np.zeros((Vmax, Jmax+1))
    NRamanPositionS = np.zeros((Vmax, Jmax+1))

    BRamanPositionQ = np.zeros((Vmax, Jmax+1))
    BRamanPositionO = np.zeros((Vmax, Jmax+1))
    BRamanPositionS = np.zeros((Vmax, Jmax+1))

    N_RamanQ_Wave = np.zeros((Vmax, Jmax+1))
    N_RamanO_Wave = np.zeros((Vmax, Jmax+1))
    N_RamanS_Wave = np.zeros((Vmax, Jmax+1))

    B_RamanQ_Wave = np.zeros((Vmax, Jmax+1))
    B_RamanO_Wave = np.zeros((Vmax, Jmax+1))
    B_RamanS_Wave = np.zeros((Vmax, Jmax+1))
    
    for v in range(Vmax):
        # ------------------
        # Q BRANCH (j = 0 → Jmax)
        # ------------------
        for j in range(Jmax+1):
            NRamanPositionQ[v][j] = NLcm - Raman_shift_Q[v][j]
            N_RamanQ_Wave[v][j] = (1.0 / NRamanPositionQ[v][j]) * 1e7

            BRamanPositionQ[v][j] = BLcm - Raman_shift_Q[v][j]
            B_RamanQ_Wave[v][j] = (1.0 / BRamanPositionQ[v][j]) * 1e7


        # ------------------
        # O BRANCH (j = 2 → Jmax)
        # ------------------
        for j in range(2, Jmax+1):
            NRamanPositionO[v][j] = NLcm - Raman_shift_O[v][j]
            N_RamanO_Wave[v][j] = (1.0 / NRamanPositionO[v][j]) * 1e7

            BRamanPositionO[v][j] = BLcm - Raman_shift_O[v][j]
            B_RamanO_Wave[v][j] = (1.0 / BRamanPositionO[v][j]) * 1e7


        # ------------------
        # S BRANCH (j = 0 → Jmax-2)
        # ------------------
        for j in range(Jmax-1):
            NRamanPositionS[v][j] = NLcm - Raman_shift_S[v][j]
            N_RamanS_Wave[v][j] = (1.0 / NRamanPositionS[v][j]) * 1e7

            BRamanPositionS[v][j] = BLcm - Raman_shift_S[v][j]
            B_RamanS_Wave[v][j] = (1.0 / BRamanPositionS[v][j]) * 1e7

    PQN = np.zeros(Jmax+1)
    PQB = np.zeros(Jmax+1)
    PON = np.zeros(Jmax+1)
    POB = np.zeros(Jmax+1)
    PSN = np.zeros(Jmax+1)
    PSB = np.zeros(Jmax+1)
    for j in range(Jmax+1):
        PQN[j] = alpha + (7*(j*(j+1))*gamma)/(45*(2*j-1)*(2*j+3))
        PQB[j] = 0.5*alpha + (13*(j*(j+1))*gamma)/(90*(2*j-1)*(2*j+3))
        PON[j] = (7*3*j*(j-1)*gamma)/(45*2*(2*j+1)*(2*j-1))
        POB[j] = (13*3*j*(j-1)*gamma)/(90*2*(2*j+1)*(2*j-1))
        PSN[j] = (7*3*(j+1)*(j+2)*gamma)/(45*2*(2*j+1)*(2*j+3))
        PSB[j] = (13*3*(j+1)*(j+2)*gamma)/(90*2*(2*j+1)*(2*j+3))    


    # --- Raman wavelengths ---
    N_wave = []
    N_int  = []
    B_wave = []
    B_int  = []
        # -----------------------------
    # Frequencies and wavelengths
    # -----------------------------

  

    # --- Spin statistics ---
    if species == 2:  # N2
        theta_r = (Be * HCK) / T_fluid
        N2_para = 0
        N2_ortho = 0
        for j in range(Jmax+1):
            if j % 2 == 0:
                N2_para += 6*(2*j+1)*np.exp(-theta_r*j*(j+1))
            else:
                N2_ortho += 3*(2*j+1)*np.exp(-theta_r*j*(j+1))
        spin_ratio = N2_ortho / N2_para
        g_even = 1
        g_odd = spin_ratio

    elif species == 3:  # O2
        g_even = 0
        g_odd = 1

    elif species == 4:  # CO
        g_even = 1
        g_odd = 1

    # --- Partition function ---
    Qvib = Q_vib(T, vib) # removed Vmax
    Qrovib = Q_rot(T, Vmax, Jmax, rovib, g_odd, g_even)
    Q_equilibrium = Qrovib

    # --- Resonance factor ---
    if species > 1:
        resonanceN = 1 / ((intermediate_state**2 - NLcm**2)**2)
        resonanceB = 1 / ((intermediate_state**2 - BLcm**2)**2)

    N_Intensity_Q = np.zeros((Vmax, Jmax+1))
    B_Intensity_Q = np.zeros((Vmax, Jmax+1))
    N_Intensity_O = np.zeros((Vmax, Jmax+1))
    B_Intensity_O = np.zeros((Vmax, Jmax+1))
    N_Intensity_S = np.zeros((Vmax, Jmax+1))
    B_Intensity_S = np.zeros((Vmax, Jmax+1))


    for v in range(Vmax):
        for j in range(Jmax+1):
            boltz = np.exp(-(HCK * rovib[v][j]) / T)
            g = g_even if j % 2 == 0 else g_odd
            
            N_Intensity_Q[v,j] = (
                int_factor * conc *
                NRamanPositionQ[v,j]**4 *
                (v+1) * PQN[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_Q[v,j] = (
                int_factor * conc *
                BRamanPositionQ[v,j]**4 *
                (v+1) * PQB[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])
    #Added O and S branch!!!!!!!
        for j in range(2, Jmax+1):
            g = g_even if j % 2 == 0 else g_odd

            N_Intensity_O[v,j] = (
                int_factor * conc *
                NRamanPositionO[v,j]**4 *
                (v+1) * PON[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_O[v,j] = (
                int_factor * conc *
                BRamanPositionO[v,j]**4 *
                (v+1) * POB[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

        for j in range(Jmax-1):
            g = g_even if j % 2 == 0 else g_odd

            N_Intensity_S[v,j] = (
                int_factor * conc *
                NRamanPositionS[v,j]**4 *
                (v+1) * PSN[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

            B_Intensity_S[v,j] = (
                int_factor * conc *
                BRamanPositionS[v,j]**4 *
                (v+1) * PSB[j] * g *
                (2*j+1) * boltz
            ) / (Q_equilibrium * VIB_Raman_shift[v])

        # -----------------------------
        # Normalize by max Q branch
        # -----------------------------
    max_N = np.max(N_Intensity_Q)
    max_B = np.max(B_Intensity_Q)


# ""_Intensity_Q in original
    N_norm_Q = N_Intensity_Q / max_N
    B_norm_Q = B_Intensity_Q / max_B
# Added for O and S branches!!!!!!!
    N_norm_O = N_Intensity_O / max_N
    B_norm_O = B_Intensity_O / max_N

    N_norm_S = N_Intensity_S / max_N
    B_norm_S = B_Intensity_S / max_N

        # -----------------------------
        # Flatten + limit transitions
        # -----------------------------

    
    ###
    # Sort and Limit the Number of Transitions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # --- initialize indexes ---
    start_N = 0
    append_N = 0
    end_N = 0

    start_B = 0
    append_B = 0
    end_B = 0

    # --- initialize appended arrays (equivalent to new double[5250]) ---
    N_appended_wave = np.zeros(5250)
    N_appended_int  = np.zeros(5250)

    B_appended_wave = np.zeros(5250)
    B_appended_int  = np.zeros(5250)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # NARROWBAND RAMAN SPECTRA
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # --- O branch ---
    for v in range(Vmax):
        for j in range(2, Jmax+1):
            if ((grid_start - 0.3) < N_RamanO_Wave[v][j] < (grid_end + 0.3)):
                #N_appended_wave[start_N] = N_RamanO_Wave[v][j]
                N_appended_int[start_N]  = N_norm_O[v][j]
                start_N += 1


    # --- Q branch ---
    append_N = 0
    for v in range(Vmax):
        for j in range(Jmax+1):
            if ((grid_start - 0.3) < N_RamanQ_Wave[v][j] < (grid_end + 0.3)):
                N_appended_wave[start_N + append_N] = N_RamanQ_Wave[v][j]
                N_appended_int[start_N + append_N]  = N_norm_Q[v][j]
                append_N += 1

    start_N = start_N + append_N - 1


    # --- S branch ---
    end_N = 0
    for v in range(Vmax):
        for j in range(Jmax-1):
            if ((grid_start - 0.3) < N_RamanS_Wave[v][j] < (grid_end + 0.3)):
                N_appended_wave[start_N + end_N] = N_RamanS_Wave[v][j]
                N_appended_int[start_N + end_N]  = N_norm_S[v][j]
                end_N += 1

    start_N = start_N + end_N - 1

    # --- total number of transitions ---
    num_transitions_N = start_N


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Limit weak narrowband transitions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    N_resized_wave = np.zeros(num_transitions_N + 1)
    N_resized_int  = np.zeros(num_transitions_N + 1)

    number_N = 0

    for i in range(num_transitions_N + 1):

        if N_appended_int[i] >= 1e-6:

            N_resized_wave[number_N] = N_appended_wave[i]
            N_resized_int[number_N]  = N_appended_int[i]

            number_N += 1


    # trim to actual size
    N_resized_wave = N_resized_wave[:number_N]
    N_resized_int  = N_resized_int[:number_N]


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sort and Limit the Number of Transitions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ============================
    # NARROWBAND RAMAN SPECTRA
    # ============================

    N_wave = []
    N_int  = []

    # --- O branch ---
    for v in range(Vmax):
        for j in range(2, Jmax+1):

            wave = N_RamanO_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                N_wave.append(wave)
                N_int.append(N_norm_O[v][j])


    # --- Q branch ---
    for v in range(Vmax):
        for j in range(Jmax+1):

            wave = N_RamanQ_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                N_wave.append(wave)
                N_int.append(N_norm_Q[v][j])


    # --- S branch ---
    for v in range(Vmax):
        for j in range(Jmax-1):

            wave = N_RamanS_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                N_wave.append(wave)
                N_int.append(N_norm_S[v][j])


    # convert to numpy
    N_wave = np.array(N_wave)
    N_int  = np.array(N_int)


    # --- limit weak transitions ---
    mask = N_int >= 1e-6

    N_resized_wave = N_wave[mask]
    N_resized_int  = N_int[mask]


    # --- sort transitions (replaces pibub) ---
    idx = np.argsort(N_resized_wave)

    N_resized_wave = N_resized_wave[idx]
    N_resized_int  = N_resized_int[idx]


    num_N_transitions = len(N_resized_wave) - 1


    # ============================
    # BROADBAND RAMAN SPECTRA
    # ============================

    B_wave = []
    B_int  = []

    # --- O branch ---
    for v in range(Vmax):
        for j in range(2, Jmax+1):

            wave = B_RamanO_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                B_wave.append(wave)
                B_int.append(B_norm_O[v][j])


    # --- Q branch ---
    for v in range(Vmax):
        for j in range(Jmax+1):

            wave = B_RamanQ_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                B_wave.append(wave)
                B_int.append(B_norm_Q[v][j])


    # --- S branch ---
    for v in range(Vmax):
        for j in range(Jmax-1):

            wave = B_RamanS_Wave[v][j]

            if (grid_start-0.3) < wave < (grid_end+0.3):

                B_wave.append(wave)
                B_int.append(B_norm_S[v][j])


    B_wave = np.array(B_wave)
    B_int  = np.array(B_int)


    # --- limit weak transitions ---
    mask = B_int >= 1e-6

    B_resized_wave = B_wave[mask]
    B_resized_int  = B_int[mask]


    # --- sort ---
    idx = np.argsort(B_resized_wave)

    B_resized_wave = B_resized_wave[idx]
    B_resized_int  = B_resized_int[idx]


    num_B_transitions = len(B_resized_wave) - 1


    # ============================
    # Spectral Filter
    # ============================

    if filter_on == 0:

        N_resized_wave, N_resized_int, \
        B_resized_wave, B_resized_int = Filter(
            filter_fit,
            N_resized_wave, N_resized_int, num_N_transitions,
            B_resized_wave, B_resized_int, num_B_transitions
        )


    return (
        N_resized_wave, N_resized_int, num_N_transitions,
        B_resized_wave, B_resized_int, num_B_transitions
    )

def H2O_Raman_Int(num_data, T, NLcm, BLcm,
                  J, spin, energy0,
                  H2Oconc, H2O_x_section, H2O_cal_fac,
                  Raman_shift_H2O,
                  grid_start, grid_end,
                  pressure, H2conc, N2conc,
                  O2conc, COconc, CO2conc, Heconc,
                  filter_on):

    rho = real_gas(T, pressure, H2conc, N2conc,
                   H2Oconc, O2conc, Heconc,
                   COconc, CO2conc)

    Q0 = 5196.15
    Q = T**1.5

    N_wave = []
    N_int = []
    B_wave = []
    B_int = []

    max_N = 0
    max_B = 0
    temp_storage = []

    for i in range(num_data+1):

        Nfreq = NLcm - Raman_shift_H2O[i]
        Bfreq = BLcm - Raman_shift_H2O[i]

        Nwave = 1e7 / Nfreq
        Bwave = 1e7 / Bfreq

        N_I = (H2O_x_section*H2O_cal_fac*H2Oconc*
               Q0*spin[i]*(2*J[i]+1) *
               np.exp(-(HCK*energy0[i])/T)) / (Q*Nfreq)

        B_I = (H2O_x_section*H2O_cal_fac*H2Oconc*
               Q0*spin[i]*(2*J[i]+1) *
               np.exp(-(HCK*energy0[i])/T)) / (Q*Bfreq)

        temp_storage.append((Nwave, N_I, Bwave, B_I))
 # I dont undertstand how this works...
        max_N = max(max_N, N_I)
        max_B = max(max_B, B_I)

    for Nwave, N_I, Bwave, B_I in temp_storage:

        N_I /= max_N
        B_I /= max_B

        if grid_start-0.25 < Nwave < grid_end and N_I > 1e-6:
            N_wave.append(Nwave)
            N_int.append(N_I)

        if grid_start-0.25 < Bwave < grid_end and B_I > 1e-6:
            B_wave.append(Bwave)
            B_int.append(B_I)
 #missing numbe_B=broad=0 ?
    N_wave, N_int = pibub(N_wave, N_int)
    B_wave, B_int = pibub(B_wave, B_int)

    if filter_on == 0:
        N_wave, N_int, B_wave, B_int = Filter(
            N_wave, N_int, B_wave, B_int)
 # check filter and filter fit
    return N_wave, N_int, B_wave, B_int, rho

def CO2_Raman_Int(num_data, T, NLcm, BLcm,
                  G, GL, A, deltaQ,
                  CO2conc, CO2_x_section, CO2_cal_fac,
                  grid_start, grid_end,
                  pressure, H2conc, N2conc,
                  O2conc, COconc, H2Oconc, Heconc,
                  filter_on):

    rho = real_gas(T, pressure, H2conc, N2conc,
                   H2Oconc, O2conc, Heconc,
                   COconc, CO2conc=CO2conc)

    Qtotal = np.sum(GL * np.exp(-(1.43877564237*G)/T))

    N_wave = []
    N_int = []
    B_wave = []
    B_int = []

    max_N = 0
    max_B = 0
    temp_storage = []

    for i in range(num_data+1):

        Nfreq = NLcm - deltaQ[i]
        Bfreq = BLcm - deltaQ[i]

        Nwave = 1e7 / Nfreq
        Bwave = 1e7 / Bfreq

        N_I = (CO2_x_section*CO2_cal_fac*CO2conc *
               A[i]**2 * GL[i] * Nfreq**3 *
               np.exp(-(HCK*G[i])/T)) / (T*Qtotal)

        B_I = (CO2_x_section*CO2_cal_fac*CO2conc *
               A[i]**2 * GL[i] * Bfreq**3 *
               np.exp(-(HCK*G[i])/T)) / (T*Qtotal)

        temp_storage.append((Nwave, N_I, Bwave, B_I))

        max_N = max(max_N, N_I)
        max_B = max(max_B, B_I)

    for Nwave, N_I, Bwave, B_I in temp_storage:

        N_I /= max_N
        B_I /= max_B

        if grid_start-0.25 < Nwave < grid_end+0.25 and N_I > 1e-6:
            N_wave.append(Nwave)
            N_int.append(N_I)

        if grid_start-0.25 < Bwave < grid_end+0.25 and B_I > 1e-6:
            B_wave.append(Bwave)
            B_int.append(B_I)

    N_wave, N_int = pibub(N_wave, N_int)
    B_wave, B_int = pibub(B_wave, B_int)

    if filter_on:
        N_wave, N_int, B_wave, B_int = Filter(
            N_wave, N_int, B_wave, B_int)
    
 # almost same as previous not sure if return vars matter
    return N_wave, N_int, B_wave, B_int, rho

#Skipped Raman VJ levels seems to only print out stuff
def Raman_VJ_Levels(species):
    """
    Determines maximum vibrational and rotational quantum levels
    and whether to calculate O and S branches.
    Returns: Vmax, Jmax, OandS
    """

    # -----------------------------
    # Set species-dependent limits
    # -----------------------------
    if species == 1:
        molecule = "H2"
        max_v = 10
        max_j = 25

    elif species in (2, 3, 4):
        max_v = 15
        max_j = 120

        if species == 2:
            molecule = "N2"
        elif species == 3:
            molecule = "O2"
        elif species == 4:
            molecule = "CO"

    else:
        raise ValueError("Invalid species number.")

    print(f"What are the {molecule} Molecular Energy Levels")

    # -----------------------------
    # Vibrational level input
    # -----------------------------
    while True:
        Vmax = int(input(f"Max vibrational level (v <= {max_v}) for {molecule}: "))
        if Vmax <= max_v:
            break
        print(f"\nFOR {molecule} MAXIMUM VIB QUANTUM # MUST BE v <= {max_v}")

    # -----------------------------
    # Rotational level input
    # -----------------------------
    while True:
        Jmax = int(input(f"Max rotational level (j <= {max_j}) for {molecule}: "))
        if Jmax <= max_j:
            break
        print(f"\nFOR {molecule} MAXIMUM ROT QUANTUM # MUST BE j <= {max_j}")

    # -----------------------------
    # O and S branch selection
    # -----------------------------
    while True:
        OandS = int(input("Calculate the O and S branches (0 is yes; 1 is no): "))
        if OandS in (0, 1):
            break
        print("Error: must be 0 (yes) or 1 (no)")

    print()
    return Vmax, Jmax, OandS

def Raman_QOS_Shifts(Vmax, Jmax, OandS,vib, rovib):

    VIB_Raman_shift = np.zeros(Vmax)
    Raman_shift_Q = np.zeros((Vmax, Jmax+1))
    Raman_shift_S = np.zeros((Vmax, Jmax+1))
    Raman_shift_O = np.zeros((Vmax, Jmax+1))

    for v in range(Vmax):
        VIB_Raman_shift[v] = vib[v+1] - vib[v]
        for j in range(Jmax+1):
            Raman_shift_Q[v][j] = rovib[v+1][j] - rovib[v][j]

    if OandS == 0:
        for v in range(Vmax):
            for j in range(Jmax-1):
                Raman_shift_S[v][j] = rovib[v+1][j+2] - rovib[v][j]
            for j in range(2, Jmax+1):
                Raman_shift_O[v][j] = rovib[v+1][j-2] - rovib[v][j]

    return VIB_Raman_shift, Raman_shift_Q, Raman_shift_S, Raman_shift_O

# Need to double check everything