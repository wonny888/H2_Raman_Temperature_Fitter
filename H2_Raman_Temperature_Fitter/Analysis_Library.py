import numpy as np
from Raman_Linewidths import convolution_function
import matplotlib.pyplot as plt


# ============================================================
# Basic Raman Generation and Fixed Parameter Analysis
# ============================================================


SS = np.zeros((125, 375))

def mix_shift_norm(
    num_meas_data_adj,num_pred_data,num_grid_points,mix,intshift_meas,waveshift_meas,
    variance,intshift_pred,waveshift_pred,intshiftN_pred,intshiftB_pred,int_pred,
    N_Int_Spectrum,B_Int_Spectrum,plot_x,noiseshift_meas
):


    # Mix narrowband and broadband
    intshift_pred[:num_pred_data+1] = (
        intshiftN_pred[:num_pred_data] * (1 - mix)
        + intshiftB_pred[:num_pred_data] * mix
    )

    # ------------------------------------------
    # Find optimal intensity scaling factor
    # ------------------------------------------

    match = 500.0
    low = 0.0

    while True:

        residual_term = (
            intshift_meas[:num_pred_data]**2
            - (match * intshift_pred[:num_pred_data])**2
        )

        diff = np.sum(residual_term * noiseshift_meas[:num_pred_data]**2)

        diff = 0.0
        match = 500.0
        while True:
            match *= 2.0
            for i in range(num_pred_data + 1):
                diff += (intshift_meas[i]**2 - (match * intshift_pred[i])**2) * noiseshift_meas[i]**2
            if diff < 0.0:
                break
        high = match


    high = match

    while True:

        match2 = match
        match = (high + low) / 2.0

        residual_term = (
            intshift_meas[:num_pred_data]**2
            - (match * intshift_pred[:num_pred_data])**2
        )

        diff = np.sum(residual_term * noiseshift_meas[:num_pred_data]**2)
        if diff > 0:
            low = match
        else:
            high = match

        if abs(match2 - match) / (match + 1e-12) < 1e-27:
            break


    # Apply scaling

    intshift_pred[:num_pred_data] *= match


    # Build full spectrum

    int_pred[:] = (
        N_Int_Spectrum * (1 - mix)
        + B_Int_Spectrum * mix
    ) * match


    # Normalize

    max_I = np.max(int_pred)

    intshift_pred[:] /= max_I
    int_pred[:] /= max_I
    intshift_meas[:] /= max_I


    intshift_pred,waveshift_meas,waveshift_pred=shift_measured(
        intshift_meas,waveshift_meas,int_pred,plot_x,intshift_pred,waveshift_pred,
        num_meas_data_adj,num_grid_points,variance
    )
    return intshift_pred, waveshift_meas, waveshift_pred, int_pred



def fixed_parameter_analysis( 
    T, method, shots, T_meas, variance,num_meas_data, printM, mix,
    wave_meas, int_meas, noise_meas,num_grid_points, plot_x,
    N_Int_Spectrum, B_Int_Spectrum,SS
):


    # Convert temperatures to string (C++ itoa equivalent)
    temp_meas = str(T_meas)
    temp_pred = str(int(T))

    T2 = int(T)

    # Allocate arrays
    int_pred = np.zeros(num_grid_points + 1)

    waveshift_pred = np.zeros(500)
    intshiftN_pred = np.zeros(500)
    intshiftB_pred = np.zeros(500)
    intshift_pred = np.zeros(500)

    waveshift_meas = np.zeros(500)
    intshift_meas = np.zeros(500)
    noiseshift_meas = np.zeros(500)

    eta = np.zeros(500)
    eta2 = np.zeros(500)
    w = np.zeros(500)

    work1_wave = np.zeros(500)
    work1_int = np.zeros(500)
    work2_wave = np.zeros(500)
    work2_int = np.zeros(500)

    # =============================
    # Find start indices
    # =============================

    if wave_meas[0] < plot_x[0]:

        for i in range(num_meas_data):
            if wave_meas[i] >= plot_x[0]:
                break

        if abs(plot_x[0] - wave_meas[i]) > abs(plot_x[0] - wave_meas[i-1]):
            i -= 1

        start_meas = i
        start_meas = max(0, start_meas)
        start_pred = 0

    else:

        for i in range(num_grid_points):
            if plot_x[i] >= wave_meas[0]:
                break

        if abs(wave_meas[0] - plot_x[i]) > abs(wave_meas[0] - plot_x[i-1]):
            i -= 1

        start_meas = 0
        start_pred = i


    # =============================
    # Find end indices
    # =============================

    if wave_meas[num_meas_data-1] > plot_x[num_grid_points-1]:

        for i in range(start_meas, num_meas_data):
            if wave_meas[i] >= plot_x[num_grid_points-1]:
                break

        end_meas = i - 1
        end_meas = min(len(int_meas) - 1, end_meas)
        end_pred = num_grid_points

    else:

        for i in range(start_pred, num_grid_points):
            if plot_x[i] >= wave_meas[num_meas_data-1]:
                break

        end_pred = i - 1
        end_meas = num_meas_data


    num_pred_data = end_pred - start_pred
    num_meas_data_adj = end_meas - start_meas - 1
    #num_meas_data_adj = end_meas - start_meas +1


    # =============================
    # Open summary file
    # =============================

    if shots > 1:

        filename_meas = f"{temp_meas}_SS_Measured.txt"
        # read shot-specific column from file into int_meas
        filename_noise = f"{temp_meas}_SS_Noise.txt"
        # read shot-specific column from file into noise_meas

        filename_Tsum = temp_pred + "_summary.txt"

        fileout_Tsum = open(filename_Tsum, "w")

        fileout_Tsum.write("Shot Number\tSofS\tmult\n")


    # =============================
    # Shot loop
    # =============================
    shots = int(shots)
    for shot in range(1, shots+1):

        # =============================
        # Match predicted and measured
        # =============================

        # Slice measured data FIRST (this fixes 90% of your bugs)
        intshift_meas = np.asarray(int_meas[start_meas:end_meas+1])
        waveshift_meas = np.asarray(wave_meas[start_meas:end_meas+1])
        noiseshift_meas = np.asarray(noise_meas[start_meas:end_meas+1])

        num_meas_data_adj = len(waveshift_meas)

        # Find closest indices in plot_x
        indices = np.searchsorted(plot_x, waveshift_meas)

        # Fix edge cases
        indices = np.clip(indices, 1, len(plot_x)-1)

        # Choose closest neighbor
        left = plot_x[indices - 1]
        right = plot_x[indices]

        choose_left = np.abs(left - waveshift_meas) < np.abs(right - waveshift_meas)
        indices[choose_left] -= 1

        # Assign values
        intshiftN_pred[:num_meas_data_adj] = N_Int_Spectrum[indices]
        intshiftB_pred[:num_meas_data_adj] = B_Int_Spectrum[indices]
        waveshift_pred[:num_meas_data_adj] = plot_x[indices]

        num_pred_data = num_meas_data_adj


        num_pred_data = num_meas_data_adj


        # Shift arrays to start at zero
        intshift_meas = np.asarray(int_meas[start_meas:end_meas+1])
        waveshift_meas = np.asarray(wave_meas[start_meas:end_meas+1])
        noiseshift_meas = np.asarray(noise_meas[start_meas:end_meas+1])

        # =============================
        # METHOD 1: Constant mix
        # =============================

        if method == 1:

            intshift_pred,waveshift_meas,waveshift_pred=mix_shift_norm(
                num_meas_data_adj,num_pred_data,num_grid_points,mix,intshift_meas,waveshift_meas,
                variance,intshift_pred,waveshift_pred,intshiftN_pred,intshiftB_pred,int_pred,
                N_Int_Spectrum,B_Int_Spectrum,plot_x,noiseshift_meas
            )


            # Compute weighted sum of squares

            residual = intshift_meas - intshift_pred[:len(intshift_meas)]
            weights = noiseshift_meas**2

            S_single_shot = np.sum(residual**2 * weights)

            SS[shot][T2] = S_single_shot

            beta = mix



        # =============================
        # METHOD 2: Estimated mix
        # =============================

        if method == 2:

            beta = mix

            filename_est = temp_pred + f"_mix_Estimator_shot_{shot}.txt"

            file_est = open(filename_est, "w")

            file_est.write("Iteration\tmix\tSofS\n")

            for k in range(1000):

                work1_wave[:] = waveshift_meas[:]
                work1_int[:] = intshift_meas[:]

                work2_wave[:] = waveshift_meas[:]
                work2_int[:] = intshift_meas[:]


                intshift_pred,waveshift_meas,waveshift_pred=mix_shift_norm(
                    num_meas_data_adj,num_pred_data,num_grid_points,beta,work1_int,work1_wave,
                    variance,eta,waveshift_pred,intshiftN_pred,intshiftB_pred,int_pred,N_Int_Spectrum,
                    B_Int_Spectrum,plot_x,noiseshift_meas
                )


                residual = work1_int[:num_meas_data_adj] - eta[:num_meas_data_adj]

                weights = noiseshift_meas[:num_meas_data_adj]**2

                E = np.sum(residual**2 * weights)

                file_est.write(f"{k+1}\t{beta:.15e}\t{E:.15e}\n")


                # Numerical derivative

                delta = beta * 0.1

                beta_test = beta + delta

                intshift_pred,waveshift_meas,waveshift_pred=mix_shift_norm(
                    num_meas_data_adj,num_pred_data,num_grid_points,beta_test,work2_int,work2_wave,
                    variance,eta2,waveshift_pred,intshiftN_pred,intshiftB_pred,int_pred,N_Int_Spectrum,
                    B_Int_Spectrum,plot_x,noiseshift_meas
                )


                derivative = (eta2 - eta) / delta


                numerator = np.sum(derivative * weights * residual)

                denominator = np.sum(derivative**2 * weights)


                beta_new = beta + 0.1 * numerator / denominator


                if abs((beta_new - beta) / (beta + 1e-90)) < 1e-5:
                    beta = beta_new
                    break

                beta = beta_new


            file_est.close()


            # Final evaluation

            intshift_pred,waveshift_meas,waveshift_pred=mix_shift_norm(
                num_meas_data_adj,num_pred_data,num_grid_points,beta,intshift_meas,waveshift_meas,variance,
                intshift_pred,waveshift_pred,intshiftN_pred,intshiftB_pred,int_pred,N_Int_Spectrum,
                B_Int_Spectrum,plot_x,noiseshift_meas
            )


            residual = intshift_meas[:num_meas_data_adj] - intshift_pred[:num_meas_data_adj]

            weights = noiseshift_meas[:num_meas_data_adj]**2

            S_single_shot = np.sum(residual**2 * weights)

            SS[shot][T2] = S_single_shot


        # =============================
        # Write summary
        # =============================

        if shots > 1:
            fileout_Tsum.write(f"{shot}\t{S_single_shot}\t{beta}\n")

    if printM == 0:
        filename = f"{temp_pred}_Fitted_Avg.txt" if shots == 1 else f"{temp_pred}_Fitted_shot#{shot}.txt"
        with open(filename, "w") as f:
            f.write("Wavelength\tIntensity\tWavelength\tIntensity\tResidual\tSofS\tWavelength\tIntensity\n")
            SofS = 0.0
            for i in range(num_meas_data_adj + 1):
                residual = intshift_meas[i] - intshift_pred[i]
                SofS += residual * noiseshift_meas[i]**2 * residual
                f.write(f"{waveshift_meas[i]:.10f}\t{intshift_meas[i]:.10f}\t"
                        f"{waveshift_pred[i]:.10f}\t{intshift_pred[i]:.10f}\t"
                        f"{residual-0.05:.10f}\t{SofS:.10f}\t"
                        f"{plot_x[i]:.10f}\t{int_pred[i]:.10f}\n")

    if shots > 1:
        fileout_Tsum.close()

    plt.figure()
    plt.plot(plot_x, int_pred, label="Predicted")
    plt.scatter(waveshift_meas, intshift_meas, color='red', s=10, label="Measured (shifted)")
    plt.legend()
    plt.show()
    return intshift_pred,waveshift_meas,waveshift_pred

def basic_raman_gen_analysis(
    T, num_grid_points,num_N_transitions,num_B_transitions,N_lineshape,B_lineshape,NGLvalue,
    BGLvalue,N_resized_wave,N_resized_int,B_resized_wave,B_resized_int,printS,species,
    pressure,rho,Jmax,H2conc,H2Oconc,N2conc,O2conc,Heconc,COconc,CO2conc,linewidthN,linewidthB,
    lambdaspect,dispersion,slitwidth,VNG,VNL,VBG,VBL,analysis,method,shots,T_meas,
    variance,num_meas_data, printM,mix,wave_meas,int_meas,noise_meas,grid_start,increment # removed SS for now?
):

    # ------------------------------------------
    # Build wavelength grid
    # ------------------------------------------

    plot_x = np.zeros(num_grid_points + 1)

    build_grid = grid_start - increment

    for i in range(num_grid_points + 1):
        build_grid += increment
        plot_x[i] = build_grid

    # ------------------------------------------
    # Initialize spectra
    # ------------------------------------------

    N_Int_Spectrum = np.zeros(num_grid_points + 1)
    B_Int_Spectrum = np.zeros(num_grid_points + 1)



    # ------------------------------------------
    # Convolution
    # ------------------------------------------

    N_Int_Spectrum, B_Int_Spectrum = convolution_function(
        T, num_grid_points,num_N_transitions,num_B_transitions,N_lineshape,B_lineshape,
        NGLvalue,BGLvalue,N_resized_wave,N_resized_int,B_resized_wave,B_resized_int,
        plot_x,species,pressure,rho,Jmax,H2conc,H2Oconc,N2conc,O2conc,Heconc,COconc,
        CO2conc,linewidthN,linewidthB,lambdaspect,dispersion,slitwidth,VNG,VNL,VBG,VBL
    )


    # ------------------------------------------
    # Analysis
    # ------------------------------------------

    if analysis == 1:

        fixed_parameter_analysis(
            T,method,shots,T_meas,variance,num_meas_data,printM,mix,
            wave_meas,int_meas,noise_meas,num_grid_points,plot_x,
            N_Int_Spectrum,B_Int_Spectrum,SS
        )


    return plot_x, N_Int_Spectrum, B_Int_Spectrum,SS

def shift_measured(
    intshift_meas,waveshift_meas,int_pred,plot_x,intshift_pred,waveshift_pred,
    num_meas_data_adj,num_grid_points,variance
):

    opt_pred = 0

    for meas in range(num_meas_data_adj):

        pred = opt_pred

        while pred <= num_grid_points and plot_x[pred] < waveshift_meas[meas]:
            pred += 1

        if abs(plot_x[pred] - waveshift_meas[meas]) > abs(plot_x[pred-1] - waveshift_meas[meas]):
            pred -= 1


        min_diff = 1e12

        for ii in range(pred - variance, pred + variance + 1):
            idx = max(0, min(ii, num_grid_points))
            diff = abs(intshift_meas[meas] - int_pred[idx])
            if diff < min_diff:
                min_diff = diff
                opt_pred = idx


        if abs(waveshift_meas[meas] - plot_x[opt_pred]) > 0.005:

            print("ERROR wavelength shift too large")


        intshift_pred[meas] = int_pred[opt_pred]
        waveshift_meas[meas] = plot_x[opt_pred]
        waveshift_pred[meas] = plot_x[opt_pred]
    
    return intshift_pred,waveshift_meas,waveshift_pred

def print_residuals(
    T_meas, Tmin, Tmax, shots, SS
):
    shots = int(shots)
    T_meas = int(T_meas)
    Tmin = int(Tmin)
    Tmax = int(Tmax)
    if shots == 1:

        with open(f"{T_meas}_Avg_Residuals.txt", "w") as f:

            f.write("T\tSofS\n")

            for T in range(Tmin, Tmax + 1):

                f.write(f"{T}\t{SS[1][T]}\n")


    else:

        with open(f"{T_meas}_SS_Res_Summary.txt", "w") as f:

            f.write("Shot\tT\tSofS\n")

            for shot in range(1, shots + 1):

                minS = np.min(SS[shot][Tmin:Tmax+1])
                optT = np.argmin(SS[shot][Tmin:Tmax+1]) + Tmin

                f.write(f"{shot}\t{optT}\t{minS}\n")

