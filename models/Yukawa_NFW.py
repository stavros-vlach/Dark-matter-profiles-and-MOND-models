import random
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy.optimize import differential_evolution, minimize
import scipy
from tqdm import tqdm
from scipy.special import gamma, gammainc
import os
import csv
from scipy.special import expi


labels = ["Galaxy",
	"Hubble Type",
        "Distance (Mpc)",
        "Mean D error (Mpc)",
        "Distance Method",
        "Inclination (deg)",
        "Mean Inc error (deg)",
        "Total Luminosity at [3.6](10+9solLum)",
        "Effective Radius at [3.6](kpc)",
        "Effective Surface Brightness at [3.6](solLum/pc2)",
        "Disk Scale Length at [3.6] (kpc)",
        "Disk Central Surface Brightness at [3.6] (solLum/pc2)",
        "Total HI mass (10+9solMass)",
        "HI radius at 1 Msun/pc2 (kpc)",
        "Asymptotically Flat Rotation Velocity(km/s)",
        "Mean Vflat error (km/s)",
        "Mean error on Vflat (km/s)",
        "Quality Flag (3)",
        "Refs"]

#constants
G = 4.3e-6
h = 0.671
rho_crit = 143.84

def total_velocity_squared(Y_star_disk, Y_star_bulge, logM_200, beta, lambda_):
    M_200 = 10 ** logM_200
    c = 10 ** (0.905) * (M_200 / (10 ** 12 * h ** (-1))) ** (-0.101)
    r_s = 28.8 * (M_200 / (10 ** 12 * h ** (-1))) ** 0.43
    rho_s = 200 / 3 * c ** 3 * rho_crit / (np.log(1 + c) - c / (1 + c))
    x = r / r_s

    v_NFW_sq = 4 * np.pi * G / r * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))

    # Yukawa correction velocity
    term1 = expi((r_s / lambda_))
    term2 = expi(-(r_s + r) / lambda_)
    term3 = expi(-r_s / lambda_)
    term4 = expi((r + r_s) / lambda_)
    
    # Calculate the Yukawa modified gravity potential velocity squared
    v_mg_sq = -2 * np.pi * G * beta * rho_s * r_s ** 3 / r * (
        2 * r / (r_s + r)
        + np.exp((r_s + r) / lambda_) * (r / lambda_ - 1) * term2
        + np.exp(-(r_s + r) / lambda_) * (1 + r / lambda_) * (np.exp(2 * r_s / lambda_) * term3 + term1 - term4)
    )
    
    # Total velocity squared
    v_total_sq = (
        v_gas_sq 
        + Y_star_disk * v_disk_sq 
        + Y_star_bulge * v_bulge_sq
        + v_NFW_sq 
        + v_mg_sq
    )
    return v_total_sq

def log_probability(theta):
    Y_star_disk, Y_star_bulge, logM_200, beta, lambda_ = theta
    M_200 = 10 ** logM_200
    if not(0.3 < Y_star_D < 0.8) or not(0.3 < Y_star_B < 0.8)\
    or not(9 < M_200 < 14) or not(-2 < beta < 2) or not(0.01 < lambda_ < 100):
        return -np.inf
    v_model_sq = calc_v_NFW_Yukawa_halo(Y_star_D, Y_star_B, logM_200, beta, l)
    chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
    chi_sq *= -0.5
    if np.any(np.isnan(chi_sq)):
        return -np.inf
    return chi_sq

def chi_square(theta):
    Y_star_disk, Y_star_bulge, logM_200, beta, lambda_ = theta
    v_model_sq = total_velocity_squared(Y_star_disk, Y_star_bulge, logM_200, beta, lambda_)
    chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
    return chi_sq / (len(r) - len(theta))

#bounds for differential_evolution
bounds = [(0.3, 0.8), (0.3, 0.8), (9, 14), (-2, 2), (0.01, 100)]

#parameters for emcee
ndim = 5
nwalkers = 500
nsteps = 1000

with open('parameters_Yukawa_NFW.csv','w') as testfile:
    csv_writer=csv.writer(testfile)
    csv_writer.writerow(["Galaxy", "Y*_disk", "Y_star_bulge", "M_200", "beta", "lambda", "Y*_disk_error", "Y*_bulge_error", "M_200_error", "beta_error", "lambda_error"])

    path = "/Rotmod_LTG/"
    for GALAXY_NAME in os.listdir(path):
        name = path + GALAXY_NAME
        R, V, Verr, Vgas, Vbul, Vdisk = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 4, 5))
        data_length = len(R)

        GALAXY_NAME = GALAXY_NAME[:-len("_rotmod.dat")]

        # The full dataset of http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt,
        # with minor changes for parsing
        data = pd.read_csv('data.txt', delimiter=';', skiprows=98, names=labels)

        data = data[data['Galaxy'] == GALAXY_NAME]
        i_set = data["Inclination (deg)"].to_numpy()[0] * np.pi / 180
        err_i_set = data["Mean Inc error (deg)"].to_numpy()[0] * np.pi / 180
        D_quot = data["Distance (Mpc)"].to_numpy()[0]
        err_D_quot = data["Mean D error (Mpc)"].to_numpy()[0]
        L_tot = data["Total Luminosity at [3.6](10+9solLum)"].to_numpy()[0]

        Lb = pd.read_csv('Lbul.txt', delimiter = '\s+', skiprows=7, names=['Galaxy', 'Lbul (10^9 Lsun)'])
        Lb = Lb[Lb['Galaxy'] == GALAXY_NAME]
        L_bul = Lb['Lbul (10^9 Lsun)'].to_numpy()[0]
        r = np.array(R)
        v_obs = np.array(V)
        v_err = np.array(Verr)
        v_disk_sq = np.array(Vdisk)**2
        v_bulge_sq = np.array(Vbul)**2
        v_gas_sq = np.array(Vgas)**2


        result = differential_evolution(chi_square, bounds)
        best_fit_params = result.x
        print(result)

        Y_star_D, Y_star_B, M_200, beta, l = best_fit_params
        v_model_sq_best = total_velocity_squared(Y_star_D, Y_star_B, M_200, beta, l)
        v_model_best = np.sqrt(v_model_sq_best)

        plt.plot(r, v_model_best, label = 'Best-fit model before emcee', color = 'red')
        plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
        plt.ylabel('$V(km/s)$')
        plt.xlabel('$R(kpc)$')
        plt.legend()
        plt.savefig(GALAXY_NAME + ' fit_results-beforeMCMC.pdf')
        plt.close()

        # Initialize the walkers
        pos = best_fit_params + 1e-2 * np.random.randn(nwalkers, ndim)

        # Set up the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        
        # Run the sampler
        sampler.run_mcmc(pos, nsteps, progress=True)

        # Extract the samples
        samples = sampler.get_chain(discard=100, thin=10, flat=True)

        fig = corner.corner(samples, labels=["$Y^*_{disk}$", "$Y^*_{bulge}$", "$M_{200}$", "$\\beta$", "$\\lambda$"], truths=[0, 0, 0, 0, 0])
        fig.savefig(GALAXY_NAME + "fit_corner.pdf")
        plt.close()

        Y_star_disk_mcmc, Y_star_bulge_mcmc, logM_200_mcmcm, beta_mcmc, lambda_mcmc = np.percentile(samples, 50, axis=0)
        Y_star_disk_mcmc_err = np.percentile(samples[:, 0], [16, 84])
        Y_star_bulge_mcmc_err = np.percentile(samples[:, 1], [16, 84])
        logM_200_mcmcm_err = np.percentile(samples[:, 2], [16, 84])
        beta_mcmc_err = np.percentile(samples[:, 3], [16, 84])
        lambda_mcmc_err = np.percentile(samples[:, 4], [16, 84])
        
        csv_writer.writerow([GALAXY_NAME, Y_star_disk_mcmc, Y_star_bulge_mcmc, 10**logM_200_mcmcm, beta_mcmc, lambda_mcmc,
                             Y_star_disk_mcmc_err, Y_star_bulge_mcmc_err, 10**logM_200_mcmcm_err, beta_mcmc_err, lambda_mcmc_err])
        
        Y_star_disk_mcmc, Y_star_bulge_mcmc, logM_200_mcmcm, beta_mcmc, lambda_mcmc = np.median(samples, axis=0)

        # Compute model velocities with best-fit parameters from MCMC
        v_model_sq_mcmc = total_velocity_squared(Y_star_disk_mcmc, Y_star_bulge_mcmc, logM_200_mcmcm, beta_mcmc, lambda_mcmc)
        v_model_mcmc = np.sqrt(v_model_sq_mcmc)

        plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
        plt.plot(r, v_model_mcmc, label='Best-fit model after emcee', color='red')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.legend()
        plt.savefig(GALAXY_NAME + ' fit_results-afterMCMC.pdf')
        plt.close()


