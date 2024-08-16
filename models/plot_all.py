import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
from tqdm import tqdm
import os

#constants
G = 4.3e-6
a0 = 3702.81 #from 1.2 * 10 ^ {-10} m / s^2 to km^2 / s^2 / kpc

path = "/Rotmod_LTG/"
	
df_NFW = pd.read_csv("parameters_NFW.csv")
df_MOND = pd.read_csv("parameters_MOND.csv")
df_Einasto = pd.read_csv("parameters_Einasto.csv")
df_pISO = pd.read_csv("parameters_pISO.csv")

def NFW_v_halo(Y_star, rho0, Rs):
	x = r / Rs
	M_NFW = 4 * np.pi * rho0 * Rs**3 * (np.log(1 + x) - x / (1 + x))
	return G * M_NFW / r
	
def einasto_v_halo(Y_star, rho0, Rs, a):
	prefactor = 4 * np.pi * rho0 * Rs**3 * np.exp(2/a) * (2/a)**(-3/a) / a
	M_einasto = prefactor * gamma(3/a) * gammainc(3/a, 2/a * (r/Rs)**a)
	return G * M_einasto / r
	
def pISO_v_halo(Y_star, rho0, Rc):
	v_halo_sq = 4 * np.pi * G * rho0 * Rc ** 2 * ( 1 - (Rc / r) * np.arctan(r / Rc))
	return v_halo_sq

def total_velocity_squared(fun, *args):
	v_star_sq = Y_star * v_disk_sq + 1.4 * Y_star * v_bulge_sq
	v_halo_sq = fun(*args)
	v_total_sq = v_star_sq + v_gas_sq + v_halo_sq
	return v_total_sq
	
def v_newtonian_square(Y_star):
	v_star_sq = Y_star * v_disk_sq + 1.4 * Y_star * v_bulge_sq
	return v_star_sq + v_gas_sq

def newtonian_acceleration(Y_star):
	v_N_sq = v_newtonian_square(Y_star)
	a_N = v_N_sq / r
	return a_N
	
def mond_velocity_squared(Y_star):
	v_N_sq = v_newtonian_square(Y_star)
	a_N = newtonian_acceleration(Y_star)
	
	v_MOND_sq = v_N_sq * np.sqrt(0.5 + 0.5 * np.sqrt(1 + (2 * a0 / a_N)**2))
	return v_MOND_sq

def parameters(df, GALAXY_NAME, values):
	data = df[df['Galaxy'] == GALAXY_NAME]
	theta = data.iloc[:,1:values+1].values[0]
	return theta
	
def plot_models(model):
	v_model = np.sqrt(v_model_sq)
	plt.plot(r, v_model, label=model)
	plt.xlabel('Radius (kpc)')
	plt.ylabel('Velocity (km/s)')
	plt.legend()


for GALAXY_NAME in tqdm(os.listdir(path)):
	name = path + GALAXY_NAME
	R, V, Verr, Vgas, Vbul, Vdisk = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 4, 5))
	GALAXY_NAME = GALAXY_NAME[:-len("_rotmod.dat")]
		
	r = np.array(R)
	v_obs = np.array(V)
	v_err = np.array(Verr)
	v_disk_sq = np.array(Vdisk)**2
	v_bulge_sq = np.array(Vbul)**2
	v_gas_sq = np.array(Vgas)**2
	
	plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
	
	#NFW
	Y_star, rho0, Rs = parameters(df_NFW, GALAXY_NAME, 3)

	v_model_sq = total_velocity_squared(NFW_v_halo, Y_star, rho0, Rs)
	
	plot_models("NFW")

	#Einasto
	Y_star, rho0, Rs, a = parameters(df_Einasto, GALAXY_NAME, 4)
	
	v_model_sq = total_velocity_squared(einasto_v_halo, Y_star, rho0, Rs, a)
	
	plot_models("Einasto")
	
	#pISO
	Y_star, rho0, Rc = parameters(df_pISO, GALAXY_NAME, 3)
	
	v_model_sq = total_velocity_squared(pISO_v_halo, Y_star, rho0, Rc)

	plot_models("pISO")

	#MOND
	Y_star = parameters(df_MOND, GALAXY_NAME, 1)
	v_model_sq = mond_velocity_squared(Y_star)

	plot_models("MOND")
	
	plt.savefig(GALAXY_NAME + ' all_models.pdf')
	plt.close()

