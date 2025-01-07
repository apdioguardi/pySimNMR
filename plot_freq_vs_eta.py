# -- coding: utf-8 --
"""
PlotFreqSpectrumVsAngle
=========
____________
Version: 0.1
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a single crystal second order pert frequency spectral positions vs angle
according to the given parameters. Additionally a measured data given as a text 
file can be displayed for comparison.

###############################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
import pySimNMR
import time
import sys


###############################################################################
### PARAMETERS ################################################################

##----NMR Parameters-----------------------------------------------------------
isotope = '115In'                # need to add many nuclei to the dictionary still... program will fail if not added at this stage of development
Ka = 0.0                        # shift tensor elements (units = percent)
Kb = 0.0
Kc = 0.0

va = None
vb = None
vc = 4.4                        # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta = 0.0                       # unitless

H0 = 0.0                        # magnetic field  (units = T)
phi_z_deg = 0.0                 # Range: (0-360) ZXZ Euler angles phi, theta, and psi for rotation of the EFG + K tensors with respect to H0
theta_x_prime_deg = 0.0        # Range: (0-180) these values are in degrees and converted to radians in the code
psi_z_prime_deg = 0.0           # Range: (0-360)

##----Simulation control-------------------------------------------------------
min_freq = 2.0                 # units = MHz
max_freq = 50.0                 # units = MHz
# angle_to_vary = 'phi_z_deg'     # string named the same as above
# angle_start = 0.0
# angle_stop = 90.0               # this range will replace the constant given in the NMR Parameters section
eta_start = 0
eta_stop = 1.0
# n_angles = 1e4                  # number of angles (x axis points)
n_etas = 1e2                    # number of eta value (x axis points)
mtx_elem_min = 0.1              # In general 0.5 is a good starting point

##----Plotting data from data File---------------------------------------------
exp_data_file=''                # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as angle in degrees, second as frequency
number_of_header_lines=0        # number of lines which are ignored in the begining of the data file
exp_data_delimiter=' '          # tell numpy which delimter your experimental data file has 

##----Exporting Simulated Spectrum---------------------------------------------
sim_export_file='eta_test.txt'              # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''

###############################################################################
###############################################################################

















# instantiate the simulation class
sim = pySimNMR.SimNMR(isotope)

# generate the eta array
eta_array = np.linspace(eta_start, eta_stop, int(n_etas))

phi_z = np.array([phi_z_deg*np.pi/180])
theta_xp = np.array([theta_x_prime_deg*np.pi/180])
psi_zp = np.array([psi_z_prime_deg*np.pi/180])

print('Running in Exact Diagonalization Mode')

#######################################
t0 = time.time()
###################
#ahoy! got to here...
# use the built in SimNMR methods to generate the rotation matrices
# lower case r matrices are for rotation of the shift tensor
# capital R matrices are for rotation of the quadrupole Hamiltonian
r, ri = sim.generate_r_matrices(phi_z,
                                theta_xp,
                                psi_zp)
SR, SRi = sim.generate_r_spin_matrices(phi_z,
                                   theta_xp,
                                   psi_zp)
rotation_matrices = (r, ri, SR, SRi)
eta_array_out=np.array([])
f=np.array([])
p=np.array([])
t=np.array([])
for eta in eta_array:
    one_f,one_p,one_t = sim.freq_prob_trans_ed(
                                    H0=H0,
                                    Ka=Ka,
                                    Kb=Kb,
                                    Kc=Kc,
                                    va=va,
                                    vb=vb,
                                    vc=vc,
                                    eta=eta,
                                    rotation_matrices=rotation_matrices,
                                    mtx_elem_min=mtx_elem_min,
                                    min_freq=min_freq,
                                    max_freq=max_freq
                                  )


    one_eta=np.full(shape=one_f.shape, fill_value=eta)
    print(one_eta.shape)
    print(one_f.shape)
    print(one_p.shape)
    print(one_t.shape)

    eta_array_out=np.append(eta_array_out,one_eta)
    f=np.append(f,one_f)
    p=np.append(p,one_p)
    t=np.append(t,one_t)

#get the isotope with which we are working
#I0=sim.isotope_data_dict[isotope]["I0"]
#integer of the number of resonances we expect (there may be a better way to get this and return
# it from the freq_prob_trans_ed function as perhaps a list of integers each associated with a number,
# this may be a "bug" if we work in the regime where H_q ~= H_z)
#n_resonances=int(I0*2)
#create an empty numpy array of the appropriate size (default dtype is float)
# angle_array_out = np.empty(n_resonances*angle_array.size)
##eta_array_out = np.empty(n_resonances*eta_array.size)
#slice this array in a loop such that we set m to the end in steps of n_resonances [start:stop:step]
# and if stop is left blank it is to the end
# so the effect is that the first n_resonances indicies are set to the first angle in angle_array
# second n_resonances are set to the second angle and so on
##for m in range(n_resonances):
##    # angle_array_out[m::n_resonances]=angle_array*(180/np.pi)
##   eta_array_out[m::n_resonances]=eta_array

# afpt=np.column_stack((angle_array_out,f,p,t))
print(len(eta_array_out))
print(len(f))
print(len(p))
print(len(t))
efpt=np.column_stack((eta_array_out,f,p))
###################
t1 = time.time()
dt_str = str(t1-t0) + ' s'
print('simulation took ' + dt_str)
#######################################

# setting the boolean variable for saving the simulation output or not
if sim_export_file!='':
    # np.savetxt(sim_export_file, afpt)
    np.savetxt(sim_export_file,efpt)

# prepare for plotting
fig = plt.figure()
ax = fig.add_subplot(111) 

# load experimental data for comparison
if exp_data_file!='':
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:,0]
    exp_y = exp_data[:,1]
    # plot experimental data first as black squares
    ax.plot(exp_x,exp_y,"ks")

# prepare simulation data for plotting
# sim_x = afpt[:,0]
# sim_y = afpt[:,1]
sim_x = efpt[:,0]
sim_y = efpt[:,1]


# plot simulation as red lines
ax.plot(sim_x,sim_y,"ro")    
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.set_ylabel('Frequency (MHz)')
ax.set_xlabel(r'$\eta$ (unitless)')
gamma = sim.isotope_data_dict[isotope]["gamma"]
isotope_str = str(isotope) + ', '
gamma_str = r'$\gamma=$ ' + str(gamma) + r' MHz/T, '
H0_str = r'$H_0$ = ' + str(H0) + r' T, '
K_a_str = r'$K_a$ = ' + str(Ka) + r' %, '
K_b_str = r'$K_b$ = ' + str(Kb) + r' %, '
K_c_str = r'$K_c$ = ' + str(Kc) + r' %, '
v_c_str = r'$\nu_c$ = ' + str(vc) + r' MHz, '
# eta_str = r'$\eta$ = ' + str(eta) + r', '
phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg) + r'$\degree$, ' 
theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_x_prime_deg) + r'$\degree$, '
psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg) + r'$\degree$'
#     psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg) + r'$\degree$'
# if angle_to_vary=='phi_z_deg':
#     phi_z_deg_str = r''
#     theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_x_prime_deg) + r'$\degree$, '
#     psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg) + r'$\degree$'
#     ax.set_xlabel(r'$\phi_z$ ($\degree$)')
# elif angle_to_vary=='theta_x_prime_deg':
#     theta_x_prime_deg_str = r''
#     phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg) + r'$\degree$, ' 
#     psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg) + r'$\degree$'
#     ax.set_xlabel(r'$\theta_{x^\prime}$ ($\degree$)')
# elif angle_to_vary=='psi_z_prime_deg':
#     psi_z_prime_deg_str = r''
#     phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg) + r'$\degree$, '
#     theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_x_prime_deg) + r'$\degree$, '
#     ax.set_xlabel(r'$\psi_{z^\prime}$ ($\degree$)')

title_str = isotope_str + gamma_str + H0_str + '\n' + v_c_str + K_a_str + K_b_str + K_c_str + '\n' + phi_z_deg_str + theta_x_prime_deg_str + psi_z_prime_deg_str
ax.set_title(title_str)

fig.tight_layout()
plt.show()
