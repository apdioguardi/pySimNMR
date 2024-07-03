# -- coding: utf-8 --
"""
plot_freq_vs_field
=========
____________
Version: 0.2
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a single crystal second order pert frequency spectral positions vs
magnetic field based on the given parameters. Additionally a measured 
data given as a text file can be displayed for comparison.
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
isotope = '75As'                 # need to add many nuclei to the dictionary still... program will fail if not added at this stage of development

va = None                    # units = MHz (set va=None if you would like to specify eta)
vb = None                      # units = MHz (se vb=None if you would like to specify eta))
vc = 21.15                    # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta = 0.0                      # unitless

Ka = 0.25                        # shift tensor elements (units = percent)
Kb = 0.25
Kc = 0.25

Hinta = 0.0
Hintb = 0.0
Hintc = 0.0

phi_z_deg = 30.234                 # Range: (0-360) ZXZ Euler angles phi, theta, and psi for rotation of the EFG + K tensors with respect to H0
theta_x_prime_deg = 85.345        # Range: (0-180) these values are in degrees and converted to radians in the code
psi_z_prime_deg = 48.1235           # Range: (0-360)

##----Simulation control-------------------------------------------------------
min_freq = 0.001                 # units = MHz
max_freq = 75.0                 # units = MHz

min_field = 0.001
max_field = 10.0
n_fields = 1000                   # number of eta value (x axis points)
#field_array=np.array([0.01,0.05,0.1,0.5,1.0,5.0])


mtx_elem_min = 1e-3            # In general 0.5 is a good starting point

##----Plot control-------------------------------------------------------
plot_legend_width_ratio=[6,1]
x_low_limit=min_field
x_high_limit=max_field
y_low_limit=min_freq
y_high_limit=max_freq
color_map_name = 'viridis_r'    #add _r to reverse the colormap, eg 'viridis_r'
                                # cmaps['Perceptually Uniform Sequential'] = [
                                #             'viridis', 'plasma', 'inferno', 'magma']
                                # cmaps['Sequential'] = [
                                #             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                #             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                #             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


##----Plotting data from data File---------------------------------------------
exp_data_file=''                # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as angle in degrees, second as frequency
number_of_header_lines=0        # number of lines which are ignored in the begining of the data file
exp_data_delimiter=' '          # tell numpy which delimter your experimental data file has 

##----Exporting Simulated Spectrum---------------------------------------------
sim_export_file='fvsH_75As_LiFeAs.txt'              # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''

###############################################################################
###############################################################################

















# instantiate the simulation class
sim = pySimNMR.SimNMR(isotope)

# generate the field array
field_array = np.linspace(min_field, max_field, n_fields)

phi_z = np.array([phi_z_deg*np.pi/180])
theta_xp = np.array([theta_x_prime_deg*np.pi/180])
psi_zp = np.array([psi_z_prime_deg*np.pi/180])


print('Running in Exact Diagonalization Mode')

#######################################
t0 = time.time()
###################
# use the built in SimNMR methods to generate the rotation matrices
# lower case r matrices are for rotation of the shift tensor
#  SR matrices are for spin-space rotation of the quadrupole Hamiltonian
r, ri = sim.generate_r_matrices(phi_z,
                                theta_xp,
                                psi_zp)
SR, SRi = sim.generate_SR_matrices(phi_z,
                                   theta_xp,
                                   psi_zp)
rotation_matrices = (r, ri, SR, SRi)

hfpt = sim.freq_vs_field_ed(H0=field_array, 
                            Ka=Ka, 
                            Kb=Kb, 
                            Kc=Kc, 
                            va=va,
                            vb=vb,
                            vc=vc,
                            eta=eta,
                            rm_SRm_tuple=rotation_matrices,
                            Hinta=Hinta,
                            Hintb=Hintb,
                            Hintc=Hintc,
                            mtx_elem_min=mtx_elem_min,
                            min_freq=min_freq,
                            max_freq=max_freq)

###################
t1 = time.time()
dt_str = str(t1-t0) + ' s'
print('simulation took ' + dt_str)
#######################################
#print(hfpt.shape)
# setting the boolean variable for saving the simulation output or not
if sim_export_file!='':
    np.savetxt(sim_export_file, hfpt)

# prepare for plotting
plt.rcParams["figure.figsize"] = 10,4.5 # width,height in inches
fig, (ax, lax) = plt.subplots(ncols=2,gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file!='':
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:, 0]
    exp_y = exp_data[:, 1]
    # plot experimental data first as black squares
    ax.plot(exp_x, exp_y, "ks")

# prepare simulation data for plotting
sim_x = hfpt[:, 0] # field
sim_y = hfpt[:, 1] # frequency
sim_z = hfpt[:, 2] # probability


# plot simulation as red lines
cax = ax.scatter(sim_x, sim_y, c=sim_z, cmap=color_map_name)

#fig.colorbar(cax,orientation='horizontal',label='Transition Probability (arb. units)')
fig.colorbar(cax,label='Transition Probability (arb. units)')

# label the axes
ax.set_ylabel('Frequency (MHz)')
ax.set_xlabel(r'$H_0$ (T)')

# generate strings for the title and legend/details

isotope_str = 'isotope = ' + str(isotope)
gamma = sim.isotope_data_dict[isotope]['gamma']
gamma_str = r'$\gamma = $ ' + str(gamma) + r' MHz/T'

K_a_str = r'$K_a$ = ' + str(Ka) + r' %'
K_b_str = r'$K_b$ = ' + str(Kb) + r' %'
K_c_str = r'$K_c$ = ' + str(Kc) + r' %'

v_c_str = r'$\nu_c$ = ' + str(vc) + r' MHz'

Hinta_str = r'$H_{int}^{a}$ = ' + str(Hinta) + r' T'
Hintb_str = r'$H_{int}^{b}$ = ' + str(Hintb) + r' T'
Hintc_str = r'$H_{int}^{c}$ = ' + str(Hintc) + r' T'

phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg) + r' $\degree$' 
theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_x_prime_deg) + r' $\degree$'
psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg) + r' $\degree$'

title_str = 'Frequency vs Field; Exact Diagonalization'
ax.set_title(title_str)

if va!=None and vb!=None:
    va_str = r'$\nu_a$ = ' + str(va) + r' MHz'
    vb_str = r'$\nu_b$ = ' + str(vb) + r' MHz'
    # Put a legend to the right of the current axis
    legend_title_str = str(isotope_str + '\n' 
                        + gamma_str + '\n' 
                        + K_a_str + '\n' 
                        + K_b_str + '\n' 
                        + K_c_str + '\n' 
                        + v_a_str + '\n' 
                        + v_b_str + '\n' 
                        + v_c_str + '\n' 
                        + Hinta_str + '\n'
                        + Hintb_str + '\n'
                        + Hintc_str + '\n'
                        + phi_z_deg_str + '\n' 
                        + theta_x_prime_deg_str + '\n' 
                        + psi_z_prime_deg_str)

elif eta!=None:
    eta_str = r'$\eta$ = ' + str(eta)
    # Put a legend to the right of the current axis
    legend_title_str = str(isotope_str + '\n' 
                        + gamma_str + '\n' 
                        + K_a_str + '\n' 
                        + K_b_str + '\n' 
                        + K_c_str + '\n' 
                        + v_c_str + '\n' 
                        + eta_str + '\n' 
                        + Hinta_str + '\n'
                        + Hintb_str + '\n'
                        + Hintc_str + '\n'
                        + phi_z_deg_str + '\n' 
                        + theta_x_prime_deg_str + '\n' 
                        + psi_z_prime_deg_str)

h,l = ax.get_legend_handles_labels()
leg = lax.legend(h,l,borderaxespad=0.)
leg.set_title(title=legend_title_str, prop = {'size':'small'})

lax.axis("off")

ax.tick_params(direction='in',bottom=True, top=True, left=True, right=True)

plt.tight_layout()
plt.show()
