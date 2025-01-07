# -- coding: utf-8 --
"""
PlotFreqSpectrum
=========
____________
Version: 0.2
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a single crystal exact diagonalization frequency spectrum according to the given parameters.
Additionally a measured data given as a text file can be displayed for comparison.

To do:
add other parameters to legend title
merge with piotr's background function
add least squares min fitting

###############################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pySimNMR
import time
import sys


###############################################################################
### PARAMETERS ################################################################

##----NMR Parameters-----------------------------------------------------------
n=2
isotope_list = ['11B']*n
site_multiplicity_list = [1, 0.5]    # scales individual relative intensities of the summed spectra
Ka_list = [0.1]*n                # shift tensor elements (units = percent)
Kb_list = [0.2]*n
Kc_list = [0.3]*n
va_list = [None]*n  # only functions with exact diag; two modes: va and vb=None and eta=number OR 
vb_list = [None]*n  # only functions with exact diag; va and vb=numbers and eta=None (be sure to satisfy va+vb+vc=0)
vc_list = [0.5]*n     # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [0.7]*n  # asymmetry parameter (unitless)
f0 = 40.813                     # observed frequency (units = MHz)
Hinta_list = [0]*n                # internal field in a direction (units = T)
Hintb_list = [0]*n                # ... b
Hintc_list = [0]*n                # ... c

#----2nd order specific inputs-------------------------------------------------
phi_deg_list = [0]*n
theta_deg_list = [0]*n

#----exact diag specific inputs------------------------------------------------
delta_f0 = 0.0016    # range over which duplicate transitions will be kept. this value should be ~< the broadening/splitting between
                     # transitions. Shold revisit this parameters/the method surrounding it there must be a better way to control this. 
                     # likely using the mixing coefficients of the transitions we can get this working better.
mtx_elem_min = 0.1              # see note below following the input parameters marked with #*! for one type of error related to this value
phi_z_deg_list = [0, 0]            # Range: (0-360) ZXZ Euler angles phi, theta, and psi for rotation of the EFG + K tensors with respect to H0
theta_x_prime_deg_list = [90, 90]    # Range: (0-180) these values are in degrees and converted to radians in the code
psi_z_prime_deg_list = [0, 90]      # Range: (0-360)

##----Simulation control-------------------------------------------------------
#sim_type mode options are either 'exact diag' or '2nd order', internal fields are taken into account in 'exact diag' mode
#sim_type = '2nd order' # ahoy!  2nd order broken, need to check it    
sim_type = 'exact diag'
min_field = 2.87                                   # units = T
max_field = 3.11                                 # units = T
n_field_points = 2000                           # number of bins for the histogram
convolution_function_list = ['gauss']*n   # 'gauss' and 'lor' are implemented
conv_FWHM_list = [0.002]*n      # gaussian or lorentzian of FWHM conv_FWHM (units = T)
conv_vQ_FWHM_list = [0.003]*n   #gaussian or lorentzian FWHM which is scaled by transition number for broadening caused by distribution of EFG values (units = T)

##----Plotting data from data File---------------------------------------------
exp_data_file=''   # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as frequency (MHz), second as intensity
number_of_header_lines=1            # number of lines which are ignored in the begining of the data file
exp_data_delimiter='\t'             # tell numpy which delimter your experimental data file has 
exp_x_scaling=0.1                     # to scale the experimental data to Tesla if units don't match
exp_y_scaling=1
exp_y_offset=0

##----Plot control-------------------------------------------------------------
plot_individual_bool=True
plot_sum_bool=True                  # plot the summation of the individual spectra
plot_legend_width_ratio=[4,1]
x_low_limit=min_field
x_high_limit=max_field
y_low_limit=-0.5    
y_high_limit=1.5
##----Exporting Simulated Spectrum---------------------------------------------
sim_export_file=''#'/Users/apd/gd/code/python/IFW/pySimNMR/v0.10/PuB4_11B_test_2nd_theta90.txt'     # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''
#sim_export_file='/Users/apd/gd/code/python/IFW/pySimNMR/v0.10/PuB4_11B_test_ed_thetaxp90.txt'
###############################################################################
###############################################################################

#*! mtx_elem_min error:
# Traceback (most recent call last):
#   File "plot_field_spectrum_multisite.py", line 254, in <module>
#     out_filename=sim_export_file
#   File "/Users/apd/gd/code/python/IFW/pySimNMR/v0.6/pySimNMR.py", line 829, in field_spec_edpp
#     delta_f0=delta_f0
#   File "/Users/apd/gd/code/python/IFW/pySimNMR/v0.6/pySimNMR.py", line 657, in freq_prob_trans_ed_HS
#     trans_array = trans_array[f0_close_bool_array]
# IndexError: boolean index did not match indexed array along dimension 0; dimension is 3000 but corresponding boolean dimension is 3387
##It seems that one possible solution here would be to catch this error and increase 
##mtx_elem_min automatically until the error goes away... may try to implement this.



def listFloat(lst):
    lst=[float(i) for i in lst]
    return lst

def listInt(lst):
    lst =[int(i) for i in lst]
    return lst

#ensure variables are of correct type:
site_multiplicity_list = listFloat(site_multiplicity_list)
Ka_list = listFloat(Ka_list)
Kb_list = listFloat(Kb_list)
Kc_list = listFloat(Kc_list)
vc_list = listFloat(vc_list)
eta_list = listFloat(eta_list)
f0 = float(f0)
phi_deg_list = listFloat(phi_deg_list)
theta_deg_list = listFloat(theta_deg_list)
min_field = float(min_field)
max_field = float(max_field)
n_field_points = int(n_field_points)
conv_FWHM_list = listFloat(conv_FWHM_list)
# conv_vQ_FWHM_list = listFloat(conv_vQ_FWHM_list)
number_of_header_lines = int(number_of_header_lines)
exp_x_scaling = float(exp_x_scaling)
exp_y_scaling = float(exp_y_scaling)

# setting the boolean variable for saving the simulation output or not
if sim_export_file=='':
    save_files_bool=False
else:
    save_files_bool=True

# run the simulation
if sim_type=='2nd order':
    print('Running in 2nd Order Perturbation Theory Mode')

    i = 0
    spec_ind_list = []
    gamma_list=[]
    spec_ind_name_list=[]

    for isotope in isotope_list:
        # instantiate the simulation class
        sim = pySimNMR.SimNMR(isotope)
        gamma=sim.isotope_data_dict[isotope]["gamma"]
        gamma_list.append(gamma)
        I0=sim.isotope_data_dict[isotope]["I0"]
        #
        sim_export_file_single=sim_export_file+'_'+isotope+'_'+str(i)

        #######################################
        t0 = time.time()
        ###################
        
        phi_array = np.array([phi_deg_list[i]])*np.pi/180
        theta_array = np.array([theta_deg_list[i]])*np.pi/180

        spec = sim.sec_ord_field_spec(
                                    I0=I0,
                                    gamma=gamma,
                                    f0=f0,
                                    Ka=Ka_list[i],
                                    Kb=Kb_list[i],
                                    Kc=Kc_list[i],
                                    vQ=vc_list[i],
                                    eta=eta_list[i],
                                    theta_array=theta_array,
                                    phi_array=phi_array,
                                    nbins=n_field_points,
                                    min_field=min_field,
                                    max_field=max_field,
                                    broadening_func=convolution_function_list[i],
                                    FWHM_T=conv_FWHM_list[i]
                                    )

        spec[:,1] = spec[:,1]*site_multiplicity_list[i]
        spec_ind_list.append(spec)
        
        spec_ind_name_list.append(isotope + '_{}'.format(i))
        i=i+1

    ###################
    t1 = time.time()
    dt_str = str(t1-t0) + ' s'
    print('simulation took ' + dt_str)
    #######################################
elif sim_type=='exact diag':
    # run the simulation
    print('Running in Exact Diagonalization Mode')

    phi_z_array = np.array(phi_z_deg_list)*np.pi/180
    theta_xp_array = np.array(theta_x_prime_deg_list)*np.pi/180
    psi_zp_array = np.array(psi_z_prime_deg_list)*np.pi/180

    H0_array = np.linspace(start=min_field, stop=max_field, num=n_field_points)

    i = 0
    spec_ind_list = []
    gamma_list=[]
    spec_ind_name_list=[]

    for isotope in isotope_list:
        # instantiate the simulation class
        sim = pySimNMR.SimNMR(isotope)
        gamma_list.append(sim.isotope_data_dict[isotope]["gamma"])

        # create arrays with the same shape as the input H0_array
        # could likely do this with proper broadcasting, but for this
        # purpose, the arrays will not be too long, and therefore 
        # this should be efficient enough.
        phi_z_i = np.full(H0_array.shape, [phi_z_array[i]])
        theta_xp_i = np.full(H0_array.shape, [theta_xp_array[i]])
        psi_zp_i = np.full(H0_array.shape, [psi_zp_array[i]])
        # use the built in SimNMR methods to generate the rotation matrices
        # lower case r matrices are for rotation of the shift tensor
        #  SR matrices are for spin-space rotation of the quadrupole Hamiltonian
        r, ri = sim.generate_r_matrices(phi_z_i,
                                        theta_xp_i,
                                        psi_zp_i)
        SR, SRi = sim.generate_r_spin_matrices(phi_z_i,
                                           theta_xp_i,
                                           psi_zp_i)
        rotation_matrices = (r, ri, SR, SRi)

        #######################################
        t0 = time.time()
        ###################

        #ahoy! working here
        spec = sim.field_spec_edpp(
                                    f0=f0, 
                                    H0_array=H0_array,
                                    Ka=Ka_list[i], 
                                    Kb=Kb_list[i], 
                                    Kc=Kc_list[i], 
                                    va=va_list[i],
                                    vb=vb_list[i],
                                    vc=vc_list[i],
                                    eta=eta_list[i],
                                    rotation_matrices=rotation_matrices,
                                    Hinta=Hinta_list[i],
                                    Hintb=Hintb_list[i],
                                    Hintc=Hintc_list[i],
                                    mtx_elem_min=mtx_elem_min, 
                                    min_field=min_field, 
                                    max_field=max_field,
                                    delta_f0=delta_f0,
                                    FWHM_T=conv_FWHM_list[i],
                                    FWHM_dvQ_T=conv_vQ_FWHM_list[i],
                                    broadening_func=convolution_function_list[i],
                                    baseline=0.5,
                                    nbins=n_field_points,
                                    save_files_bool=False,
                                    out_filename=sim_export_file
                               )
        spec[:,1] = spec[:,1]*site_multiplicity_list[i]
        spec_ind_list.append(spec)
        
        spec_ind_name_list.append(isotope + '_{}'.format(i))
        i=i+1

    ###################
    t1 = time.time()
    dt_str = str(t1-t0) + ' s'
    print('simulation took ' + dt_str)
    #######################################
else:
    print("please specify a valid mode; either sim_type='2nd order' or sim_type='exact diag'")
# prepare for plotting
plt.rcParams["figure.figsize"] = 10,4.5 # width,height in inches
#fig = plt.figure()
#ax = fig.add_subplot(111) 
fig, (ax,lax) = plt.subplots(ncols=2,gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file!='':
    print('plotting experimental data for comparison')
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:,0]*exp_x_scaling
    exp_y = exp_data[:,1]
    # normalize
    exp_y=exp_y_scaling*exp_y/(exp_y.max()) + exp_y_offset
    # plot experimental data first as black lines
    ax.plot(exp_x, exp_y, "k-")

# prepare simulation data for plotting
for n in range(len(spec_ind_list)):
    if n==0:
        spec_sum = np.copy(spec_ind_list[n])
    else:
        spec_sum[:,1] = spec_sum[:,1] + spec_ind_list[n][:,1]

sim_x = spec_sum[:,0]
sim_y = spec_sum[:,1]
#normalize
max_sum_value = sim_y.max()
sim_y = sim_y/max_sum_value

# save full spectrum
if save_files_bool:
    if sim_export_file[-4:]=='.txt':
        sim_export_filename_str=sim_export_file
    else:
        sim_export_filename_str = sim_export_file+'.txt'
    spec_out = np.column_stack((sim_x,sim_y))
    np.savetxt(sim_export_filename_str, spec_out)

# plot and save individual spectra if there are more than one
if len(spec_ind_list)>1:
    for n in range(len(spec_ind_list)):
        if plot_individual_bool:
            ax.fill(spec_ind_list[n][:, 0],
                    spec_ind_list[n][:, 1]/max_sum_value,
                    label=spec_ind_name_list[n],
                    linewidth=2,
                    alpha=0.4)
        if save_files_bool:
            ind_spec_out = np.column_stack((spec_ind_list[n][:,0],spec_ind_list[n][:,1]/max_sum_value))
            ind_filename_out = sim_export_filename_str[:-4] + '_' + isotope_list[n] + '_{}.txt'.format(n)
            np.savetxt(ind_filename_out, ind_spec_out)

# plot simulation as red lines
if plot_sum_bool==True:
    ax.plot(sim_x, sim_y, "r-", label='total', linewidth=2, alpha=0.6) 

# label the axes
ax.set_xlabel('Field (T)')
ax.set_ylabel('Intensity (arb. units)')

# generate strings for the title and legend/details
f0_str = r'$f_0$=' + str(f0) + r'MHz'

isotope_str = 'isotopes = ' + str(isotope_list)
gamma_str = r'$\gamma = $ ' + str(gamma_list) + r' MHz/T'
K_a_str = r'$K_a$ = ' + str(Ka_list) + r' %'
K_b_str = r'$K_b$ = ' + str(Kb_list) + r' %'
K_c_str = r'$K_c$ = ' + str(Kc_list) + r' %'
v_c_str = r'$\nu_c$ = ' + str(vc_list) + r' MHz'
eta_str = r'$\eta$ = ' + str(eta_list)
convfunc_str = 'conv.func. = ' + str(convolution_function_list)
conv_FWHM_str = r'FWHM = ' + str(conv_FWHM_list) + r' MHz'
conv_vQ_FWHM_str = r'FWHMdvQ = ' + str(conv_vQ_FWHM_list) + r' MHz'
Hinta_str = r'$H_{int}^{a}$ = ' + str(Hinta_list) + r' T'
Hintb_str = r'$H_{int}^{b}$ = ' + str(Hintb_list) + r' T'
Hintc_str = r'$H_{int}^{c}$ = ' + str(Hintc_list) + r' T'
phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg_list) + r' $\degree$' 
theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_x_prime_deg_list) + r' $\degree$'
psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_z_prime_deg_list) + r' $\degree$'

phi_deg_str = r'$\phi$ = ' + str(phi_deg_list) + r' $\degree$' 
theta_deg_str = r'$\theta$ = ' + str(theta_deg_list) + r' $\degree$'

title_str = 'Field-swept spectrum; ' + f0_str
ax.set_title(title_str)

#set axes range
ax.set_xlim(x_low_limit, x_high_limit)
ax.set_ylim(y_low_limit, y_high_limit)

# Put a legend to the right of the current axis
if sim_type=='2nd order':
    legend_title_str = str(isotope_str + '\n'
    + K_a_str + '\n' 
    + K_b_str + '\n' 
    + K_c_str + '\n' 
    + v_c_str + '\n' 
    + eta_str + '\n' 
    + convfunc_str + '\n'
    + conv_FWHM_str + '\n'
    + conv_vQ_FWHM_str + '\n'
    + Hinta_str + '\n'
    + Hintb_str + '\n'
    + Hintc_str + '\n'
    + phi_deg_str + '\n' 
    + theta_deg_str)
        #+ gamma_str + '\n' 
elif sim_type=='exact diag':
    legend_title_str = str(isotope_str + '\n'
    + K_a_str + '\n' 
    + K_b_str + '\n' 
    + K_c_str + '\n' 
    + v_c_str + '\n' 
    + eta_str + '\n' 
    + convfunc_str + '\n'
    + conv_FWHM_str + '\n'
    + conv_vQ_FWHM_str + '\n'
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
