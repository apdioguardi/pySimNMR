# -- coding: utf-8 --
"""
plot_freq_spectrum_multisite
=========
____________
Version: 0.2
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a single crystal exact diagonalization or 2nd order perturbation theory
frequency spectrum according to the given parameters. Additionally, a measured
spectrum from a text file can be displayed for comparison.

To do:
- add least squares min fitting

- I = 1/2 nuclei give an error here with exact diag mode... need to fix this:
     Traceback (most recent call last):
       File "plot_freq_spectrum_multisite.py", line 270, in <module>
         out_filename=sim_export_file
       File "/Users/apd/gd/code/python/IFW/pySimNMR/v0.10/pySimNMR.py", line 1359, in freq_spec_edpp
         print('f_calc = ' + str(freq_array[2]))
     IndexError: index 2 is out of bounds for axis 0 with size 1
###############################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pySimNMR
import time
import sys
import os


###############################################################################
### PARAMETERS ################################################################

##----NMR Parameters-----------------------------------------------------------
n = 1 #number of sites, to make testing easier (lists can be multiplied by an integer n, giving a list with n identical elements)
isotope_list = ['73Ge'] #63Ge #I use 11.285 MHz/T from Arneil with 0.2394 % shift for copper metal
site_multiplicity_list = [1]*n  # scales individual relative intensities
Ka_list = [0]*n                # shift tensor elements (units = percent)
Kb_list = [0]*n
Kc_list = [0]*n
va_list = [None]*n  # only functions with exact diag; two modes: va and vb=None and eta=number OR 
vb_list = [None]*n  # only functions with exact diag; va and vb=numbers and eta=None (be sure to satisfy va+vb+vc=0)
vc_list = [0.25]*n     # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [0.7]*n  # asymmetry parameter (unitless)
H0 = 8.0           # magnetic field  (units = T)
Hinta_list = [0]*n  # internal field in a direction (units = T)
Hintb_list = [0]*n  # ... b
Hintc_list = [0]*n  # ... c these are only taken into account in exact diag

#----2nd-order-specific inputs-------------------------------------------------
phi_deg_list = [0]*n 
theta_deg_list = [35]*n

#----exact diag specific inputs------------------------------------------------
mtx_elem_min = 0.5                     # minimum allowed value for the probability of the transition (arbitrary units). Increase to remove forbidden transitions.
phi_z_deg_list =         [0]*n           # Range: (0-360) ZXZ Euler angles phi, theta, and psi for rotation of the EFG + K tensors with respect to H0
theta_x_prime_deg_list = [35]*n # Range: (0-180) these values are in degrees and converted to radians in the code
psi_z_prime_deg_list =   [0]*n    # Range: (0-360)

##----Simulation control-------------------------------------------------------
# sim_type mode options are either 'exact diag' or '2nd order', internal fields are taken into account in 'exact diag' mode
# 2nd oder should not be used if \nu_L ~ \nu_Q. For high-spin nuclei (eg 115In) see https://doi.org/10.1103/PhysRev.145.302 for discussion of 2nd- and 3rd-order effects. 3rd order effects begin to manifest at approximately \nu_L/\nu_Q ~ 5, and then ed should be used.

#sim_type = 'exact diag'
sim_type = '2nd order'

cent_freq = 1.5*H0
min_freq = cent_freq - 0.6       # units = MHz
max_freq = cent_freq + 0.6       # units = MHz
n_freq_points = 1e3                    # number of bins for the histogram
convolution_function_list = ['gauss']*n   # 'gauss' and 'lor' are implemented
conv_FWHM_list = [0.01]*n      # Gaussian or Lorentzian of FWHM conv_FWHM (units = MHz)
conv_vQ_FWHM_list = [1e-6]*n    # Gaussian or Lorentzian FWHM which is scaled by transition number for broadening caused by distribution of EFG values (units = MHz)

##----Background control-------------------------------------------------------
bgd = [0]   #[0] = no background
            #[offset] = constant background
            #[offset, slope] = linear background
            #[center, width, intensity] = gaussian background

##----Plotting data from data File---------------------------------------------
#cwd=os.getcwd()
#exp_data_file = 'D:\\Adam\\Seafile\\ifw\\data\\LiFeAs\\toyoda_2018_LiFeAs_75As_180K_H0parFeFe_spec.txt'   # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as frequency (MHz), second as intensity
exp_data_file=''
number_of_header_lines = 0                # number of lines which are ignored in the begining of the data file
exp_data_delimiter = ' '                 # tell numpy which delimter your experimental data file has 
missing_values_string = 'nan'
exp_x_scaling = 1                     # to scale the experimental data to MHz if units don't match
exp_y_scaling = 1

##----Plot control-------------------------------------------------------------
plot_individual_bool = True
plot_sum_bool = True                     # plot the summation of the individual spectra
plot_legend_width_ratio = [3.25,1]
x_low_limit = min_freq
x_high_limit = max_freq
y_low_limit = 0
y_high_limit = 1.1

##----Exporting Simulated Spectrum---------------------------------------------
sim_export_file='fit_test_input_data_2ndord.txt'    # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''

###############################################################################
###############################################################################
# 0.075 T
# freq_spec_ed; freq_array = [0.69344074 0.49266436 0.51590857]
# freq_array 2nd order =     [0.6464687  0.49132494 0.45078439]

# 0.085 T
# freq_spec_ed; freq_array = [0.76209857 0.57212001 0.58091727]
# freq_array 2nd order =     [0.71961891 0.57121651 0.52393459]

# 0.1 T
# freq_spec_ed; freq_array = [0.86646358 0.68906905 0.68153087]
# freq_array 2nd order =     [0.82934422 0.68852586 0.6336599]

# 0.15 T
# freq_spec_ed; freq_array = [1.2211497  1.06875722 1.03042906]
# freq_array 2nd order =     [1.19509525 1.06860229 0.99941093]

# 0.2 T
# freq_spec_ed; freq_array = [1.58088506 1.44158036 1.38803081]
# freq_array 2nd order =     [1.56084628 1.44151602 1.36516196]  

# 0.3 T
# freq_spec_ed; freq_array = [2.30603977 2.18019961 2.11162555]
# freq_array 2nd order =     [2.29234834 2.18018078 2.09666402]

# 0.4 T
# freq_spec_ed; freq_array = [3.03424301 2.9152721  2.83927556]
# freq_array 2nd order =     [3.0238504  2.91526419 2.82816608]

# 0.5 T
# freq_spec_ed; freq_array = [3.76372581 3.6489191  3.56850101]
# freq_array 2nd order =     [3.75535246 3.64891506 3.55966815]

# 0.75 T
# freq_spec_ed; freq_array = [5.58974223 5.48053649 5.39426247]
# freq_array 2nd order =     [5.3884233  5.48053529 5.58410761]

# 1 T
# freq_spec_ed; freq_array = [7.41710827 7.31072349 7.22153907]
# freq_array 2nd order =     [7.21717845 7.31072299 7.41286277]
# 1.5 T
# freq_spec_ed; freq_array = [11.07321637 10.96966598 10.87758324]
# freq_array 2nd order =     [11.07037307 10.96966583 10.87468875]

# 3 T
# freq_spec_ed; freq_array = [22.04433208 21.9436293  21.84866057]
# freq_array 2nd order =     [22.04290398 21.94362928 21.84721966]

# 7.5 T
# freq_spec_ed; freq_array = [54.96106949 54.86208154 54.76538723]
# freq_array 2nd order =     [54.96049671 54.86208154 54.76481239] 

# 15 T
# freq_spec_ed; freq_array = [109.82343791 109.7250226  109.62775411]
# freq_array 2nd order =     [109.82315126 109.7250226  109.62746695]

# 30 T
# freq_spec_ed; freq_array = [219.54860376 219.45047496 219.35291957]
# freq_array 2nd order =     [219.54846037 219.45047496 219.35277605]

# 100 T
# freq_spec_ed; freq_array = [731.59994589 731.50201773 731.40426159]
# freq_array 2nd order =     [731.59990286 731.50201773 731.40421855  ]

# 1000 T
# freq_spec_ed; freq_array = [7315.11845352 7315.0206027 6 7314.92276921]
# freq_array =               [7315.11844922 7315.02060276 7314.9227649]



def float_try_none(input_value):
    try:
        output_value=float(input_value)
    except TypeError:
        output_value=input_value
    return output_value

def listFloat(lst):
    lst=[float_try_none(i) for i in lst]
    return lst

def listInt(lst):
    lst =[int(i) for i in lst]
    return lst

#ensure variables are of correct type:
site_multiplicity_list = listFloat(site_multiplicity_list)
Ka_list = listFloat(Ka_list)
Kb_list = listFloat(Kb_list)
Kc_list = listFloat(Kc_list)
va_list = listFloat(va_list)
vb_list = listFloat(vb_list)
vc_list = listFloat(vc_list)
eta_list = listFloat(eta_list)
H0 = float(H0)
phi_deg_list = listFloat(phi_deg_list) 
theta_deg_list = listFloat(theta_deg_list)  
phi_z_deg_list = listFloat(phi_z_deg_list)
theta_x_prime_deg_list = listFloat(theta_x_prime_deg_list)
psi_z_prime_deg_list = listFloat(psi_z_prime_deg_list)
Hinta_list=listFloat(Hinta_list)
Hintb_list=listFloat(Hintb_list)
Hintc_list=listFloat(Hintc_list)
min_freq = float(min_freq)
max_freq = float(max_freq)
n_freq_points = int(n_freq_points)
conv_FWHM_list = listFloat(conv_FWHM_list)
conv_vQ_FWHM_list = listFloat(conv_vQ_FWHM_list)
mtx_elem_min = float(mtx_elem_min)
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
        gamma = sim.isotope_data_dict[isotope]["gamma"]
        gamma_list.append(gamma)
        I0 = sim.isotope_data_dict[isotope]["I0"]
        print('I0', I0 )
        sim_export_file_single = sim_export_file + '_' + isotope + '_' + str(i)

        #######################################
        t0 = time.time()
        ###################
        
        phi_array = np.array([phi_deg_list[i]])*np.pi/180
        theta_array = np.array([theta_deg_list[i]])*np.pi/180

        spec = sim.sec_ord_freq_spec(
                                    I0=I0,
                                    gamma=gamma,
                                    H0=H0,
                                    Ka=Ka_list[i],
                                    Kb=Kb_list[i],
                                    Kc=Kc_list[i],
                                    vQ=vc_list[i],
                                    eta=eta_list[i],
                                    theta_array=theta_array,
                                    phi_array=phi_array,
                                    nbins=n_freq_points,
                                    min_freq=min_freq,
                                    max_freq=max_freq,
                                    broadening_func=convolution_function_list[i],
                                    FWHM_MHz=conv_FWHM_list[i]
                                    )
        sim_x = spec[:,0]
        spec_y = spec[:,1]*site_multiplicity_list[i]
        spec_ind_list.append(spec_y)
        
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

    i = 0
    spec_ind_list = []
    gamma_list=[]
    spec_ind_name_list=[]
    # create freq array at which to simulate spectrum
    sim_x = np.linspace(min_freq, max_freq, n_freq_points)

    for isotope in isotope_list:
        # instantiate the simulation class
        sim = pySimNMR.SimNMR(isotope)
        gamma_list.append(sim.isotope_data_dict[isotope]["gamma_sigfigs"])
        # use the built in SimNMR methods to generate the rotation matrices
        # lower case r matrices are for rotation of the shift tensor
        #  SR matrices are for spin-space rotation of the quadrupole Hamiltonian
        phi_z_i = np.array([phi_z_array[i]])
        theta_xp_i = np.array([theta_xp_array[i]])
        psi_zp_i = np.array([psi_zp_array[i]])
        r, ri = sim.generate_r_matrices(phi_z_i,
                                        theta_xp_i,
                                        psi_zp_i)
        SR, SRi = sim.generate_SR_matrices(phi_z_i,
                                           theta_xp_i,
                                           psi_zp_i)
        rotation_matrices = (r, ri, SR, SRi)
        #######################################
        t0 = time.time()
        ###################
        spec = sim.freq_spec_ed(
                                x=sim_x,
                                H0=H0,
                                Ka=Ka_list[i], 
                                Kb=Kb_list[i], 
                                Kc=Kc_list[i], 
                                va=va_list[i],
                                vb=vb_list[i],
                                vc=vc_list[i],
                                eta=eta_list[i],
                                rm_SRm_tuple=rotation_matrices,
                                Hinta=Hinta_list[i],
                                Hintb=Hintb_list[i],
                                Hintc=Hintc_list[i],
                                mtx_elem_min=mtx_elem_min,
                                min_freq=min_freq, 
                                max_freq=max_freq,
                                FWHM=conv_FWHM_list[i],
                                FWHM_vQ=conv_vQ_FWHM_list[i],
                                line_shape_func=convolution_function_list[i]
                               )
        spec = spec*site_multiplicity_list[i]
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
plt.rcParams["figure.figsize"] = 10, 4.5 # width,height in inches
fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file != '':
    print('plotting experimental data for comparison')
    exp_data = np.genfromtxt(fname=exp_data_file, 
                             delimiter=exp_data_delimiter, 
                             skip_header=number_of_header_lines, 
                             missing_values=missing_values_string)
    exp_x = exp_data[:, 0]*exp_x_scaling
    exp_y = exp_data[:, 1]
    # normalize
    exp_y=exp_y_scaling*exp_y/np.nanmax(exp_y)
    # plot experimental data first as black lines
    ax.plot(exp_x, exp_y, "k-")
# prepare simulation data for plotting
for n in range(len(spec_ind_list)):
    if n==0:
        spec_sum = np.copy(spec_ind_list[n])
    else:
        spec_sum = spec_sum + spec_ind_list[n]

sim_x = sim_x
sim_y = spec_sum
#print('sim_y', sim_y)
#normalize
max_sum_value = sim_y.max()
#print('max_sum_value', max_sum_value)
sim_y = sim_y/max_sum_value
#print('sim_y normalized', sim_y)

# background correction
if len(bgd) == 1:
    sim_y = (sim_y + bgd[0])/(sim_y.max() + bgd[0])
if len(bgd) == 2:
    sim_y = bgd[1] + sim_x*bgd[2] + sim_y
    sim_y = sim_y/sim_y.max()
if len(bgd) == 3:
    corr = (1.0/(np.sqrt(2*np.pi)*bgd[1]))*np.exp(-(sim_x-bgd[0])**2/(2*bgd[1]**2))
    corr = corr/corr.max()*bgd[2]
    sim_y = corr + sim_y
    sim_y = sim_y/sim_y.max()

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
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Intensity (arb. units)')

#set axes range
ax.set_xlim(x_low_limit, x_high_limit)
ax.set_ylim(y_low_limit, y_high_limit)

# generate strings for the title and legend/details
H0_str = r'$H_0$=' + str(H0) + r'T'

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

title_str = 'Frequency-swept spectrum; ' + H0_str
ax.set_title(title_str)

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
