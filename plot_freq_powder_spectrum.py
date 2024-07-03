# -- coding: utf-8 --
"""
plot_freq_powder_spectrum
=========
____________
Version: 0.2
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a powder spectrum according to the given parameters.
Additionally a measured data given as a text file can be displayed for comparison.

###############################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
import pySimNMR
import time
import sys
import os.path


###############################################################################
### PARAMETERS ################################################################

##----NMR Parameters-----------------------------------------------------------
n = 1
isotope_list = ['63Cu']             # see isotopeDict.py for names of nuclear species, typically nucleon number and element, unless some special reference, eg As from the Fe-based SC literature
site_multiplicity_list = [1]*n               # scales individual relative intensities of the summed powder patterns
Ka_list = [0.01]*n                      # shift tensor elements (units = percent)
Kb_list = [0.01]*n
Kc_list = [0.01]*n
va_list = [None]*n
vb_list = [None]*n
vc_list = [25.98]*n                          # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [0.01]*n                       # unitless

H0 = 10 # magnetic field  (units = T)

##----Simulation control-------------------------------------------------------
sim_type = 'exact diag'        # options are 'exact diag' and '2nd order'
min_freq = 75                 # units = MHz, note should make this apply for 2nd order as well.
max_freq = 150                # units = MHz
n_freq_points = 1e4            # number of bins for the histogram
convolution_function_list = ['gauss']*n  # 'gauss' and 'lor' are implemented
conv_FWHM_list = [0.5]*n      # correct value here should be scaled down for the lower field... check with hajo          # gaussian or lorentzian of FWHM conv_FWHM (units = MHz)
conv_vQ_FWHM_list = [0.5]*n              # gaussian or lorentzian FWHM which is scaled by transition number for broadening caused by distribution of EFG values (units = MHz)
mtx_elem_min = 0.01             # exact diagonalization only. In general 0.5 is a good starting point. This may cause issues with conv_vQ_FWHM, can resolve by increasing mtx_elem_min
recalc_random_samples = False     # if True, calculate fresh random angle sampling; if False, use the samples from saved (only relevant for exact diag)
n_samples = 1e5                 # good exact diag powder spectra at roughly 1e5 (for very fast calcs). 2nd ord pert much faster so 1e7 possible
                                # NOTE: the saved binary files are actually the stacked arrays of rotation matrices and can get quite large (hundreds of MB) and 
                                # memory issues can arrise here...

##----Background control-------------------------------------------------------
bgd = [0]          #[0] = no background
                   #[offset] = constant background
                   #[offset, slope] = linear background
                   #[center, width, intensity] = gaussian background

##----Plotting data from data File---------------------------------------------
exp_data_file = '' #D:\ifw\code\python\pySimNMR\development\Haase_three_halves_+51.0433.txt'      # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as frequency (MHz), second as intensity
number_of_header_lines = 1        # number of lines which are ignored in the begining of the data file
exp_data_delimiter = ','          # tell numpy which delimter your experimental data file has 
exp_x_scaling = 1               # to scale the experimental data to MHz if units don't match
exp_y_scaling = 0.03

##----Plot control-------------------------------------------------------------
plot_individual_bool = True
plot_sum_bool = True                     # plot the summation of the individual spectra
plot_legend_width_ratio = [3.25, 1]
x_low_limit = min_freq
x_high_limit = max_freq
y_low_limit = 0
y_high_limit = 1.1

##----Exporting simulated spectrum---------------------------------------------
sim_export_file = '' # r'D:\gd\data_and_analysis\TPS3\NiPS3\31P_PP_sim_for_compare_to_torres_1.85T.txt'              # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''

###############################################################################
###############################################################################











#sanity checks
if sim_type == 'exact diag' and n_samples >= 2e5 and recalc_random_samples:
    print('You are attempting to run an exact diag sim with n_samples >= 2e5')
    print('while generating new random samples. This may take several minutes')
    print('and cause your computer to run out of memory or become unresponsive.')
    while True:
        try:
            user_input = input('Do you want to continue with this madness? [y/n]')
            if user_input == 'y':
                print('running the simulation')
                break
            elif user_input == 'n':
                print('user canceled the simulation')
                sys.exit()
            else:
                print("please type 'y' for yes, or 'n' for no, and then press enter.")
                raise Exception
        except Exception:
            continue
                
        

if sim_type == '2nd order' and n_samples > 2e7:
    print('You are attempting to run an 2nd order sim with n_samples >= 2e7.')
    print('This may take a long time, cause your computer to run out of')
    print('memory, or become unresponsive.')
    while True:
        try:
            user_input = input('Do you want to continue with this madness? [y/n]')
            if user_input == 'y':
                print('running the simulation')
                break
            elif user_input == 'n':
                print('user canceled the simulation')
                sys.exit()
            else:
                print("please type 'y' for yes, or 'n' for no, and then press enter.")
                raise Exception
        except Exception:
            continue

def listFloat(lst):
    lst = [float(i) for i in lst]
    return lst

def listInt(lst):
    lst =[int(i) for i in lst]
    return lst

#ensure variables are of correct type:
site_multiplicity_list = listFloat(site_multiplicity_list)
vc_list = listFloat(vc_list)
eta_list = listFloat(eta_list)
Ka_list = listFloat(Ka_list)
Kb_list = listFloat(Kb_list)
Kc_list = listFloat(Kc_list)
H0 = float(H0)
min_freq = float(min_freq)
max_freq = float(max_freq)
n_freq_points = int(n_freq_points)
conv_FWHM_list = listFloat(conv_FWHM_list)
conv_vQ_FWHM_list = listFloat(conv_vQ_FWHM_list)
mtx_elem_min = float(mtx_elem_min)
n_samples = int(n_samples)
number_of_header_lines = int(number_of_header_lines)
exp_x_scaling = float(exp_x_scaling)
exp_y_scaling = float(exp_y_scaling)

# setting the boolean variable for saving the simulation output or not
if sim_export_file == '':
    save_files_bool = False
else:
    save_files_bool = True

if recalc_random_samples == False and sim_type == 'exact diag':
    r_ex_bool = os.path.isfile('r.npy') 
    ri_ex_bool = os.path.isfile('ri.npy') 
    SR_ex_bool = os.path.isfile('SR.npy') 
    SRi_ex_bool = os.path.isfile('SRi.npy') 
    if not (r_ex_bool and ri_ex_bool and SR_ex_bool and SRi_ex_bool):
        print('Could not find the rotation matrix npy files. Generating new random samples...')
        recalc_random_samples = True

i = 0
pp_ind_list = []
gamma_list = []
pp_ind_name_list = []

for isotope in isotope_list:
    # instantiate the simulation class
    sim = pySimNMR.SimNMR(isotope)
    gamma_list.append(sim.isotope_data_dict[isotope]["gamma"])
    
    #
    sim_export_file_single = sim_export_file + '_' + isotope + '_' + str(i)
    # run the simulation
    if sim_type == 'exact diag':
        print('Running in Exact Diagonalization Mode')
        # calculate (and save) or load uniform random sample of angles
        # note, have not allowed for saving with 2nd order pert...
        #######################################
        t0 = time.time()
        ###################
        if recalc_random_samples:
            print('calculating and saving random samples for exact diagonalization...')
        
            phi_z_array = np.random.uniform(0, 2*np.pi, size=int(n_samples))
            theta_xp_array = np.arccos(np.random.uniform(1, -1, size=int(n_samples)))
            psi_zp_array = np.random.uniform(0, 2*np.pi, size=int(n_samples))

            # use the built in SimNMR methods to generate the rotation matrices
            # lower case r matrices are for rotation of the shift tensor
            # capital R matrices are for rotation of the quadrupole Hamiltonian
            r, ri = sim.generate_r_matrices(phi_z_array,
                                            theta_xp_array,
                                            psi_zp_array)
            SR, SRi = sim.generate_SR_matrices(phi_z_array,
                                               theta_xp_array,
                                               psi_zp_array)

            np.save('r.npy', r)
            np.save('ri.npy', ri)
            np.save('SR.npy', SR)
            np.save('SRi.npy', SRi) #the files are saved in the working directory of the sim software

            ###################
            t1 = time.time()
            dt_str = str(t1-t0) + ' s'
            print('random sample calculation took ' + dt_str)
            #######################################
        else:
            print('loading random samples for exact diagonalization...')
            
            r = np.load('r.npy')
            ri = np.load('ri.npy')
            SR = np.load('SR.npy')
            SRi = np.load('SRi.npy')
        
            ###################
            t1 = time.time()
            dt_str = str(t1-t0) + ' s'
            print('random sample loading took ' + dt_str)
            #######################################
        
        #######################################
        t0 = time.time()
        ###################
        rotation_matrices = (r, ri, SR, SRi)
        
        print('Simulating powder pattern...')
        
        pp = sim.freq_spec_edpp(
                                H0=H0, 
                                Ka=Ka_list[i], 
                                Kb=Kb_list[i], 
                                Kc=Kc_list[i], 
                                va=va_list[i],
                                vb=vb_list[i],
                                vc=vc_list[i],
                                eta=eta_list[i],
                                rm_SRm_tuple=rotation_matrices,
                                mtx_elem_min=mtx_elem_min, 
                                min_freq=min_freq, 
                                max_freq=max_freq,
                                FWHM_MHz=conv_FWHM_list[i],
                                FWHM_dvQ_MHz=conv_vQ_FWHM_list[i],
                                broadening_func=convolution_function_list[i],
                                baseline=0.5,
                                nbins=n_freq_points,
                                save_files_bool=save_files_bool,
                                out_filename=sim_export_file_single)
    elif sim_type=='2nd order':
        print('Running in 2nd Order Perturbation Theory Mode')
        #######################################
        t0 = time.time()
        ###################
        print('Simulating powder pattern...')
        pp = sim.sec_ord_freq_pp(
                                H0=H0,
                                Ka=Ka_list[i],
                                Kb=Kb_list[i],
                                Kc=Kc_list[i],
                                vQ=vc_list[i],
                                eta=eta_list[i],
                                nrands=int(n_samples),
                                nbins=n_freq_points,
                                min_freq=min_freq,
                                max_freq=max_freq,
                                broadening_func=convolution_function_list[i],
                                FWHM_MHz=conv_FWHM_list[i],
                                FWHM_dvQ_MHz=conv_vQ_FWHM_list[i],
                                save_files_bool=save_files_bool,
                                out_filename=sim_export_file_single,
                                baseline=0.5)
    else:
        print("please set sim_type = 'exact diag' or sim_type = '2nd order'")
        sys.exit()

    pp[:,1] = pp[:,1]*site_multiplicity_list[i]
    pp_ind_list.append(pp)
    
    pp_ind_name_list.append(isotope + '_{}'.format(i))
    i=i+1

###################
t1 = time.time()
dt_str = str(t1 - t0) + ' s'
print('simulation took ' + dt_str)
#######################################

# prepare for plotting
plt.rcParams["figure.figsize"] = 10,4.5 # width,height in inches
#fig = plt.figure()
#ax = fig.add_subplot(111) 
fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file != '':
    print('plotting experimental data for comparison')
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:,0]*exp_x_scaling
    exp_y = exp_data[:,1]
    # normalize
    exp_y = exp_y_scaling*exp_y/(exp_y.max())
    # plot experimental data first as black lines
    ax.plot(exp_x, exp_y, "k-")

# prepare simulation data for plotting
for n in range(len(pp_ind_list)):
    if n == 0:
        pp_sum = np.copy(pp_ind_list[n])
    else:
        pp_sum[:, 1] = pp_sum[:, 1] + pp_ind_list[n][:, 1]

sim_x = pp_sum[:, 0]
sim_y = pp_sum[:, 1]

#normalize
max_sum_value = sim_y.max()
sim_y = sim_y/max_sum_value

# background correction
if len(bgd) == 1:
    sim_y = (sim_y+bgd[0])/(max(sim_y)+bgd[0])
if len(bgd) == 2:
    sim_y = bgd[1] + sim_x*bgd[2] + sim_y
    sim_y = sim_y/max(sim_y)
if len(bgd) == 3:
    corr = (1.0/(np.sqrt(2*np.pi)*bgd[1]))*np.exp(-(sim_x-bgd[0])**2/(2*bgd[1]**2))
    corr = corr/max(corr)*bgd[2]
    sim_y = corr + sim_y
    sim_y = sim_y/max(sim_y)

if save_files_bool:
    if sim_export_file[-4:] == '.txt':
        sim_export_filename_str = sim_export_file[:-4] + '_pp.txt'
    else:
        sim_export_filename_str = sim_export_file + '_pp.txt'
    spec_out = np.column_stack((sim_x, sim_y))
    np.savetxt(sim_export_filename_str, spec_out)

# plot and save individual spectra if there are more than one
if len(pp_ind_list) > 1:
    for n in range(len(pp_ind_list)):
        if plot_individual_bool:
            ax.fill(pp_ind_list[n][:, 0], 
                    pp_ind_list[n][:,1]/max_sum_value,
                    label=pp_ind_name_list[n],
                    linewidth=2,
                    alpha=0.4)
        if save_files_bool:
            ind_spec_out = np.column_stack((pp_ind_list[n][:, 0], pp_ind_list[n][:, 1]/max_sum_value))
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
H0_str = r'$H_0$ = ' + str(H0) + r' T, '

isotope_str = 'isotopes = ' + str(isotope_list)
gamma_str = r'$\gamma = $ ' + str(gamma_list) + r' MHz/T'
K_a_str = r'$K_a$ = ' + str(Ka_list) + r' %'
K_b_str = r'$K_b$ = ' + str(Kb_list) + r' %'
K_c_str = r'$K_c$ = ' + str(Kc_list) + r' %'
v_c_str = r'$\nu_c$ = ' + str(vc_list) + r' MHz'
eta_str = r'$\eta$ = ' + str(eta_list)
convfunc_str = 'conv.func. = ' + str(convolution_function_list)
conv_FWHM_str = r'FWHM = ' + str(conv_FWHM_list) + r' MHz'

title_str = 'Frequency-swept spectrum; ' + H0_str
ax.set_title(title_str)

# Put a legend to the right of the current axis
legend_title_str = str(isotope_str + '\n' 
+ K_a_str + '\n' 
+ K_b_str + '\n' 
+ K_c_str + '\n' 
+ v_c_str + '\n' 
+ eta_str + '\n' 
+ convfunc_str + '\n'
+ conv_FWHM_str)

# + gamma_str + '\n' 
# + conv_vQ_FWHM_str + '\n'
# + Hinta_str + '\n'
# + Hintb_str + '\n'
# + Hintc_str + '\n'
# + phi_z_deg_str + '\n' 
# + theta_x_prime_deg_str + '\n' 
# + psi_z_prime_deg_str)

h,l = ax.get_legend_handles_labels()
leg = lax.legend(h,l,borderaxespad=0.)
leg.set_title(title=legend_title_str, prop = {'size':'small'})

lax.axis("off")

ax.tick_params(direction='in',bottom=True, top=True, left=True, right=True)

plt.tight_layout()
plt.show()
