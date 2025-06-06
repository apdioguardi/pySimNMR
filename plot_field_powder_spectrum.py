# -- coding: utf-8 --
"""
plot_field_powder_spectrum
=========
____________
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a field-swept powder spectrum according to the given parameters.
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
isotope_list = ['51V']                # need to add many nuclei to the dictionary still... program will fail if not added at this stage of development
site_multiplicity_list = [1]               # scales individual relative intensities of the summed powder patterns

Ka_list = [1.65]                        # shift tensor elements (units = percent)
Kb_list = [1.0]
Kc_list = [0.8]

vQ_list = [0.445]                          # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [0.7]                       # unitless
vc_list = [np.round(vQ/np.sqrt(1 - eta**2/3), 3) for vQ,eta in zip(vQ_list, eta_list)]      

Hinta_list = [0]
Hintb_list = [0]
Hintc_list = [0]

f0 = 34.75        # observed frequency (units = MHz)

##----Simulation control-------------------------------------------------------
sim_type = 'exact diag' # '2nd order'        # '2nd order' or 'exact diag'
min_field = 2.85                 # units = T
max_field = 3.25               # units = T
n_field_points = 250            # number of points on the x (field) axis
delta_f0 = 0.015                      #frequency tolerance (+-delta_f0/2) freqs will be kept during exact diagonalization. may need to play with this in relation to the broadening functions below. if spectra is broad making this larger will make the powder pattern less noisy
convolution_function_list = ['gauss']  # 'gauss' and 'lor' are implemented
conv_FWHM_list = [0.005]                # gaussian or lorentzian of FWHM conv_FWHM (units = MHz)
conv_vQ_FWHM_list = [0.1e-7]              # gaussian or lorentzian FWHM which is scaled by transition number for broadening caused by distribution of EFG values (units = MHz)
mtx_elem_min = 0.5              # exact diagonalization only. In general 0.5 is a good starting point. This may cause issues with conv_vQ_FWHM, can resolve by increasing mtx_elem_min
recalc_random_samples = False     # if True, calculate fresh random angle sampling; if False, use the samples from saved (only relevant for exact diag)
n_samples = 1e3                 # field-swept pp will be significantly slower than freq swept until i figure out how to double-cast"
                                # NOTE: the saved hdf5 file contains the stacked arrays of rotation matrices and can get quite large (hundreds of MB or more) and 
                                # memory issues can arrise here...

##----Background control-------------------------------------------------------
bgd = [0]          #[0] = no background
                                #[offset] = constant background
                                #[offset, slope] = linear background
                                #[center, width, intensity] = gaussian background

##----Plotting data from data File---------------------------------------------
exp_data_file='Intensity_field_sweep_H=2.868to3.2T_2d_ft_integral_field_nmrresponse.txt'#'B11_HS_47p213MHz_4K_ei_H++.txt'                # if you want to plot data also, enter the path to the file here, otherwise write datafile='';
#exp_data_file='B11_HS_49p277MHz_4K_ei_H++.txt'  
number_of_header_lines=1        # number of lines which are ignored in the begining of the data file
exp_data_delimiter=' '          # tell numpy which delimter your experimental data file has 
exp_data_x_column_index = 0           # column index of the independent variable (field)
exp_data_y_column_index = 1           # column index of the dependent variable (intensity)
exp_x_scaling=1               # to scale the experimental data to Tesla if units don't match
exp_y_scaling=1

##----Plot control-------------------------------------------------------------
plot_individual_bool=True
plot_sum_bool=True                     # plot the summation of the individual spectra
plot_legend_width_ratio=[3.25,1]
x_low_limit=min_field
x_high_limit=max_field
y_low_limit=0
y_high_limit=1.1

##----Exporting simulated spectrum---------------------------------------------
sim_export_file= ''  #'B11_HS_49p277MHz_4K_ei_H++'          # if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''

###############################################################################
###############################################################################



















#sanity checks
if sim_type=='exact diag' and n_samples>=2e5 and recalc_random_samples:
    print('You are attempting to run an exact diag sim with n_samples >= 2e5')
    print('while generating new random samples. This may take several minutes')
    print('and cause your computer to run out of memory or become unresponsive.')
    user_input = input('Do you want to continue with this madness? [y/n]')
    if user_input=='y':
        print('continuing with simulation')
    elif user_input=='n':
        print('user canceled the simulation')
        sys.exit()

if sim_type=='2nd order' and n_samples>2e7:
    print('You are attempting to run an 2nd order sim with n_samples >= 2e7.')
    print('This may take a long time, cause your computer to run out of')
    print('memory, or become unresponsive.')
    user_input = input('Do you want to continue with this madness? [y/n]')
    if user_input=='y':
        print('continuing with simulation')
    elif user_input=='n':
        print('user canceled the simulation')
        sys.exit()

def listFloat(lst):
    lst=[float(i) for i in lst]
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
f0 = float(f0)
min_field = float(min_field)
max_field = float(max_field)
n_field_points = int(n_field_points)
conv_FWHM_list = listFloat(conv_FWHM_list)
conv_vQ_FWHM_list = listFloat(conv_vQ_FWHM_list)
mtx_elem_min = float(mtx_elem_min)
n_samples = int(n_samples)
number_of_header_lines = int(number_of_header_lines)
exp_x_scaling = float(exp_x_scaling)
exp_y_scaling = float(exp_y_scaling)

# setting the boolean variable for saving the simulation output or not
if sim_export_file=='':
    save_files_bool=False
else:
    save_files_bool=True

# calculate (and save) or load uniform random sample of angles via rotation matrices
if sim_type == 'exact diag':
    #######################################
    t0 = time.time()
    ###################
    print('field sweep exact diagonalization not yet fully implemented. this is a test:')
    print('Running in Exact Diagonalization Mode')
    sim = pySimNMR.SimNMR('1H') # here the isotope does not matter as the isotope list will be fed into the function below
    rotation_matrix_dict = sim.random_rotation_matrices(isotope_list,
                                                recalc_random_samples=recalc_random_samples,
                                                n_samples=n_samples)

i = 0
pp_ind_list = []
gamma_list=[]
I0_list=[]
pp_ind_name_list=[]

for isotope in isotope_list:
    # instantiate the simulation class
    sim = pySimNMR.SimNMR(isotope)
    gamma_list.append(sim.isotope_data_dict[isotope]["gamma"])
    I0_list.append(sim.isotope_data_dict[isotope]["I0"])
    sim_export_file_single=f'{sim_export_file}_{isotope}_{i}'
    # run the simulation
    if sim_type=='exact diag':
        I0_string = sim.isotope_data_dict[isotope]['I0_string']
        print('Simulating powder pattern...')
        r = rotation_matrix_dict['real_space']['r']
        ri = rotation_matrix_dict['real_space']['ri']
        r_spin = rotation_matrix_dict['spin_space'][f'I0={I0_string}']['r_spin']
        ri_spin = rotation_matrix_dict['spin_space'][f'I0={I0_string}']['ri_spin']
        rotation_matrices = (r, ri, r_spin, ri_spin)

        va_list = [None]*len(vc_list)  # only need eta here so just make these transparently the same length of none lists
        vb_list = [None]*len(vc_list)  # 

        H0_array = np.linspace(start=min_field, stop=max_field, num=n_field_points)

        pp = sim.field_spec_edpp_loop_parallel(
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
                                      FWHM=conv_FWHM_list[i],
                                      FWHM_dvQ=conv_vQ_FWHM_list[i],
                                      broadening_func=convolution_function_list[i],
                                      save_files_bool=False,
                                      out_filename=sim_export_file
                                     )
    elif sim_type=='2nd order':
        print('Running in 2nd Order Perturbation Theory Mode')
        #######################################
        t0 = time.time()
        ###################
        print('Simulating powder pattern...')
        pp = sim.sec_ord_field_pp(
                                I0=I0_list[i],
                                gamma=gamma_list[i],
                                f0=f0,
                                Ka=Ka_list[i],
                                Kb=Kb_list[i],
                                Kc=Kc_list[i],
                                vQ=vc_list[i],
                                eta=eta_list[i],
                                nrands=int(n_samples),
                                nbins=n_field_points,
                                min_field=min_field,
                                max_field=max_field,
                                broadening_func=convolution_function_list[i],
                                FWHM_T=conv_FWHM_list[i]
                                )
    else:
        print("please set sim_type = 'exact diag' or sim_type = '2nd order'")
        sys.exit()

    pp[:,1] = pp[:,1]*site_multiplicity_list[i]
    pp_ind_list.append(pp)
    
    pp_ind_name_list.append(isotope + '_{}'.format(i))
    i=i+1

###################
t1 = time.time()
dt_str = str(t1-t0) + ' s'
print('simulation took ' + dt_str)
#######################################

# prepare for plotting
plt.rcParams["figure.figsize"] = 10,4.5 # width,height in inches
#fig = plt.figure()
#ax = fig.add_subplot(111) 
fig, (ax,lax) = plt.subplots(ncols=2,gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file!='':
    print('plotting experimental data for comparison')
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:,exp_data_x_column_index]*exp_x_scaling
    exp_y = exp_data[:,exp_data_y_column_index]
    # normalize
    exp_y=exp_y_scaling*exp_y/exp_y.max()
    # plot experimental data first as black lines
    ax.plot(exp_x,exp_y,"k-")

# prepare simulation data for plotting
for n in range(len(pp_ind_list)):
    if n==0:
        pp_sum = np.copy(pp_ind_list[n])
    else:
        pp_sum[:,1] = pp_sum[:,1] + pp_ind_list[n][:,1]

sim_x = pp_sum[:,0]
sim_y = pp_sum[:,1]
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
    if sim_export_file[-4:]=='.txt':
        sim_export_filename_str=sim_export_file[:-4]+'_pp.txt'
    else:
        sim_export_filename_str = sim_export_file+'_pp.txt'
    spec_out = np.column_stack((sim_x,sim_y))
    np.savetxt(sim_export_filename_str, spec_out)

# plot and save individual spectra if there are more than one
if len(pp_ind_list)>1:
    for n in range(len(pp_ind_list)):
        if plot_individual_bool:
            ax.plot(pp_ind_list[n][:,0],pp_ind_list[n][:,1]/max_sum_value,label=pp_ind_name_list[n])
        if save_files_bool:
            ind_spec_out = np.column_stack((pp_ind_list[n][:,0],pp_ind_list[n][:,1]/max_sum_value))
            ind_filename_out = sim_export_filename_str[:-4] + '_' + isotope_list[n] + '_{}.txt'.format(n)
            np.savetxt(ind_filename_out, ind_spec_out)

# plot simulation as red lines
if plot_sum_bool==True:
    ax.plot(sim_x,sim_y,"r-",label='total') 

# label the axes
ax.set_xlabel('Field (T)')
ax.set_ylabel('Intensity (arb. units)')

#set axes range
ax.set_xlim(x_low_limit, x_high_limit)
ax.set_ylim(y_low_limit, y_high_limit)

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

title_str = 'Field-swept spectrum; ' + f0_str
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
