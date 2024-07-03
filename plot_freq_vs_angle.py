# -- coding: utf-8 --
"""
PlotFreqSpectrumVsAngle
=========
____________
Version: 0.2
Author: Adam P. Dioguardi
------------

This file is part of the pySimNMR project.

Plots a single crystal spectral positions vs angle
according to the given parameters. Additionally a measured data given as a text 
file can be displayed for comparison.

###############################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
import pySimNMR
import time
import sys
import platform


###############################################################################
### PARAMETERS ################################################################

##----NMR Parameters-----------------------------------------------------------
isotope_list = ['73Ge']      #add possibility of directly inputting the gamma here... for esr or some different standard.
n = len(isotope_list)
Ka_list = [0]*n                 # shift tensor elements (units = percent)
Kb_list = [0]*n
Kc_list = [0]*n

va_list = [None]*n              # two modes: va and vb=None and eta=number OR 
vb_list = [None]*n              # va and vb=numbers and eta=None (be sure to satisfy va+vb+vc=0)
vc_list = [0.25]*n             # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [0]*n         # asymmetry parameter (unitless)

H0 = 25                         # magnetic field  (units = T)
                                    # the angles in the following lists are initial condition rotations of the shift tensor, EFG tensor, and Hint vector,
                                    # simulation control below allows one to set the axis of rotation and 

phi_z_deg_init_list = [0]             # Range: (0-360) ZXZ Euler angles phi, theta, and psi for rotation of the EFG + K tensors with respect to H0
theta_x_prime_deg_init_list = [0]*n    # Range: (0-180) these values are in degrees and converted to radians in the code
psi_z_prime_deg_init_list = [0]*n        # Range: (0-360)

Hinta_list = [0]*n                # internal field in a direction (units = T)
Hintb_list = [0]*n                # ... b 0.009748 +- 0.000111
Hintc_list = [0]*n           # ... c

##----Simulation control-------------------------------------------------------
min_freq = 36                # units = MHz
max_freq = 38.5                 # units = MHz
angle_to_vary = 'theta_x_prime_deg'     # 'phi_z_deg', 'theta_x_prime_deg', or 'psi_z_prime_deg'
angle_start = 0
angle_stop = 90              # this range will replace the constant given in the NMR Parameters section
n_angles = 150                  # number of angles to calculate in the output plot
mtx_elem_min = 0.1              # In general, 0.5 is a good starting point

##----Plotting data from data File---------------------------------------------
exp_data_file=''                # if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as angle in degrees, second as frequency
number_of_header_lines=0        # number of lines which are ignored in the begining of the data file
exp_data_delimiter=' '          # tell numpy which delimter your experimental data file has 
angle_offset = 0

##----Plot control-------------------------------------------------------------
plot_legend_width_ratio=[3.25,1]
x_low_limit=angle_start
x_high_limit=angle_stop
y_low_limit=min_freq
y_high_limit=max_freq

##----Exporting Simulated Spectrum---------------------------------------------
# if you want to export your simulation, enter the path to the file here, otherwise write exportfile = ''
sim_export_file='' #'freq_vs_angle_CeRh6Ge4_Ge(1).txt'

###############################################################################
###############################################################################



















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
Ka_list = listFloat(Ka_list)
Kb_list = listFloat(Kb_list)
Kc_list = listFloat(Kc_list)
va_list = listFloat(va_list)
vb_list = listFloat(vb_list)
vc_list = listFloat(vc_list)
eta_list = listFloat(eta_list)
H0 = float(H0)
phi_z_deg_init_list = listFloat(phi_z_deg_init_list)
theta_x_prime_deg_init_list = listFloat(theta_x_prime_deg_init_list)
psi_z_prime_deg_init_list = listFloat(psi_z_prime_deg_init_list)
Hinta_list=listFloat(Hinta_list)
Hintb_list=listFloat(Hintb_list)
Hintc_list=listFloat(Hintc_list)
min_freq = float(min_freq)
max_freq = float(max_freq)
n_angles = int(n_angles)
angle_start = float(angle_start)
angle_stop = float(angle_stop)
mtx_elem_min = float(mtx_elem_min)
number_of_header_lines = int(number_of_header_lines)

# setting the boolean variable for saving the simulation output or not
if sim_export_file=='':
    save_files_bool=False
else:
    save_files_bool=True


angle_array = np.linspace(angle_start*np.pi/180, angle_stop*np.pi/180, int(n_angles))

if angle_to_vary=='phi_z_deg':
    phi_z_array = angle_array
    theta_xp_array = np.full(shape=angle_array.shape, fill_value=0.0)
    psi_zp_array = np.full(shape=angle_array.shape, fill_value=0.0)
elif angle_to_vary=='theta_x_prime_deg':
    theta_xp_array = angle_array
    phi_z_array = np.full(shape=angle_array.shape, fill_value=0.0)
    psi_zp_array = np.full(shape=angle_array.shape, fill_value=0.0)
elif angle_to_vary=='psi_z_prime_deg':
    psi_zp_array = angle_array
    phi_z_array = np.full(shape=angle_array.shape, fill_value=0.0)
    theta_xp_array = np.full(shape=angle_array.shape, fill_value=0.0)
else:
    print("please designate angle_to_vary='phi_z_deg', 'theta_x_prime_deg', or 'psi_z_prime_deg' and try again.")
    print('exiting program...')
    sys.exit()

#######################################
t0 = time.time()
###################

i = 0
afpt_ind_list = []
gamma_list=[]
afpt_ind_name_list=[]

# run the simulation
print('Running in Exact Diagonalization Mode')
for isotope in isotope_list:
    # instantiate the simulation class
    sim = pySimNMR.SimNMR(isotope)
    
    gamma_list.append(sim.isotope_data_dict[isotope]["gamma_sigfigs"])

    # use the built in SimNMR methods to generate the rotation matrices
    # lower case r matrices are for rotation of the shift tensor
    # SR matrices are for spin-space rotation of the quadrupole Hamiltonian
    r, ri = sim.generate_r_matrices(phi_z_array,
                                    theta_xp_array,
                                    psi_zp_array)
    SR, SRi = sim.generate_SR_matrices(phi_z_array,
                                       theta_xp_array,
                                       psi_zp_array)
    rotation_matrices = (r, ri, SR, SRi)

    phi_z_init_array_i = np.array([phi_z_deg_init_list[i]])*np.pi/180
    theta_xp_init_array_i = np.array([theta_x_prime_deg_init_list[i]])*np.pi/180
    psi_zp_init_array_i = np.array([psi_z_prime_deg_init_list[i]])*np.pi/180
    #phi_z_init_array = np.full(shape=angle_array.shape, fill_value=phi_z_deg_init_list[i]*np.pi/180)
    #theta_xp_init_array = np.full(shape=angle_array.shape, fill_value=theta_x_prime_deg_init_list[i]*np.pi/180)
    #psi_zp_init_array = np.full(shape=angle_array.shape, fill_value=psi_z_prime_deg_init_list[i]*np.pi/180)
    
    # rz_init,rzi_init = sim.rz_matrices(phi_z_init_array) #use the built in SimNMR methods to generate the rotation matrices
    # rxp_init,rxpi_init = sim.rx_matrices(theta_xp_init_array)  # lower case r matrices are for rotation of the shift tensor
    # rzp_init,rzpi_init = sim.rz_matrices(psi_zp_init_array)
    # Rz_init,Rzi_init = sim.Rz_matrices(phi_z_init_array)       # capital R matrices are for rotation of the quadurpolar hamiltonian
    # Rxp_init,Rxpi_init = sim.Rx_matrices(theta_xp_init_array)
    # Rzp_init,Rzpi_init = sim.Rz_matrices(psi_zp_init_array)

    # r_init = rzp_init @ rxp_init @ rz_init # @ indicates matrix multiplication np.matmul
    # ri_init = rzi_init @ rxpi_init @ rzpi_init
    # RR_init = Rzp_init @ Rxp_init @ Rz_init
    # RRi_init = Rzi_init @ Rxpi_init @ Rzpi_init

    phi_z_init_i = np.array([phi_z_init_array_i[i]])
    theta_xp_init_i = np.array([theta_xp_init_array_i[i]])
    psi_zp_init_i = np.array([psi_zp_init_array_i[i]])
    r_init, ri_init = sim.generate_r_matrices(phi_z_init_i,
                                              theta_xp_init_i,
                                              psi_zp_init_i)
    SR_init, SRi_init = sim.generate_SR_matrices(phi_z_init_i,
                                                 theta_xp_init_i,
                                                 psi_zp_init_i)
    rotation_matrices_init = (r_init, ri_init, SR_init, SRi_init)

    f,p,t = sim.freq_prob_trans_ed(H0=H0,
                                   Ka=Ka_list[i],
                                   Kb=Kb_list[i],
                                   Kc=Kc_list[i],
                                   va=va_list[i],
                                   vb=vb_list[i],
                                   vc=vc_list[i],
                                   eta=eta_list[i],
                                   rm_SRm_tuple=rotation_matrices,
                                   rm_SRm_init_tuple=rotation_matrices_init,
                                   Hinta=Hinta_list[i],
                                   Hintb=Hintb_list[i],
                                   Hintc=Hintc_list[i],
                                   mtx_elem_min=mtx_elem_min,
                                   min_freq=min_freq,
                                   max_freq=max_freq)

    #get the isotope with which we are working
    I0 = sim.isotope_data_dict[isotope]["I0"]
    # integer of the number of resonances we expect (there may be a better way to get this and return
    # it from the freq_prob_trans_ed function as perhaps a list of integers each associated with a number,
    # this may be a "bug" if we work in the regime where H_q ~= H_z)
    n_resonances=int(I0*2)
    #create an empty numpy array of the appropriate size (default dtype is float)
    angle_array_out = np.empty(n_resonances*angle_array.size)
    #slice this array in a loop such that we set m to the end in steps of n_resonances [start:stop:step]
    # and if stop is left blank it is to the end
    # so the effect is that the first n_resonances indexes are set to the first angle in angle_array
    # second n_resonances are set to the second angle and so on
    for m in range(n_resonances):
        angle_array_out[m::n_resonances]=angle_array*(180/np.pi)
    # print('angle_array_out,f,p,t:')
    # print(angle_array_out,f,p,t)
    afpt=np.column_stack((angle_array_out,f,p,t))

    afpt_ind_list.append(afpt)

    afpt_ind_name_list.append(isotope + '_{}'.format(i))
    i=i+1
###################
t1 = time.time()
dt_str = str(t1-t0) + ' s'
print('simulation took ' + dt_str)
#######################################

# prepare for plotting
plt.rcParams["figure.figsize"] = 10,4.5 # width,height in inches
fig, (ax,lax) = plt.subplots(ncols=2,gridspec_kw={"width_ratios":plot_legend_width_ratio})

# load experimental data for comparison
if exp_data_file!='':
    exp_data = np.genfromtxt(exp_data_file, delimiter=exp_data_delimiter, skip_header=number_of_header_lines)
    exp_x = exp_data[:,0] + angle_offset
    exp_y = exp_data[:,1]
    # plot experimental data first as black squares
    ax.plot(exp_x,exp_y,"ks")

# plot
#print('afpt_ind_list')
#print(afpt_ind_list)
#mask = (z[:, 0] == 6)
#z[mask, :]
# import numpy as np
# np_array = np.array([[0,4],[0,5],[3,5],[6,8],[9,1],[6,1]])
# rows=np.where(np_array[:,0]==6)
# print(np_array[rows])
#print('central trans only')
#masky = afpt_ind_list[0][:,3]==0
#print(afpt_ind_list[0][masky,:])

#marker_style = dict(color='tab:blue', 
#                    linestyle='none', 
#                    marker='o',
#                    #markersize=15, 
#                    #markerfacecoloralt='tab:red'
#                    )

for n in range(len(afpt_ind_list)):
    sc = ax.scatter(afpt_ind_list[n][:,0],
               afpt_ind_list[n][:,1], 
               c=afpt_ind_list[n][:,3],
               alpha=afpt_ind_list[n][:,2]/afpt_ind_list[n][:,2].max(),
               #cmap=''
               label=afpt_ind_name_list[n],
               edgecolors='none',
               #**marker_style
               )
    cbar = fig.colorbar(sc)
    cbar.set_label('Parent Transition Index')

# export all as one:
if save_files_bool:
    if sim_export_file[-4:]=='.txt':
        sim_export_filename_str=sim_export_file
    else:
        sim_export_filename_str = sim_export_file+'.txt'

    for n in range(len(afpt_ind_list)):
        isotope_number = np.full(shape=angle_array.shape, fill_value=n)
        afpt_ind_list[n]=np.column_stack((afpt_ind_list[n],isotope_number))
    output_array = np.vstack(tuple(afpt_ind_list))
    np.savetxt(sim_export_filename_str, output_array)

# plot simulation as red lines

# label the axes
if angle_to_vary=='phi_z_deg':
    ax.set_xlabel(r'$\phi_z$ ($\degree$)')
elif angle_to_vary=='theta_x_prime_deg':
    ax.set_xlabel(r'$\theta_{x^\prime}$ ($\degree$)')
elif angle_to_vary=='psi_z_prime_deg':
    ax.set_xlabel(r'$\psi_{z^\prime}$ ($\degree$)')
ax.set_ylabel('Frequency (MHz)')

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

Hinta_str = r'$H_{int}^{a}$ = ' + str(Hinta_list) + r' T'
Hintb_str = r'$H_{int}^{b}$ = ' + str(Hintb_list) + r' T'
Hintc_str = r'$H_{int}^{c}$ = ' + str(Hintc_list) + r' T'

phi_z_deg_init_str = r'$\phi_{z,\mathrm{init}}$ = ' + str(phi_z_deg_init_list) + r' $\degree$' 
theta_x_prime_deg_init_str = r'$\theta_{x^\prime,\mathrm{init}}$ = ' + str(theta_x_prime_deg_init_list) + r' $\degree$'
psi_z_prime_deg_init_str = r'$\psi_{z^\prime,\mathrm{init}}$ = ' + str(psi_z_prime_deg_init_list) + r' $\degree$'

title_str = 'Angular-dependent spectrum; ' + H0_str
ax.set_title(title_str)

legend_title_str = str(isotope_str + '\n' 
+ K_a_str + '\n' 
+ K_b_str + '\n' 
+ K_c_str + '\n' 
+ v_c_str + '\n' 
+ eta_str + '\n' 
+ Hinta_str + '\n' 
+ Hintb_str + '\n' 
+ Hintc_str + '\n' 
+ phi_z_deg_init_str + '\n' 
+ theta_x_prime_deg_init_str + '\n' 
+ psi_z_prime_deg_init_str)

h,l = ax.get_legend_handles_labels()
leg = lax.legend(h,l,borderaxespad=0.)
leg.set_title(title=legend_title_str, prop = {'size':'small'})

lax.axis("off")

ax.tick_params(direction='in',bottom=True, top=True, left=True, right=True)

plt.tight_layout()
plt.show()

