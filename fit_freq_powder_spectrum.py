
###############################################################################
### PARAMETERS ################################################################
##----experimental data file---------------------------------------------------
# if you want to plot data also, enter the path to the file here, otherwise write datafile=''; first column is interpreted as frequency (MHz), second as intensity
exp_data_file = 'dummy_freq_pp.txt'
number_of_header_lines = 0    # number of lines which are ignored in the begining of the data file
exp_data_delimiter = ' '      # tell numpy which delimter your experimental data file has 
missing_values_string = 'nan'
exp_x_scaling = 1             # to scale the experimental data's frequency axis to MHz if units don't match

##----NMR Parameters/Least squares fitting control-----------------------------
minimization_algorithm = 'leastsq' # leastsq (default), dual_annealing, cg, nelder, basinhopping # https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize
epsilon = None # 0.001 # step size for calculating derivatives in the curve fitting, set to None for default value
verbose_bool = False # set to True to show the change of variables per iteration of the minimizer
H0 = [7.0]              # magnetic field  (units = T)
isotope_list = ['75As']

# vary and constrain parameters
# format is list of sublists of the form:
# [constant_value(float)]: parameter will be held constant (vary=False)
# [initial_value(float), vary(bool)]: parameter will be allowed to vary in the fitting procedure
# [initial_value(float), vary(bool), minimum(float), maximum(float)]: same as above with constraints
# [expression_based_on_par_names(string)]: parameter will be constrained to be identical to a different parameter
# the naming scheme for holding parameters equal to others follows the convention: 
# isotope_name + '_' + site_number(starting with zero) + '_' expression_based_on_par_names 
# expression_based_on_par_names can include a variety of operators, functions, and constants: https://lmfit.github.io/lmfit-py/constraints.html
# example: Ka_list = [[0.5, True, 0, 1.5], [1.0], ['Ka_75As_0']]

amplitude_list = [[0.01, True]]        # scales individual relative intensities of the summed spectra
Ka_list = [[0]]           # shift tensor elements (units = percent)
Kb_list = [[0]]
Kc_list = [[0.01, True]]
va_list = [[None]]  # only functions with exact diag; two modes: va and vb=None and eta=number OR 
vb_list = [[None]]  # only functions with exact diag; va and vb=numbers and eta=None (be sure to satisfy va+vb+vc=0)
vc_list = [[0.25, True]]     # units = MHz (note, in this simulation software princ axes of efg and shift tensors are fixed to be coincident.
eta_list = [[0]]  # asymmetry parameter (unitless)

Hinta_list = [[0]]  # internal field in a direction (units = T)
Hintb_list = [[0]]  # ... b
Hintc_list = [[0]]  # ... c these are only taken into account in exact diag

#----exact-diagonalization-specific inputs------------------------------------------------
mtx_elem_min = 0.1                # minimum allowed value for the probability of the transition (arbitrary units). Increase to remove forbidden transitions.

##----Simulation control-------------------------------------------------------
sim_type = 'exact diag'        # options are 'exact diag' and '2nd order'
min_freq = 50.5                           # units = MHz
max_freq = 52                             # units = MHz
n_plot_points = 1000                    # number of points in the plotted guess and fit
line_shape_func_list = ['gauss']   # 'gauss' (Gaussian) and 'lor' (Lorentzian) line shapes are implemented
FWHM_list = [[0.015, True, 0, 1]]       #  (units = MHz)
FWHM_vQ_list = [[0.015, True, 0, 1]]    # additional FWHM applied to scaled by transition number for 
                                # line shape caused by distribution of EFG values (units = MHz)
                                # total line shape satellite transitions will be (FWHM_list[i] + n*FWHM_vQ_list[i])
                                # where n is the transition index (1st satellite has n=1, 2nd has n=2, etc.)
recalc_random_samples = False     # if True, calculate fresh random angle sampling; if False, use the samples from saved (only relevant for exact diag)
                                # note that it would be best practice to generate this the first time, and then perform all subsequent fits using the same
                                # random samples. if we change it for each fit, the best fit params may change slightly. the same samples will be used for each iteration
                                # if this variable is true, and then the rotation matrices will be saved to files r.npy, ri.npy, SR.npy, and SRi.npy. These can then be used again in the next fits.
n_samples = 1e5                 # good exact diag powder spectra at roughly 1e5 (for very fast calcs). 2nd ord pert much faster so 1e7 possible
                                # NOTE: the saved binary files are actually the stacked arrays of rotation matrices and can get quite large (hundreds of MB) and 
                                # memory issues can arrise here...

##----Background control-------------------------------------------------------
#[[offset]] = constant background
#[[offset], [slope]] = linear background
#[[center], [width], [intensity]] = gaussian background
# so for offset initial value of 5.0 allowed to vary between 0 and 10: background_list = [[5.0, True, 0, 10]]
background_list = [[0.1]]
#background_list = [[0.01, True, 0, 1], [2, True, 1, 3]]
#background_list = [[0.01, True, 0, 1], [2.5, True, 0.1, 10], [5, True, 3, 6]]

##----Convolution control-------------------------------------------------------
#convolve the spectrum with a user-defined function (initially for WURST, but can consider sinc, gaussian, etc.)
#def convolution_fn:
#    asdf

##----Plot control-------------------------------------------------------------
plot_initial_guess_bool = True
plot_individual_bool = False
plot_sum_bool = True                     # plot the summation of the individual spectra
plot_legend_width_ratio = [3.25, 1]
x_axis_min = min_freq
x_axis_max = max_freq
y_axis_min = None
y_axis_max = None

##----Exporting Simulated Spectrum---------------------------------------------
# if you want to export your simulation, enter the path to the file here
# otherwise write exportfile = ''
sim_export_file = 'dummy_freq_pp_best_fit.txt'

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#do not edit code beyond this point
























import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pySimNMR
import lmfit


class InputMetadata:
    minimization_algorithm = None
    verbose_bool = True

    isotope_list = None

    H0 = None
    amplitude_list = None
    Ka_list = None
    Kb_list = None
    Kc_list = None
    va_list = None
    vb_list = None
    vc_list = None
    eta_list = None
    Hinta_list = None
    Hintb_list = None
    Hintc_list = None
    phi_z_deg_list = None
    theta_xp_deg_list = None
    psi_zp_deg_list = None
    FWHM_list = None
    FWHM_vQ_list = None
    background_list = None

    sim_export_file = None
    line_shape_func_list = None
    min_freq = None
    max_freq = None
    n_plot_points = None
    plot_initial_guess_bool = True
    plot_individual_bool = None
    plot_sum_bool = None
    plot_legend_width_ratio = None
    x_axis_min = None
    x_axis_max = None
    y_axis_min = None
    y_axis_max = None



class OutputMetadata:
    gamma_list = []
    spec_ind_list = []
    spec_ind_name_list = []
    xdata_init_bestfit = None


# function to create output_par_dict from the input parameters above
def populate_par_dict(output_par_dict,
                      H0,
                      isotope_list,
                      amplitude_list,
                      Ka_list,
                      Kb_list,
                      Kc_list,
                      va_list,
                      vb_list,
                      vc_list,
                      eta_list,
                      Hinta_list,
                      Hintb_list,
                      Hintc_list,
                      phi_z_deg_list,
                      theta_xp_deg_list,
                      psi_zp_deg_list,
                      FWHM_list,
                      FWHM_vQ_list,
                      background_list):
    """
    expects an output_par_dict to be previously defined as: output_par_dict = Parameters(), after
    from lmfit import Parameters
    parameter lists are expected to contain only sublists of the following formats:
    # [constant_value(float)]: parameter will be held constant (vary=False)
    # [initial_value(float), vary(bool)]: parameter will be allowed to vary in the fitting procedure
    # [initial_value(float), vary(bool), minimum(float), maximum(float)]: same as above with constraints
    # [expression_based_on_par_names(string)]
    """
    # parse the H0 list, which will be the same for all sites
    if len(H0) == 1:
        output_par_dict.add('H0', value=H0[0], vary=False)
    elif len(H0) == 2:
        output_par_dict.add('H0', value=H0[0], vary=H0[1])
    elif len(par_values) == 4:
        output_par_dict.add('H0',
                            value=H0[0],
                            vary=H0[1],
                            min=H0[2],
                            max=H0[3])
    else:
        print('H0 list has an unexpected number of elements: len(H0) = ' + str(len(H0)))

    # parse the background list
    if len(background_list) == 1:
        #[[offset]] = constant background
        if len(background_list[0]) == 1:
            #constant offset
            output_par_dict.add('offset', value=background_list[0][0], vary=False)
        elif len(background_list[0]) == 2:
            #variable offset background (or allow for mistake with vary=False)
            output_par_dict.add('offset',
                                value=background_list[0][0],
                                vary=background_list[0][1])
        elif len(background_list[0]) == 4:
            #variable offset constrained
            output_par_dict.add('offset',
                                value=background_list[0][0],
                                vary=True,
                                min=background_list[0][2],
                                max=background_list[0][3])
        else:
            print('offset background_list has an unexpected number of entries: len(background_list[0]) = ' + str(len(background_list[0])))
    elif len(background_list) == 2:
        #[[offset], [slope]] = linear background
        if len(background_list[0]) == 1:
            #constant offset
            output_par_dict.add('offset', value=background_list[0][0], vary=False)
        elif len(background_list[0]) == 2:
            #variable offset background (or allow for mistake with vary=False)
            output_par_dict.add('offset',
                                value=background_list[0][0],
                                vary=background_list[0][1])
        elif len(background_list[0]) == 4:
            #variable offset constrained
            output_par_dict.add('offset',
                                value=background_list[0][0],
                                vary=True,
                                min=background_list[0][2],
                                max=background_list[0][3])
        else:
            print('linear offset background_list has an unexpected number of entries: len(background_list[0]) = ' + str(len(background_list[0])))
        if len(background_list[1]) == 1:
            #constant offset
            output_par_dict.add('slope', value=background_list[1][0], vary=False)
        elif len(background_list[1]) == 2:
            #variable offset background (or allow for mistake with vary=False)
            output_par_dict.add('slope',
                                value=background_list[1][0],
                                vary=background_list[1][1])
        elif len(background_list[1]) == 4:
            #variable offset constrained
            output_par_dict.add('slope',
                                value=background_list[1][0],
                                vary=True,
                                min=background_list[1][2],
                                max=background_list[1][3])
        else:
            print('linear slope background_list has an unexpected number of entries: len(background_list[1]) = ' + str(len(background_list[1])))
    elif len(background_list) == 3:
        #[[center], [width], [intensity]] = gaussian background
        if len(background_list[0]) == 1:
            #constant center
            output_par_dict.add('center', value=background_list[0][0], vary=False)
        elif len(background_list[0]) == 2:
            #variable center (or allow for mistake with vary=False)
            output_par_dict.add('center',
                                value=background_list[0][0],
                                vary=background_list[0][1])
        elif len(background_list[0]) == 4:
            #variable center constrained
            output_par_dict.add('center',
                                value=background_list[0][0],
                                vary=True,
                                min=background_list[0][2],
                                max=background_list[0][3])
        else:
            print('gaussian center background_list has an unexpected number of entries: len(background_list[0]) = ' + str(len(background_list[0])))
        if len(background_list[1]) == 1:
            #constant width
            output_par_dict.add('width', value=background_list[1][0], vary=False)
        elif len(background_list[1]) == 2:
            #variable width (or allow for mistake with vary=False)
            output_par_dict.add('width',
                                value=background_list[1][0],
                                vary=background_list[1][1])
        elif len(background_list[1]) == 4:
            #variable width constrained
            output_par_dict.add('width',
                                value=background_list[1][0],
                                vary=True,
                                min=background_list[1][2],
                                max=background_list[1][3])
        else:
            print('gaussian width background_list has an unexpected number of entries: len(background_list[1]) = ' + str(len(background_list[1])))
        if len(background_list[2]) == 1:
            #constant intensity
            output_par_dict.add('intensity', value=background_list[2][0], vary=False)
        elif len(background_list[2]) == 2:
            #variable intensity (or allow for mistake with vary=False)
            output_par_dict.add('intensity',
                                value=background_list[2][0],
                                vary=background_list[2][1])
        elif len(background_list[2]) == 4:
            #variable intensity constrained
            output_par_dict.add('intensity',
                                value=background_list[2][0],
                                vary=True,
                                min=background_list[2][2],
                                max=background_list[2][3])
        else:
            print('gaussian intensity background_list has an unexpected number of entries: len(background_list[2]) = ' + str(len(background_list[2])))
    else:
        print('background_list has an unexpected number of entries: len(background_list) = ' + str(len(background_list)))
    
    all_pars_dict = {'amplitude': amplitude_list,
                     'Ka': Ka_list,
                     'Kb': Kb_list,
                     'Kc': Kc_list,
                     'va': va_list,
                     'vb': vb_list,
                     'vc': vc_list,
                     'eta': eta_list,
                     'Hinta': Hinta_list,
                     'Hintb': Hintb_list,
                     'Hintc': Hintc_list,
                     'phi_z_deg': phi_z_deg_list,
                     'theta_xp_deg': theta_xp_deg_list,
                     'psi_zp_deg': psi_zp_deg_list,
                     'FWHM': FWHM_list,
                     'FWHM_vQ': FWHM_vQ_list}
    
    # create dictionary to hold the expressions, which need to be added to the Parameters() dictionary 
    # after the pars with names that need to be existing python objects
    expression_dict = {}
    for i in range(len(isotope_list)):
        for general_par_name in all_pars_dict:
            specific_par_name = general_par_name + '_' + isotope_list[i] + '_{}'.format(i)
            general_par_values = all_pars_dict[general_par_name]
            if len(general_par_values[i]) == 1:
                if type(general_par_values[i][0]) is str:
                    # length 1 list is either a string expression, in which case
                    # we need to add the other non-expression based 
                    expression_dict[specific_par_name] = general_par_values[i][0]
                else:
                    # or a value to be held constant:
                    output_par_dict.add(specific_par_name, 
                                        value=general_par_values[i][0], 
                                        vary=False)
            elif len(general_par_values[i]) == 2:
                # length 2 list is a parameter to be varied (or allow for a mistake 
                # and the second list element can be False)
                output_par_dict.add(specific_par_name, 
                                    value=general_par_values[i][0], 
                                    vary=general_par_values[i][1])
            elif len(general_par_values[i]) == 4:
                # 4 element list will be a variable parameter with min and max constraints
                # note that we are forcing both min and max instead of one or another
                output_par_dict.add(specific_par_name, 
                                    value=general_par_values[i][0], 
                                    vary=general_par_values[i][1],
                                    min=general_par_values[i][2],
                                    max=general_par_values[i][3])
    # now process and add expressions to the list. this must happen as there is a limitation
    # within lmfit that requires the expressions to contain only existing python objects
    for expression_par_name, expression in expression_dict.items():
        output_par_dict.add(expression_par_name, expr=expression)



def model_function(par_dict, 
                   input_metadata, 
                   output_metadata,
                   x):
    """

    par_dict = lmfit.parameter.Parameters instance. specifically expected to
        the output of the populate_par_dict function
    input_metadata = a class instance of InputMetadata containing necessary 
        parameters that are not to be varied, but are required for the fitting
        function to work
    output_metadata = a class instance of OutputMetadata which contains (iunitially)
        empty lists needed for plotting/building the legend
    x = is the independent axis (frequency in this case) values at which 
        to calculate the spectrum. Can be the experimental data for fitting
        or dummy values for making a theoretical spectrum.
    """
    # extract the necessary parameters from the input_metadata class instance
    isotope_list = input_metadata.isotope_list
    min_freq = input_metadata.min_freq
    max_freq = input_metadata.max_freq
    n_plot_points = input_metadata.n_plot_points
    line_shape_func_list = input_metadata.line_shape_func_list
    # same for output_metadata class instance
    gamma_list = output_metadata.gamma_list
    spec_ind_list = output_metadata.spec_ind_list
    spec_ind_name_list = output_metadata.spec_ind_name_list
    # although we want these outputs, we also want them to be zeroed out
    # after each function call.
    gamma_list = []
    spec_ind_list = []
    spec_ind_name_list = []
    # loop over the isotope list to generate the individual spectra
    i = 0
    for isotope in isotope_list:
        # instantiate the simulation class
        sim = pySimNMR.SimNMR(isotope)
        # build the list of rounded gyromagnetic ratios
        gamma_list.append(sim.isotope_data_dict[isotope]["gamma_sigfigs"])
        
        # define the dictionary keys that we need to access the required parameters
        amplitude_key = 'amplitude' + '_' + isotope + '_{}'.format(i)
        Ka_key = 'Ka_' + isotope + '_{}'.format(i)
        Kb_key = 'Kb_' + isotope + '_{}'.format(i)
        Kc_key = 'Kc_' + isotope + '_{}'.format(i)
        va_key = 'va_' + isotope + '_{}'.format(i)
        vb_key = 'vb_' + isotope + '_{}'.format(i)
        vc_key = 'vc_' + isotope + '_{}'.format(i)
        eta_key = 'eta_' + isotope + '_{}'.format(i)
        Hinta_key = 'Hinta_' + isotope + '_{}'.format(i)
        Hintb_key = 'Hintb_' + isotope + '_{}'.format(i)
        Hintc_key = 'Hintc_' + isotope + '_{}'.format(i)
        phi_z_deg_key = 'phi_z_deg_' + isotope + '_{}'.format(i)
        theta_xp_deg_key = 'theta_xp_deg_' + isotope + '_{}'.format(i)
        psi_zp_deg_key = 'psi_zp_deg_' + isotope + '_{}'.format(i)
        FWHM_key = 'FWHM_' + isotope + '_{}'.format(i)
        FWHM_vQ_key = 'FWHM_vQ_' + isotope + '_{}'.format(i)
        
        # use the built in SimNMR methods to generate the rotation matrices
        # lower case r matrices are for rotation of the shift tensor
        #  SR matrices are for spin-space rotation of the quadrupole Hamiltonian
        phi_z = np.array([par_dict[phi_z_deg_key]])*np.pi/180
        theta_xp = np.array([par_dict[theta_xp_deg_key]])*np.pi/180
        psi_zp = np.array([par_dict[psi_zp_deg_key]])*np.pi/180

        r, ri = sim.generate_r_matrices(phi_z,
                                        theta_xp,
                                        psi_zp)
        SR, SRi = sim.generate_SR_matrices(phi_z,
                                           theta_xp,
                                           psi_zp)
        rotation_matrices = (r, ri, SR, SRi)
        
        spec = sim.freq_spec_ed(x=x,
                                H0 = par_dict['H0'],
                                Ka=par_dict[Ka_key], 
                                Kb=par_dict[Kb_key], 
                                Kc=par_dict[Kc_key], 
                                va=par_dict[va_key],
                                vb=par_dict[vb_key],
                                vc=par_dict[vc_key],
                                eta=par_dict[eta_key],
                                rm_SRm_tuple=rotation_matrices,
                                Hinta=par_dict[Hinta_key],
                                Hintb=par_dict[Hintb_key],
                                Hintc=par_dict[Hintc_key],
                                mtx_elem_min=mtx_elem_min,
                                min_freq=min_freq, 
                                max_freq=max_freq,
                                FWHM=par_dict[FWHM_key],
                                FWHM_vQ=par_dict[FWHM_vQ_key],
                                line_shape_func=line_shape_func_list[i])
        
        spec = spec*par_dict[amplitude_key]
        spec_ind_list.append(spec)
        spec_ind_name_list.append(isotope + '_{}'.format(i))
        i = i + 1

    # sum the individual spectra
    spec_sum = sum(spec_ind_list)

    # background correction
    # constant background
    if ('offset' in par_dict.keys()) and ('slope' not in par_dict.keys()):
        sim_y = spec_sum + par_dict['offset']
    # linear background
    if ('offset' in par_dict.keys()) and ('slope' in par_dict.keys()):
        sim_y = spec_sum + par_dict['offset'] + x*par_dict['slope']
    # gaussian background
    if ('center' in par_dict.keys()) and ('width' in par_dict.keys()) and ('intensity' in par_dict.keys()):
        gaussian = (par_dict['intensity']/(np.sqrt(2*np.pi)*par_dict['width']))*np.exp(-(x - par_dict['center'])**2/(2*par_dict['width']**2))
        sim_y = spec_sum + gaussian

    return sim_y


def residuals(par_dict, 
              input_metadata, 
              output_metadata, 
              x, 
              y_data=None, 
              y_data_err=None):
    model = model_function(par_dict, input_metadata, output_metadata, x)
    if y_data is None:
        return model
    if y_data_err is None:
        return model - y_data    
    return (model - y_data)/y_data_err


def per_iteration(par_dict, iter, residuals, *args, **kws):    
    list_to_print = []
    if iter == -1:
        for key, parameter in par_dict.items():
            list_to_print.append(key)
        print('parameter names: ', list_to_print)
    else:    
        for key, parameter in par_dict.items():
            list_to_print.append(round(parameter.value, 5))
        print("function evals =", iter, list_to_print)

# or as follows with pretty printing
#def per_iteration(par_dict, iter, residuals, *args, **kws):
#    par_dict.pretty_print()


def fit_and_plot(input_metadata,
                 exp_data_x,
                 exp_data_y,
                 exp_data_y_err):
    """
    see https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize for minimization_algorithm options.
    """
    #extract the input parameters and various additional variables
    minimization_algorithm = input_metadata.minimization_algorithm
    if minimization_algorithm == None:
        minimization_algorithm = 'leastsq'
    epsilon = input_metadata.epsilon
    
    verbose_bool = input_metadata.verbose_bool
    
    isotope_list = input_metadata.isotope_list
    
    H0 = input_metadata.H0
    amplitude_list = input_metadata.amplitude_list
    Ka_list = input_metadata.Ka_list
    Kb_list = input_metadata.Kb_list
    Kc_list = input_metadata.Kc_list
    va_list = input_metadata.va_list
    vb_list = input_metadata.vb_list
    vc_list = input_metadata.vc_list
    eta_list = input_metadata.eta_list
    Hinta_list = input_metadata.Hinta_list
    Hintb_list = input_metadata.Hintb_list
    Hintc_list = input_metadata.Hintc_list
    phi_z_deg_list = input_metadata.phi_z_deg_list
    theta_xp_deg_list = input_metadata.theta_xp_deg_list
    psi_zp_deg_list = input_metadata.psi_zp_deg_list
    FWHM_list = input_metadata.FWHM_list
    FWHM_vQ_list = input_metadata.FWHM_vQ_list
    background_list = input_metadata.background_list
    
    n_plot_points = input_metadata.n_plot_points
    min_freq = input_metadata.min_freq
    max_freq = input_metadata.max_freq
    
    sim_export_file = input_metadata.sim_export_file
    plot_initial_guess_bool = input_metadata.plot_initial_guess_bool
    plot_individual_bool = input_metadata.plot_individual_bool
    plot_sum_bool = input_metadata.plot_sum_bool
    plot_legend_width_ratio = input_metadata.plot_legend_width_ratio
    x_axis_min = input_metadata.x_axis_min
    x_axis_max = input_metadata.x_axis_max
    y_axis_min = input_metadata.y_axis_min
    y_axis_max = input_metadata.y_axis_max
    
    # create an empty parameter dictionary instance and populate it
    par_dict = lmfit.Parameters()
    populate_par_dict(par_dict,
                      H0,
                      isotope_list,
                      amplitude_list,
                      Ka_list,
                      Kb_list,
                      Kc_list,
                      va_list,
                      vb_list,
                      vc_list,
                      eta_list,
                      Hinta_list,
                      Hintb_list,
                      Hintc_list,
                      phi_z_deg_list,
                      theta_xp_deg_list,
                      psi_zp_deg_list,
                      FWHM_list,
                      FWHM_vQ_list,
                      background_list)
    
    # instantiate the plot metadata classes. the first two are just placeholders,
    # could add a switch instead, but this is fine for now/testing
    output_metadata_init = OutputMetadata()
    output_metadata_fitting = OutputMetadata()
    output_metadata_best_fit = OutputMetadata()
    
    # if we want to constrain the fit to some freq range, 
    # then modify exp_data_x, exp_data_y, and exp_data_y_err
    mask = np.logical_and(exp_data_x > min_freq, exp_data_x < max_freq)
    exp_data_x_masked = exp_data_x[mask]
    exp_data_y_masked = exp_data_y[mask]
    if exp_data_y_err is not None:
        exp_data_y_err_masked = exp_data_y_err[mask]
    else:
        exp_data_y_err_masked = exp_data_y_err
    
    # setup the Minimizer object
    #class Minimizer(userfcn, params, fcn_args=None, fcn_kws=None, iter_cb=None, 
    #                scale_covar=True, nan_policy='raise', reduce_fcn=None, 
    #                calc_covar=True, max_nfev=None, **kws)
    # epsfcn only works with Levenberg--Marquardt (or possibly others, but throws 
    # an error with, at least, Nelder--Mead)
    if minimization_algorithm == 'leastsq':
        if verbose_bool:
            mini = lmfit.Minimizer(userfcn=residuals,
                                   params=par_dict,
                                   fcn_args=(input_metadata, output_metadata_fitting, exp_data_x_masked), 
                                   fcn_kws={'y_data':exp_data_y_masked, 'y_data_err':exp_data_y_err_masked},
                                   iter_cb=per_iteration,
                                   epsfcn=epsilon)
        else:
            mini = lmfit.Minimizer(userfcn=residuals,
                                   params=par_dict,
                                   fcn_args=(input_metadata, output_metadata_fitting, exp_data_x_masked), 
                                   fcn_kws={'y_data':exp_data_y_masked, 'y_data_err':exp_data_y_err_masked},
                                   epsfcn=epsilon)
    else:
        if verbose_bool:
            mini = lmfit.Minimizer(userfcn=residuals,
                                   params=par_dict,
                                   fcn_args=(input_metadata, output_metadata_fitting, exp_data_x_masked), 
                                   fcn_kws={'y_data':exp_data_y_masked, 'y_data_err':exp_data_y_err_masked},
                                   iter_cb=per_iteration)
        else:
            mini = lmfit.Minimizer(userfcn=residuals,
                                   params=par_dict,
                                   fcn_args=(input_metadata, output_metadata_fitting, exp_data_x_masked), 
                                   fcn_kws={'y_data':exp_data_y_masked, 'y_data_err':exp_data_y_err_masked})
    # i think we could not have output_metadata_fitting here but instead include a switch, 
    # but this would allow for some kind of iteration plotting, so lets just leave it for now
    # and check it out later during testing
    
    # perform the fit with options
    results = mini.minimize(method=minimization_algorithm)
    
    # print the fit report
    print(lmfit.fit_report(results))

    # if min_freq (max_freq) is less (greater) than the minimum (maximum) frequency of the 
    # experimental data, then define the simulation min and max values to not plot spectra
    # outside of the bounds of the data to fit
#     if min_freq < np.nanmin(exp_data_x_masked):
#         sim_x_min = np.nanmin(exp_data_x_masked)
#     if max_freq > np.nanmax(exp_data_x_masked):
#         sim_x_max = np.nanmax(exp_data_x_masked)
#     sim_x = np.linspace(sim_x_min, sim_x_max, n_plot_points)
    # instead of the obove block, lets retain full control over the simulation frequency 
    # range (could help to understand where other peaks show up, that were not observed in 
    # the experiment)
    sim_x = np.linspace(min_freq, max_freq, n_plot_points)
    
    # produce the initial guess spectrum for plotting
    init_guess_y = residuals(par_dict, input_metadata, output_metadata_init, sim_x)
        
    # produce the best fit for plotting
    best_fit_y = residuals(results.params, input_metadata, output_metadata_best_fit, sim_x)
    
    # could also include an interpolated output perhaps... but leave for now
    output_guess_best_fit = np.column_stack((sim_x, init_guess_y, best_fit_y))
    # add the output data to the output_metadata_best_fit class to be returned
    output_metadata_best_fit.xdata_init_bestfit = output_guess_best_fit
    
    # setting the boolean variable for saving the simulation output or not
    if sim_export_file=='':
        save_files_bool=False
    else:
        save_files_bool=True
    # save the output and fit report
    if save_files_bool:
        if sim_export_file[-4:] == '.txt':
            sim_export_filename_str = sim_export_file
        else:
            sim_export_filename_str = sim_export_file + '.txt'
        np.savetxt(sim_export_filename_str, 
                   output_guess_best_fit, 
                   header='frequency inital_guess best_fit')
        # save the fit report
        with open(sim_export_file + '_fit_report.txt', 'a') as penske_file:
            penske_file.write(lmfit.fit_report(results))
    
    # prepare for plotting
    plt.rcParams['figure.figsize'] = 10, 4.5 # width,height in inches
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={'width_ratios':plot_legend_width_ratio})

    # plot experimental data first as black lines
    ax.plot(exp_data_x, exp_data_y, 'k-')

    # plot and save individual spectra if there are more than one
    spec_ind_list = output_metadata_best_fit.spec_ind_list
    if len(spec_ind_list) > 1:
        for n in range(len(spec_ind_list)):
            if plot_individual_bool:
                ax.fill(spec_ind_list[n][:, 0], 
                        spec_ind_list[n][:, 1], 
                        label=spec_ind_name_list[n],
                        linewidth=2,
                        alpha=0.4)
            if save_files_bool:
                ind_spec_out = np.column_stack((spec_ind_list[n][:, 0], spec_ind_list[n][:, 1]))
                ind_filename_out = sim_export_filename_str[:-4] + '_' + isotope_list[n] + '_{}.txt'.format(n)
                np.savetxt(ind_filename_out, ind_spec_out)

    if plot_initial_guess_bool:
        ax.plot(sim_x,
                init_guess_y,
                "b-",
                label='init. guess',
                linewidth=2,
                alpha=0.7)
    
    # plot simulation as red lines
    if plot_sum_bool:
        ax.plot(sim_x,
                best_fit_y,
                "r-",
                label='best fit',
                linewidth=2,
                alpha=0.7)

    # label the axes
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Intensity (arb. units)')

    #set axes range
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)

    # generate strings for the title and legend/details
    H0_str = r'$H_0$ = ' + str(H0) + r'T'

    # ahoy! need to extract the best fit parameters for plotting. 
    # for now leave inputs and rely on the printed/saved fit report
    gamma_list = output_metadata_best_fit.gamma_list
    isotope_str = 'isotopes = ' + str(isotope_list)
    gamma_str = r'$\gamma = $ ' + str(gamma_list) + r' MHz/T'
    K_a_str = r'$K_a$ = ' + str(Ka_list) + r' %'
    K_b_str = r'$K_b$ = ' + str(Kb_list) + r' %'
    K_c_str = r'$K_c$ = ' + str(Kc_list) + r' %'
    v_c_str = r'$\nu_c$ = ' + str(vc_list) + r' MHz'
    eta_str = r'$\eta$ = ' + str(eta_list)
    convfunc_str = 'conv.func. = ' + str(line_shape_func_list)
    FWHM_str = r'FWHM = ' + str(FWHM_list) + r' MHz'
    FWHM_vQ_str = r'FWHM$_{\nu_Q}$ = ' + str(FWHM_vQ_list) + r' MHz'
    Hinta_str = r'$H_{int}^{a}$ = ' + str(Hinta_list) + r' T'
    Hintb_str = r'$H_{int}^{b}$ = ' + str(Hintb_list) + r' T'
    Hintc_str = r'$H_{int}^{c}$ = ' + str(Hintc_list) + r' T'
    phi_z_deg_str = r'$\phi_z$ = ' + str(phi_z_deg_list) + r' $\degree$' 
    theta_x_prime_deg_str = r'$\theta_{x^\prime}$ = ' + str(theta_xp_deg_list) + r' $\degree$'
    psi_z_prime_deg_str = r'$\psi_{z^\prime}$ = ' + str(psi_zp_deg_list) + r' $\degree$'

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
                           + FWHM_str + '\n'
                           + FWHM_vQ_str + '\n'
                           + Hinta_str + '\n'
                           + Hintb_str + '\n'
                           + Hintc_str + '\n'
                           + phi_z_deg_str + '\n' 
                           + theta_x_prime_deg_str + '\n' 
                           + psi_z_prime_deg_str)

    h, l = ax.get_legend_handles_labels()
    leg = lax.legend(h, l, borderaxespad=0.0)
    leg.set_title(title=legend_title_str, prop = {'size':'small'})

    lax.axis("off")

    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)

    plt.tight_layout()
    plt.show()
    
    return (output_metadata_best_fit)



##############################################################################
##############################################################################
##############################################################################
##############################################################################
# instantiate the input metadata class and 

input_metadata = InputMetadata()

input_metadata.minimization_algorithm = minimization_algorithm
input_metadata.epsilon = epsilon
input_metadata.verbose_bool = verbose_bool

input_metadata.isotope_list = isotope_list

input_metadata.H0 = H0
input_metadata.amplitude_list = amplitude_list
input_metadata.Ka_list = Ka_list
input_metadata.Kb_list = Kb_list
input_metadata.Kc_list = Kc_list
input_metadata.va_list = va_list
input_metadata.vb_list = vb_list
input_metadata.vc_list = vc_list
input_metadata.eta_list = eta_list
input_metadata.Hinta_list = Hinta_list
input_metadata.Hintb_list = Hintb_list
input_metadata.Hintc_list = Hintc_list
input_metadata.phi_z_deg_list = phi_z_deg_list
input_metadata.theta_xp_deg_list = theta_xp_deg_list
input_metadata.psi_zp_deg_list = psi_zp_deg_list
input_metadata.FWHM_list = FWHM_list
input_metadata.FWHM_vQ_list = FWHM_vQ_list
input_metadata.background_list = background_list

input_metadata.min_freq = min_freq
input_metadata.max_freq = max_freq
input_metadata.n_plot_points = n_plot_points
input_metadata.line_shape_func_list = line_shape_func_list
input_metadata.sim_export_file = sim_export_file
input_metadata.plot_initial_guess_bool = plot_initial_guess_bool
input_metadata.plot_individual_bool = plot_individual_bool
input_metadata.plot_sum_bool = plot_sum_bool
input_metadata.plot_legend_width_ratio = plot_legend_width_ratio
input_metadata.x_axis_min = x_axis_min
input_metadata.x_axis_max = x_axis_max
input_metadata.y_axis_min = y_axis_min
input_metadata.y_axis_max = y_axis_max


# load experimental data to fit
exp_data = np.genfromtxt(fname=exp_data_file, 
                         delimiter=exp_data_delimiter, 
                         skip_header=number_of_header_lines, 
                         missing_values=missing_values_string)
exp_x = exp_data[:, 0]*exp_x_scaling
exp_y = exp_data[:, 1]



fit_output = fit_and_plot(input_metadata,
                          exp_data_x=exp_x,
                          exp_data_y=exp_y,
                          exp_data_y_err=None)
