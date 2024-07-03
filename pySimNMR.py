
import numpy as np
pi = np.pi
import time
import matplotlib.pyplot as plt
import sys
from isotopeDict import isotope_data_dict


class SimNMR(object):
    def __init__(self, isotope):
        self.isotope = isotope

        #ahoy! incomplete!
        #data source: R. K. Harris, et al. "NMR Nomenclature. Nuclear Spin Properties and
        #Conventions for Chemical Shifts" doi: 10.1351/pac200173111795
        # I0 = nuclear spin
        # gamma = gyromagnetic ratio in units of MHz/T
        # Q = barns (1 barn = 1e-28 m^2)
        # abundance = natural isotopic abundance in percent
        # sensitivity = relative sensitivity to 1H in constant field strength, out of 1
        #access like H1_gamma = self.isotope_data_dict["1H"]["gamma"]
        self.isotope_data_dict = isotope_data_dict
        # 115In and 113In are not consistent with the Harris standards
        # define the quantum spin operators etc.
        try:
            I0 = self.isotope_data_dict[isotope]["I0"]
        except:
            print('Selected isotope not yet implemented; please add it to the dictionary to continue.')
            sys.exit()

        dim = int((I0+1/2)*2)

        m = np.array([I0-i for i in range(dim)])
        
        Iz = []
        for row_i in range(dim):
            row = []
            for column_j in range(dim):
                if column_j==row_i:
                    row.append(m[column_j])
                else:
                    row.append(0)
            Iz.append(row)
        Iz = np.array(Iz)

        Iplus = []
        for row_i in range(dim):
            row = []
            for column_j in range(dim):
                if column_j==row_i + 1:
                    row.append(np.sqrt(I0*(I0+1)-m[column_j]*(m[column_j]+1)))
                else:
                    row.append(0)
            Iplus.append(row)
        Iplus = np.array(Iplus)

        Iminus = []
        for row_i in range(dim):
            row = []
            for column_j in range(dim):
                if column_j==row_i - 1:
                    row.append(np.sqrt(I0*(I0+1)-m[column_j]*(m[column_j]-1)))
                else:
                    row.append(0)
            Iminus.append(row)
        Iminus = np.array(Iminus)

        Ix = (Iplus + Iminus)/2

        Iy = (Iplus - Iminus)/2j

        I2 = Ix@Ix + Iy@Iy + Iz@Iz
        
        Ivec = np.vstack((Ix,Iy,Iz))

        II3 = np.eye(3)

        self.I0 = I0
        self.dim = dim
        self.m = m
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.I2 = I2
        self.Iplus = Iplus
        self.Iminus = Iminus
        self.II3 = II3
        self.Ivec = Ivec


    def elevels_vs_field_ed(self, 
                            H0, 
                            Ka, 
                            Kb, 
                            Kc, 
                            va,
                            vb,
                            vc,
                            eta,
                            rm_SRm_tuple,
                            Hinta=0.0,
                            Hintb=0.0,
                            Hintc=0.0,
                            mtx_elem_min=0.1,
                            min_freq=0.1,
                            max_freq=500):
        """
            Trying to figure out how to index the states that I find...
        """
        I0 = self.I0
        dim = self.dim
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Iplus = self.Iplus
        Iminus = self.Iminus
        I2 = self.I2
        II3 = self.II3
        Ivec = self.Ivec

        if va == None and vb == None and eta != None:
            va,vb = self.vx_vy_from_vz_eta(vc,eta)

        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        r, ri, SR, SRi  = rm_SRm_tuple

        # Knight shift tensor and rotations
        KtensorInit = np.array([[Ka/100.0, 0, 0],
                                [0, Kb/100.0, 0],
                                [0, 0, Kc/100.0]])
        Ktensor = r@KtensorInit@ri

        ## Zeeman Hamiltonian
        H0_array=H0
        # the extra dimension at the end  of the next line is to make 
        # sure that the @ operator in the line below KdotH0 = eyePlusK@H0vec
        # works properly to perform matrix multiplication (dot product)
        # on the the stacked matrices:eyePlusK times the stacked 
        # vectors:H0vec, then we will have to reshape again later
        H0vec=np.zeros(shape=(H0_array.shape[0],3,1)) 
        H0vec[:,2,0]=H0_array # populate the z directions of the H0vec from the input array
        eyePlusK = (II3 + Ktensor)
        KdotH0 = eyePlusK@H0vec
        KdotH0 = np.reshape(KdotH0,(H0_array.shape[0],3))
        Hze = gamma*(KdotH0[:,0,np.newaxis,np.newaxis]*Ix + KdotH0[:,1,np.newaxis,np.newaxis]*Iy + KdotH0[:,2,np.newaxis,np.newaxis]*Iz)
        ## allow for an internal magnetic field due to hyperfine coupling to magnetic order
        # may want to modify this to actually be S@A@Ivec in the future 
        HintVecInit = np.array([Hinta,Hintb,Hintc])
        HintVec = r@HintVecInit
        HzeInt =  gamma*(HintVec[:,0,np.newaxis,np.newaxis]*Ix + HintVec[:,1,np.newaxis,np.newaxis]*Iy + HintVec[:,2,np.newaxis,np.newaxis]*Iz)
        ## Quadrupole Hamiltonian
        try:
            HqInit = (vc/6.0)*(3.0*Iz@Iz - I2 + ((va - vb)/vc)*(Ix@Ix - Iy@Iy))
            Hq = SR@HqInit@SRi
            ## total Hamiltonian
            Htot = Hze + Hq + HzeInt
        except ZeroDivisionError:
            print('HqInit zero division error')
            HqInit = 0.0
            Htot = Hze + HzeInt
        ## calculate eigenvales and eigenvectors, sorted with respect to each other
        ## U is a matrix of eigenvectors columnwise
        evals, U = np.linalg.eig(Htot)
        ## U and Ui are the matricies of the eigenvectors which will be used to diagonalize the Hamiltonian
        Ui = np.transpose(np.conj(U),(0,2,1))

        #HtotD = Ui@Htot@U # checked nicks claim that this should produce evals in the 
                          # ordered parent basis based on matrix position... and 
                          # found that it does give a digaonal matrix with the evals
                          # ordered in the same order as the evecs, so it seems that 
                          # the evals may already be ordered based on the parent states
        Uiconj=np.conj(Ui)
        state_mixing_coefs = np.abs(Ui*Uiconj) # these are just the transposed eigenvectors (Ui) multiplied element-wise by their complex conjugates
                                                # the transpose makes it so that the elements of the rows correspond to the normalized coefs of each eigen state
                                                # the ordering is such that a nonzero coef in the first position indicates character of the eigenstate |I0>, second 
                                                # position |I0-1> and soforth. So a resulting eigenstate can be decomposed into a linear superposition of 
                                                # parent eigenstates (eigenstates of Iz, which are just [1,0,0,0], etc) as folowing:
                                                # |psi> = a0|3/2> + a1|1/2> + a2|-1/2> + a3|-3/2>
        state_mixing_coefs = state_mixing_coefs.reshape(state_mixing_coefs.shape[0]*state_mixing_coefs.shape[1], state_mixing_coefs.shape[2])
        ## use U and Ui on Iplus to get the matrix elements
        ## extract the real part of the eigenvalues (the imaginary parts are already zero since we have a hermitian hamiltonian)
        real_evals = evals.real
        ## produce the field associated with a given eigenvalue
        H0_array_reshape = H0_array.reshape((H0_array.shape[0],1))
        zeros = np.zeros(real_evals.shape[1])
        field_broadcast=H0_array_reshape+zeros
        elevels_fields_out = np.column_stack((field_broadcast.flatten(),real_evals.flatten()))

        outtuple = (elevels_fields_out,state_mixing_coefs)

        return(outtuple)


    def hfp_evals_ed(self, 
                     H0, 
                     Ka, 
                     Kb, 
                     Kc, 
                     va,
                     vb,
                     vc,
                     eta,
                     rm_SRm_tuple,
                     Hinta=0.0,
                     Hintb=0.0,
                     Hintc=0.0,
                     mtx_elem_min=0.1,
                     min_freq=0.1,
                     max_freq=500):
        """
        example output for 75As, zero field, v_c = 7 MHz, eta = 0
           field, freq, prob, eval_1, eval_2, p_state_1, p_state_2, mix_coefs_1,  mix_coefs_2
        [[ 0.     7.    3.    3.5    -3.5     1.5        0.5        1. 0. 0. 0.   0. 1. 0. 0. ]
         [ 0.     7.    3.   -3.5     3.5    -0.5       -1.5        0. 0. 1. 0.   0. 0. 0. 1. ]]
         
         
        """
        I0 = self.I0
        dim = self.dim
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Iplus = self.Iplus
        Iminus = self.Iminus
        I2 = self.I2
        II3 = self.II3
        Ivec = self.Ivec
        if va==None and vb==None and eta!=None:
            va,vb = self.vx_vy_from_vz_eta(vc,eta)
        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        r, ri, SR, SRi  = rm_SRm_tuple
        # Knight shift tensor and rotations
        KtensorInit = np.array([[Ka/100.0, 0, 0],
                                [0, Kb/100.0, 0],
                                [0, 0, Kc/100.0]])
        Ktensor = r@KtensorInit@ri
        ## Zeeman Hamiltonian
        H0_array=H0
        # the extra dimension at the end of the next line is to make 
        # sure that the @ operator in the line below KdotH0 = eyePlusK@H0vec
        # works properly to perform matrix multiplication (dot product)
        # on the the stacked matrices:eyePlusK times the stacked 
        # vectors:H0vec, then we will have to reshape again later
        H0vec=np.zeros(shape=(H0_array.shape[0],3,1)) 
        H0vec[:,2,0]=H0_array # populate the z directions of the H0vec from the input array
        eyePlusK = (II3 + Ktensor)
        KdotH0 = eyePlusK@H0vec
        KdotH0 = np.reshape(KdotH0,(H0_array.shape[0],3))
        Hze = gamma*(KdotH0[:,0,np.newaxis,np.newaxis]*Ix + KdotH0[:,1,np.newaxis,np.newaxis]*Iy + KdotH0[:,2,np.newaxis,np.newaxis]*Iz)
        ## allow for an internal magnetic field due to hyperfine coupling to magnetic order
        # may want to modify this to actually be S@A@Ivec in the future 
        HintVecInit = np.array([Hinta,Hintb,Hintc])
        HintVec = r@HintVecInit
        HzeInt =  gamma*(HintVec[:,0,np.newaxis,np.newaxis]*Ix + HintVec[:,1,np.newaxis,np.newaxis]*Iy + HintVec[:,2,np.newaxis,np.newaxis]*Iz)
        ## Quadrupole Hamiltonian
        try:
            HqInit = (vc/6.0)*(3.0*Iz@Iz - I2 + ((va - vb)/vc)*(Ix@Ix - Iy@Iy))
            Hq = SR@HqInit@SRi
            ## total Hamiltonian
            Htot = Hze + Hq + HzeInt
        except ZeroDivisionError:
            print('HqInit zero division error')
            HqInit = 0.0
            Htot = Hze + HzeInt
        ## calculate eigenvales and eigenvectors, sorted with respect to each other
        ## U is a matrix of eigenvectors columnwise
        evals, U = np.linalg.eig(Htot)
        ## U and Ui are the matricies of the eigenvectors which will be used to diagonalize the Hamiltonian
        Ui = np.transpose(np.conj(U),(0,2,1))
        #HtotD = Ui@Htot@U # checked nicks claim that this should produce evals in the 
                          # ordered parent basis based on matrix position... and 
                          # found that it does give a digaonal matrix with the evals
                          # ordered in the same order as the evecs, so it seems that 
                          # the evals may already be ordered based on the parent states
        Uiconj=np.conj(Ui)
        # the next line are just the transposed eigenvectors (Ui) multiplied element-wise by their complex conjugates
        # the transpose makes it so that the elements of the rows correspond to the normalized coefs of each eigen state
        # the ordering is such that a nonzero coef in the first position indicates character of the eigenstate |I0>, second 
        # position |I0-1> and soforth. So a resulting eigenstate can be decomposed into a linear superposition of 
        # parent eigenstates (eigenstates of Iz, which are just [1,0,0,0], etc) as folowing:
        # |psi> = a0|3/2> + a1|1/2> + a2|-1/2> + a3|-3/2>
        state_mixing_coefs = np.abs(Ui*Uiconj) 
        # the following reshapes the state_mixing_coefs so that they are flattened by one dimension
        #state_mixing_coefs = state_mixing_coefs.reshape(state_mixing_coefs.shape[0]*state_mixing_coefs.shape[1],state_mixing_coefs.shape[2])
        ## use U and Ui on Ix to get the matrix elements for all possible transitions
        H1x = Ui@Ix@U
        trans_probs_array = np.abs(H1x)**2
        # There seems to have been an issue where, where, when eta is nonzero, the following transition probability 
        # intensities were not all captured by |<n_i|I+|n_j>|^2 for all i and j for eta != 0, instead the Ix = (Iplus + Iminus) / 2 operator 
        # should be used to calculate the transition intensities. This may be related to the mixing of eigenstates for eta != 0, and therefore
        # the spin lowering operator must be used to capture some of the allowed superposed-state transitions. This also seems to only matter for
        # I0 > 3/2. Tested on 115In(2) site in CeRhIn5

        allowed_trans_idxs = np.argwhere(trans_probs_array > mtx_elem_min)
        ## extract the real part of the eigenvalues (the imaginary parts are already zero since we have a hermitian hamiltonian)
        real_evals = evals.real
        # remember the naming scheme is high m = high energy and not the other way around
        allowed_eval_m_idxs = allowed_trans_idxs[:, [0, 1]]
        allowed_eval_mPlus1_idxs = allowed_trans_idxs[:, [0, 2]]
        allowed_eval_m = evals[allowed_eval_m_idxs[:, 0], allowed_eval_m_idxs[:, 1]]
        allowed_eval_mPlus1 = evals[allowed_eval_mPlus1_idxs[:, 0], allowed_eval_mPlus1_idxs[:, 1]]
        field = H0_array[allowed_eval_m_idxs[:, 0]]
        # there will likely exist duplicate frequencies and will need to sum the probabilities
        freq = np.abs(allowed_eval_mPlus1 - allowed_eval_m)
        prob = trans_probs_array[trans_probs_array > mtx_elem_min]
        # evals1 and 2
        eval_1 = real_evals[allowed_eval_m_idxs[:, 0], allowed_eval_m_idxs[:, 1]]
        eval_2 = real_evals[allowed_eval_mPlus1_idxs[:, 0], allowed_eval_mPlus1_idxs[:, 1]]
        # index the 'parent states', that is, the states in the zeeman basis in which Iz is diagonal
        # these states can be connected in the rotated basis and yet be not technically m<-->m+1 though
        # they should have some character of states that are separated by only one quantum number
        parent_state_names = np.linspace(I0, -I0, dim)
        p_state_1 = parent_state_names[allowed_eval_m_idxs[:, 1]]
        p_state_2 = parent_state_names[allowed_eval_mPlus1_idxs[:, 1]]
        mix_coefs_1 = state_mixing_coefs[allowed_eval_m_idxs[:, 0], allowed_eval_m_idxs[:, 1], :]
        # print('mix_coefs_1')
        # print(np.around(mix_coefs_1,decimals=3))
        mix_coefs_2 = state_mixing_coefs[allowed_eval_mPlus1_idxs[:, 0], allowed_eval_mPlus1_idxs[:, 1], :]
        # produce the field associated with a given eigenvalue
        #H0_array_reshape = H0_array.reshape((H0_array.shape[0],1))
        #zeros = np.zeros(real_evals.shape[1])
        #field_broadcast=H0_array_reshape+zeros
        #elevels_fields_out = np.column_stack((field_broadcast.flatten(),real_evals.flatten()))
        # the mix_coefs_1 and mix_coefs_2 will each have n_colmns = dim
        output = np.column_stack((field,
                                  freq,
                                  prob,
                                  eval_1,
                                  eval_2,
                                  p_state_1,
                                  p_state_2,
                                  mix_coefs_1,
                                  mix_coefs_2))
        # delete zero-frequency transitions
        zero_freq_idx = np.argwhere(output[:,1] < 1e-6)
        output = np.delete(output, zero_freq_idx, axis=0)
        # sum the probabilities
        return(output)


    def vx_vy_from_vz_eta(self,vz,eta):
        vx = 0.5*(eta - 1)*vz
        vy = -0.5*(eta + 1)*vz
        return (vx,vy)


    def generate_r_matrices(self,
                            phi,
                            theta,
                            psi):
        """
        expects three 1D numpy arrays of angles in radians
        all three input arrays must have the same length
        generates an array of 3x3 arrays of dimension (len(phi), 3, 3) 
        of the following form for the euler angle intrinsic rotations. 
        For example, if we had no phi or psi rotations, then we would 
        obtain only rotations by the elements of theta around the x axis:
        [[[1, 0,                 0                 ],
          [0, np.cos(theta[0]), -1*np.sin(theta[0])],
          [0, np.sin(theta[0]),  np.cos(theta[0])  ]],
         [[1, 0,                 0],
          [0, np.cos(theta[1]), -1*np.sin(theta[1])],
          [0, np.sin(theta[1]),  np.cos(theta[1])  ]],
         .
         .
         .
        ]
        returns a two element tuple where the first element is the array of
        rotation matrices and the second element is an array of transposed
        (inverse) arrays for applying rotations to tensors.
        """
        n_angles = len(phi)
        # create an empty array with the correct length
        r = np.empty(shape=(3*3*n_angles,))
        
        # set the individual matrix elements based on our calculation of the intrinsic
        # rotations by the three euler angles
        r[0::9] = -np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi)
        r[1::9] = ((1/2)*np.sin(phi - psi) - (1/2)*np.sin(phi + psi)
                   + (1/4)*np.sin(-phi + psi + theta) - 1/4*np.sin(phi - psi + theta)
                   - 1/4*np.sin(phi + psi - theta) - 1/4*np.sin(phi + psi + theta))
        r[2::9] = np.sin(phi)*np.sin(theta)
        r[3::9] = np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta)
        r[4::9] = -np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta)
        r[5::9] = -np.sin(theta)*np.cos(phi)
        r[6::9] = np.sin(psi)*np.sin(theta)
        r[7::9] = np.sin(theta)*np.cos(psi)
        r[8::9] = np.cos(theta)
        
        # reshape the array to (len(phi), 3, 3)  so we can deploy it later 
        r = np.reshape(r, (n_angles, 3, 3))
        ri = np.transpose(r, (0, 2, 1))

        return (r, ri)


    def generate_SR_matrices(self,
                             phi_1D,
                             theta_1D,
                             psi_1D):
        """
            expects three 1D numpy arrays of angles in radians
            all three input arrays must have the same length
            generates an array of dim x dim arrays of dimension (len(phi), dim, dim) 
            for the Euler angle intrinsic rotations in spin-space
            see https://mathworld.wolfram.com/EigenDecomposition.html and 
            https://en.wikipedia.org/wiki/Rotation_operator_(quantum_mechanics) 
            for further details
            """
        # get class variables (spin operators, and dimension of the Hilbert space
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        dim = self.dim
        
        # create angle arrays that will broadcast properly 
        # (operations are multiplication, not matrix mult. until the performing the
        # reverse unitary transformation at the end)
        phi = phi_1D[:, np.newaxis, np.newaxis]
        theta = theta_1D[:, np.newaxis, np.newaxis]
        psi = psi_1D[:, np.newaxis, np.newaxis]

        # build the arrays to be exponentiated
        z_exponent = -1j*phi*Iz
        xp_exponent = -1j*theta*(Ix*np.cos(phi) + Iy*np.sin(phi))
        zp_exponent = -1j*psi*(Ix*np.sin(phi)*np.sin(theta)
                               - Iy*np.sin(theta)*np.cos(phi) + Iz*np.cos(theta))

        # find eigenvalues e and eigenvectors U
        e_z, U_z = np.linalg.eig(z_exponent)
        e_xp, U_xp = np.linalg.eig(xp_exponent)
        e_zp, U_zp = np.linalg.eig(zp_exponent)

        # conjugate-transpose that leaves the blocks intact and performs
        # as expected for the individual rotation matrices
        U_z_i = np.transpose(np.conj(U_z), (0, 2, 1))
        U_xp_i = np.transpose(np.conj(U_xp), (0, 2, 1))
        U_zp_i = np.transpose(np.conj(U_zp), (0, 2, 1))

        # exponentiate the evals
        exp_e_z = np.exp(e_z)
        exp_e_xp = np.exp(e_xp)
        exp_e_zp = np.exp(e_zp)

        # create exponentiated diagonal matrix of evals
        expD_z = np.eye(dim)*exp_e_z[:, np.newaxis, :]
        expD_xp = np.eye(dim)*exp_e_xp[:, np.newaxis, :]
        expD_zp = np.eye(dim)*exp_e_zp[:, np.newaxis, :]

        # apply the reverse unitary transpformation using evecs to get
        # the spin-space rotation matrix
        Rz = U_z@expD_z@U_z_i
        Rxp = U_xp@expD_xp@U_xp_i
        Rzp = U_zp@expD_zp@U_zp_i

        # produce the inverse rotation matrix via conjugate-transpose
        Rzi = np.transpose(np.conj(Rz), (0, 2, 1))
        Rxpi = np.transpose(np.conj(Rxp), (0, 2, 1))
        Rzpi = np.transpose(np.conj(Rzp), (0, 2, 1))

        # combine the matrices and their inverses and return them as a tuple

        return (Rzp@Rxp@Rz, Rzi@Rxpi@Rzpi)



    def freq_prob_trans_ed(self,
                           H0,
                           Ka,
                           Kb,
                           Kc,
                           va,
                           vb,
                           vc,
                           eta,
                           rm_SRm_tuple,
                           rm_SRm_init_tuple=None,
                           Hinta=0.0,
                           Hintb=0.0,
                           Hintc=0.0,
                           mtx_elem_min=0.1,
                           min_freq=None,
                           max_freq=None
                          ):
        """
        This method calculates the resonant frequences for a given nucleus for the input parameters:
        args:
            - Ka, Kb, and Kc are the components of the Knight shift (or rather just shift tensor) in percent
            - vc and eta are the quadrupolar parameters of the EFG tensor in MHz
            - r_SR_tuple expects a four element tuple of the form: 
                (r, ri, SR, SRi) so these do not need to be recalculated 
                each iteration of the function during curve fitting.
                progressive rotations about those axes (in the code, the rotations are 
                performed in the order z, xprime, zprime euler angle convention)
        kwargs:
            - mtx_elem_min is the minimum value a matrix element for a given transition must
                have before being discarded after matrix diagonalization
            - min_freq is the lower frequency limit of the calculation, default = 0.1 MHz
            - max_freq is the highest frequency limit of the calculation, default = 500 MHz
        notes:
            - if spectra are not making sense below, may need to mess around with some of the args and kwargs
            - need to add removing points outside of the frequency range desired.
        """
        I0 = self.I0
        dim = self.dim
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Iplus = self.Iplus
        Iminus = self.Iminus
        I2 = self.I2
        II3 = self.II3
        Ivec = self.Ivec
        # allow for curve fitting with both vc+eta OR va, vb, and vc
        try:
            if (va.value == float('-inf') or va.value == float('inf')) and (vb.value == float('-inf') or vb.value == float('inf')) and (eta.value != float('-inf') or eta.value != float('inf')):
                va, vb = self.vx_vy_from_vz_eta(vc, eta)
        except:
            print("value doesn't exist, therefore we are not curve fitting...")
        # calculate va and vb if necessary
        if (va == None) and (vb == None) and (eta != None):
            va, vb = self.vx_vy_from_vz_eta(vc, eta)
        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        if rm_SRm_init_tuple is not None:
            r_init, ri_init, SR_init, SRi_init  = rm_SRm_init_tuple
        r, ri, SR, SRi  = rm_SRm_tuple
        # Knight shift tensor and rotations
        KtensorInit = np.array([[Ka/100.0, 0, 0],
                                [0, Kb/100.0, 0],
                                [0, 0, Kc/100.0]])
        if rm_SRm_init_tuple is not None:
            KtensorInit = r_init@KtensorInit@ri_init
        Ktensor = r@KtensorInit@ri
        ## Zeeman Hamiltonian
        H0vec = np.array([0, 0, H0])
        eyePlusK = (II3 + Ktensor)
        KdotH0 = H0vec@eyePlusK
        Hze = gamma*(KdotH0[:, 0, np.newaxis, np.newaxis]*Ix + KdotH0[:, 1, np.newaxis, np.newaxis]*Iy + KdotH0[:, 2, np.newaxis, np.newaxis]*Iz)
        ## allow for an internal magnetic field due to hyperfine coupling to magnetic order
        # may want to modify this to actually be S@A@Ivec in the future 
        HintVecInit = np.array([Hinta,Hintb,Hintc])
        if rm_SRm_init_tuple is not None:
            HintVecInit = np.squeeze(r_init@HintVecInit) # dimensions of the resultant vector caused issues with casting during the next line, so needed to squeeze the resulting vector. in the next line it is desireable.
        HintVec = r@HintVecInit
        HzeInt =  gamma*(HintVec[:, 0, np.newaxis, np.newaxis]*Ix + HintVec[:, 1, np.newaxis, np.newaxis]*Iy + HintVec[:, 2, np.newaxis, np.newaxis]*Iz)
        ## Quadrupole Hamiltonian
        try:
            HqInit = (vc/6.0)*(3.0*Iz@Iz - I2 + ((va - vb)/vc)*(Ix@Ix - Iy@Iy))
            if rm_SRm_init_tuple is not None:
                HqInit = SR_init@HqInit@SRi_init
            Hq = SR@HqInit@SRi
            ## total Hamiltonian
            Htot = Hze + Hq + HzeInt
        except ZeroDivisionError:
            print('HqInit zero division error')
            HqInit = 0.0
            Htot = Hze + HzeInt
        ## calculate eigenvales and eigenvectors
        evals, U = np.linalg.eig(Htot)
        ## build U and Ui which are the matricies of the eigenvectors which will be used to diagonalize the Hamiltonian
        Ui = np.transpose(np.conj(U),(0,2,1))
        
        ## use U and Ui on Ix to get the matrix elements for all possible transitions
        H1x = Ui@Ix@U
        trans_probs_array = np.abs(H1x)**2
        # There seems to have been an issue where, where, when eta is nonzero, the following transition probability 
        # intensities were not all captured by |<n_i|I+|n_j>|^2 for all i and j for eta != 0, instead the Ix = (Iplus + Iminus) / 2 operator 
        # should be used to calculate the transition intensities. This may be related to the mixing of eigenstates for eta != 0, and therefore
        # the spin lowering operator must be used to capture some of the allowed superposed-state transitions. This also seems to only matter for
        # I0 > 3/2. Tested on 115In(2) site in CeRhIn5
        
        # allow the user to throw away very low probability transitions
        allowed_trans_idxs = np.argwhere(trans_probs_array > mtx_elem_min)
        allowed_eval_m_idxs = allowed_trans_idxs[:,[0,1]]
        allowed_eval_mPlus1_idxs = allowed_trans_idxs[:,[0,2]]
        allowed_eval_m = evals[allowed_eval_m_idxs[:,0],allowed_eval_m_idxs[:,1]]
        allowed_eval_mPlus1 = evals[allowed_eval_mPlus1_idxs[:,0],allowed_eval_mPlus1_idxs[:,1]]
        #freq_array = np.abs(allowed_eval_mPlus1 - allowed_eval_m)
        freq_array = allowed_eval_mPlus1 - allowed_eval_m
        prob_array = trans_probs_array[trans_probs_array > mtx_elem_min]
        #remove zero frequency transitions
        prob_array = prob_array[freq_array > 0]
        freq_array = np.abs(freq_array[freq_array > 0])
        # use U and Ui to get the corresponding transition numbers (absolute value 0=cent, 1=1st sat, 2=2nd sat, to indicate simply which satellite it is) for convolution of delta_vq
        m_val_mtcs = Ui@Iz@U #ahoy! this is not correct, need to return to this later
        #ahoy! print('m_val_mtcs = Ui@Iz@U is incorrect!, trans is not correct... actually a superposition of states!') need to deal with this eventually, but for now, will let it stand so we can keep simulating and fitting... see examples of plot_elevels_vs_field to get an idea of this. the eigenstate character also changes with angle and other parameters during the crossover from quadrupole dominated to zeeman (internal zeeman) dominated
        trans = (m_val_mtcs - 0.5).real
        trans = np.diagonal(trans, axis1=1, axis2=2)
        #trans = np.abs(np.around(trans[np.where(trans>-I0)]))
        trans = np.around(trans[np.where(trans>-I0)])
        trans_array = trans.astype(int)
        return (freq_array, prob_array, trans_array)


    def gaussian(self,
                 x,
                 mu,
                 FWHM,
                 intensity):
        """
        Gaussian function, see https://en.wikipedia.org/wiki/Normal_distribution
        Inputs:
            - x input numpy array for x axis (so frequency or field in MHz or
                Tesla in this case)
            - mu is the peak position
            - FWHM (full width at half maximum, also commonly written as Gamma)
            - intensity is the integral, so we can scale the resonance via
                calculated transition probability matrix elements
        """
        sigma = FWHM/(2*(2*np.log(2))**0.5)
        return intensity/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((x - mu)/sigma)**2)


    def lorentzian(self,
                   x,
                   x0,
                   FWHM,
                   intensity):
        """
        Lorentzian function, see https://mathworld.wolfram.com/LorentzianFunction.html
        Inputs:
            - x input numpy array for x axis (so frequency or field in MHz or
                Tesla in this case)
            - x0 is the peak position
            - FWHM (full width at half maximum, also commonly written as Gamma)
            - intensity is the integral, so we can scale the resonance via
                calculated transition probability matrix elements
        """
        return (intensity/np.pi)*((0.5*FWHM)/((x - x0)**2 + (0.5*FWHM)**2))


    def freq_spec_ed(self,
                     x,
                     H0,
                     Ka,
                     Kb,
                     Kc,
                     va,
                     vb,
                     vc,
                     eta,
                     rm_SRm_tuple,
                     Hinta=0.0,
                     Hintb=0.0,
                     Hintc=0.0,
                     mtx_elem_min=0.1,
                     min_freq=None,
                     max_freq=None,
                     FWHM=0.01,
                     FWHM_vQ=0.0,
                     line_shape_func='gauss'):
        """
        This method calculates a spectrum at given x values (frequencies) based
        on the input parameters:
        args:
            - x is expected to be a numpy array of dimension 0 containing 
                frequency values at which the spectrum should be calculated 
                in units of MHz
            - H0 applied external field
            - Ka, Kb, and Kc are the components of the shift tensor in percent
            - vc and eta are the quadrupole parameters of the EFG tensor in MHz
                (alternatively one can specify va, vb, and vc)
            - r_SR_tuple expects a four element tuple of the form:
                (r, ri, SR, SRi) so these do not need to be recalculated 
                each iteration of the function during curve fitting.
                progressive rotations about those axes (in the code, the 
                rotations are performed in the order z, xprime, zprime euler 
                angle convention)
        kwargs:
            - Hinta, Hintb, and Hintc define the internal hyperfine field 
                vector
            - mtx_elem_min is the minimum value a matrix element for a given 
                transition must have before being discarded after matrix 
                diagonalization
            - min_freq and max_freq are the lower and upper limits of the calc
            - FWHM characterizes the magnetic broadening of the spectrum (MHz)
            - FWHM_vQ characterizes the quadrupolar broadening in MHz, that
                is scaled with the satellite transition. (Note that we do not
                account for second order quadrupole broadening!) Total line width
                for a satellite transition will be FWHM + n*FWHM_vQ, where n is 
                the index of the satellite transition (1st sat has n=1, 2nd has 
                n=2, etc)
            - line_shape_func can be 'gaussian' or 'lorentz' can add more later
        notes:
            - if spectra are not making sense below, may need to mess around 
                with some of the args and kwargs. Especially extra peaks that 
                are disallowed transitions that exist but do not have a larger
                enough matrix element to observe
            - The scaling of the broadening of the satellite transitions may 
                break down for eta != 0, or in the case of Hzeeman ~ Hquad for
                arbitrary rotations. This will require testing and possibly some
                literature review.
        """
        I0 = self.I0
        (freq_array,
         prob_array,
         trans_array) = self.freq_prob_trans_ed(H0=H0, 
                                                Ka=Ka, 
                                                Kb=Kb, 
                                                Kc=Kc, 
                                                va=va,
                                                vb=vb,
                                                vc=vc,
                                                eta=eta,
                                                rm_SRm_tuple=rm_SRm_tuple,
                                                Hinta=Hinta,
                                                Hintb=Hintb,
                                                Hintc=Hintc,
                                                mtx_elem_min=mtx_elem_min,
                                                min_freq=min_freq,
                                                max_freq=max_freq)
        
        print('freq_spec_ed; freq_array =', freq_array)    
        print('freq_spec_ed; prob_array =', prob_array)    
        print('freq_spec_ed; trans_array =', trans_array)
        # need to loop through the transitions and set their appropriate linwidths,
        # then can calculate the individual spectra using the gaussian or lorentzian
        # functions with properly scaled FWHM (so magnetic for all plus FWHM_dvQ_MHz*float(trans))
        summed_spectrum = np.zeros(shape=x.shape)
        for i in range(freq_array.shape[0]):
            if trans_array[i] == 0:
                FWHM_total = FWHM
            else:
                FWHM_total = FWHM + FWHM_vQ*float(trans_array[i]) # note this might breakdown for eta != 0
            if line_shape_func == 'gauss':
                spectrum_i = self.gaussian(x=x,
                                           mu=freq_array[i],
                                           FWHM=FWHM_total,
                                           intensity=prob_array[i])
            elif line_shape_func == 'lor':
                spectrum_i = self.lorentzian(x=x,
                                             x0=freq_array[i],
                                             FWHM=FWHM_total,
                                             intensity=prob_array[i])
            else:
                print('line shape function ', line_shape_func, 'not implemented.')
            summed_spectrum += spectrum_i
        return summed_spectrum


    def freq_vs_field_ed(self, 
                           H0, 
                           Ka, 
                           Kb, 
                           Kc, 
                           va,
                           vb,
                           vc,
                           eta,
                           rm_SRm_tuple,
                           Hinta=0.0,
                           Hintb=0.0,
                           Hintc=0.0,
                           mtx_elem_min=0.1,
                           min_freq=0.1,
                           max_freq=500
                          ):
        """
        This method calculates the resonant frequences for a given nucleus for the input parameters:
        args:
            - Ka, Kb, and Kc are the components of the Knight shift (or rather just shift tensor) in percent
            - vc and eta are the quadrupolar parameters of the EFG tensor in MHz
            - r_SR_tuple expects a four element tuple of the form: 
                (r, ri, SR, SRi) so these do not need to be recalculated 
                each iteration of the function during curve fitting.
                progressive rotations about those axes (in the code, the rotations are 
                performed in the order z, xprime, zprime euler angle convention)
        kwargs:
            - mtx_elem_min is the minimum value a matrix element for a given transition must
                have before being discarded after matrix diagonalization
            - min_freq is the lower frequency limit of the calculation, default = 0.1 MHz
            - max_freq is the highest frequency limit of the calculation, default = 500 MHz
        notes:
            - if spectra are not making sense below, may need to mess around with some of the args and kwargs
            - need to add removing points outside of the frequency range desired.
        """
        I0 = self.I0
        dim = self.dim
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Iplus = self.Iplus
        Iminus = self.Iminus
        I2 = self.I2
        II3 = self.II3
        Ivec = self.Ivec
        
        if va==None and vb==None and eta!=None:
            va,vb = self.vx_vy_from_vz_eta(vc,eta)

        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        r, ri, SR, SRi  = rm_SRm_tuple

        # Knight shift tensor and rotations
        KtensorInit = np.array([[Ka/100.0, 0, 0],
                                [0, Kb/100.0, 0],
                                [0, 0, Kc/100.0]])
        Ktensor = r@KtensorInit@ri

        ## Zeeman Hamiltonian
        H0_array=H0
        H0vec=np.zeros(shape=(H0_array.shape[0],3,1)) # the extra dimension at the end is to make sure that
                                                      # the @ operator in the line below KdotH0 = eyePlusK@H0vec
                                                      # works properly to perform matrix multiplication (dot product)
                                                      # on the the stacked matrices:eyePlusK times the stacked 
                                                      # vectors:H0vec, then we will have to reshape again later
        H0vec[:,2,0]=H0_array # populate the z directions of the H0vec from the input array

        eyePlusK = (II3 + Ktensor)

        KdotH0 = eyePlusK@H0vec
        KdotH0 = np.reshape(KdotH0,(H0_array.shape[0],3))

        Hze = gamma*(KdotH0[:,0,np.newaxis,np.newaxis]*Ix + KdotH0[:,1,np.newaxis,np.newaxis]*Iy + KdotH0[:,2,np.newaxis,np.newaxis]*Iz)

        ## allow for an internal magnetic field due to hyperfine coupling to magnetic order
        # may want to modify this to actually be S@A@Ivec in the future 
        HintVecInit = np.array([Hinta,Hintb,Hintc])
        HintVec = r@HintVecInit
        HzeInt =  gamma*(HintVec[:,0,np.newaxis,np.newaxis]*Ix + HintVec[:,1,np.newaxis,np.newaxis]*Iy + HintVec[:,2,np.newaxis,np.newaxis]*Iz)

        ## Quadrupole Hamiltonian
        try:
            HqInit = (vc/6.0)*(3.0*Iz@Iz - I2 + ((va - vb)/vc)*(Ix@Ix - Iy@Iy))
            Hq = SR@HqInit@SRi

            ## total Hamiltonian
            Htot = Hze + Hq + HzeInt
        except ZeroDivisionError:
            print('HqInit zero division error')
            HqInit = 0.0
            Htot = Hze + HzeInt

        ## calculate eigenvales and eigenvectors
        evals, U = np.linalg.eig(Htot)
            
        ## build U and Ui which are the matricies of the eigenvectors which will be used to diagonalize the Hamiltonian
        Ui = np.transpose(np.conj(U),(0,2,1))

        ## use U and Ui on Ix to get the matrix elements for all possible transitions
        H1x = Ui@Ix@U
        trans_probs_array = np.abs(H1x)**2
        # There seems to have been an issue where, where, when eta is nonzero, the following transition probability 
        # intensities were not all captured by |<n_i|I+|n_j>|^2 for all i and j for eta != 0, instead the Ix = (Iplus + Iminus) / 2 operator 
        # should be used to calculate the transition intensities. This may be related to the mixing of eigenstates for eta != 0, and therefore
        # the spin lowering operator must be used to capture some of the allowed superposed-state transitions. This also seems to only matter for
        # I0 > 3/2. Tested on 115In(2) site in CeRhIn5
        
        allowed_trans_idxs = np.argwhere(trans_probs_array > mtx_elem_min)
        # print('allowed_trans_idxs:')
        # print(allowed_trans_idxs)
        
        #need to index the H0 wave to figure out which field value actually has a transition within the tolerance
        # maybe can use the argwhere above here (or rather the first dimension thereof)

        allowed_eval_m_idxs = allowed_trans_idxs[:,[0,1]]
        # print('allowed_eval_m_idxs:')
        # print(allowed_eval_m_idxs)
        allowed_eval_mPlus1_idxs = allowed_trans_idxs[:,[0,2]]
        # print('allowed_eval_mPlus1_idxs:')
        # print(allowed_eval_mPlus1_idxs)
        allowed_eval_m = evals[allowed_eval_m_idxs[:,0],allowed_eval_m_idxs[:,1]]
        # print('allowed_eval_m:')
        # print(allowed_eval_m)
        allowed_eval_mPlus1 = evals[allowed_eval_mPlus1_idxs[:,0],allowed_eval_mPlus1_idxs[:,1]]
        # print('allowed_eval_mPlus1:')
        # print(allowed_eval_mPlus1)

        # print('allowed_eval_m_idxs[:,0]')
        # print(allowed_eval_m_idxs[:,0])
        H0_array = H0_array[allowed_eval_m_idxs[:,0]]
        # print('H0_array_out')
        # print(H0_array_out)
        freq_array = np.abs(allowed_eval_mPlus1-allowed_eval_m)
        #freq_array = np.abs(allowed_eval_mPlus1) #ahoy!
        prob_array = trans_probs_array[trans_probs_array > mtx_elem_min]

        # use U and Ui to get the corresponding transition numbers (absolute value 0=cent, 1=1st sat, 2=2nd sat, to indicate simply which satellite it is) for convolution of delta_vq
        # the following is incorrect!
        m_val_mtcs = Ui@Iz@U
        
        trans = (m_val_mtcs - 0.5).real
        trans = np.diagonal(trans, axis1=1, axis2=2)
        #trans = np.abs(np.around(trans[np.where(trans>-I0)]))
        trans = np.around(trans[np.where(trans>-I0)])
        trans_array = trans.astype(int)

        # in the following block of code, there is the possibility that the following
        # error might arise for values of mtx_elem_min that are small (smaller than ~0.5 so far):
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


        # want to keep only the largest matrix element when both H0_array_out[i] and freq_array[i] 
        # have the same values paired and repeated.
        # My H0 array that contains duplicates of pairs of H0 and f, which have different probs
        input_array = np.column_stack((H0_array, freq_array, prob_array))
        # print(input_array)
        #print(input_array[:,[0,1]])
        #np.unique gets us the unique array (sorted first by field and then frequency) and with the option inverse
        # gives us the indicies of the location of the duplicates along axis zero.
        unique_H0_f, idxs = np.unique(input_array[:,[0,1]], return_inverse=True, axis=0)

        # print(unique_H0_f)
        # print(idxs)
        #prob_array_out=prob_array
        # we can then use np.where to get the probabilities associated with all the duplicates
        # then the probabilities can be summed... doing this in a loop, which is slow
        # could likely be accomplished in a numerical pythonic way to improve speed if
        # that is required
        prob_array_out = np.array([])
        for i in range(len(unique_H0_f)):
            duplicate_idxs = np.where(idxs.flatten()==i)
            prob_array_out = np.append(prob_array_out, np.sum(prob_array[duplicate_idxs]))

        f0 = min_freq + (max_freq - min_freq)/2
        delta_f0 = max_freq - min_freq
        # frequency pruning:
        f0_close_bool_array=np.isclose(f0, 
                                       unique_H0_f[:,1], 
                                       rtol=0.0, 
                                       atol=delta_f0/2.0)
        #print('f0_close_bool_array.shape='+str(f0_close_bool_array.shape))
        H0_array_out = unique_H0_f[f0_close_bool_array,[0]]
        freq_array_out = unique_H0_f[f0_close_bool_array,[1]]
        prob_array_out = prob_array_out[f0_close_bool_array]

        try:
            trans_array_out = trans_array[f0_close_bool_array]
        except:
            print('ERROR in array sizes',
                  'proceeding with trans_array_out = np.zeros(shape=H0_array_out.shape)')
            # print('H0_array_out.shape=' + str(H0_array_out.shape))
            #print(H_array_out)
            # print('freq_array.shape=' + str(freq_array_out.shape))
            #print(freq_array)
            # print('prob_array.shape=' + str(prob_array_out.shape))
            #print(prob_array)
            # print('trans_array.shape=' + str(trans_array.shape))
            #print(trans_array)
            trans_array_out = np.zeros(shape=H0_array_out.shape)


        return np.column_stack((H0_array_out,freq_array_out,prob_array_out,trans_array_out))


    def freq_prob_trans_ed_HS(self, 
                               H0, 
                               Ka, 
                               Kb, 
                               Kc, 
                               va,
                               vb,
                               vc,
                               eta,
                               rm_SRm_tuple,
                               Hinta=0.0,
                               Hintb=0.0,
                               Hintc=0.0,
                               mtx_elem_min=0.1,
                               f0=None,
                               delta_f0=None
                              ):
        """
        This method calculates the resonant frequences for a given nucleus for the input parameters:
        args:
            - Ka, Kb, and Kc are the components of the Knight shift (or rather just shift tensor) in percent
            - vc and eta are the quadrupolar parameters of the EFG tensor in MHz, alternatively one can 
                overdefine the efg tensor by providing va, vb, and vc and setting eta=None
            - r_SR_tuple expects a four element tuple of the form: 
                (r, ri, SR, SRi) so these do not need to be recalculated 
                each iteration of the function during curve fitting.
                progressive rotations about those axes (in the code, the rotations are 
                performed in the order z, xprime, zprime euler angle convention)
        kwargs:
            - Hinta, Hintb, and Hintc are the elements of the internal field vector at the nuclear site(s)
            - mtx_elem_min is the minimum value a matrix element for a given transition must
                have before being discarded after matrix diagonalization
            - f0 is the observed frequency of a field sweep
            - delta_f0 is tolerance around the resonance frequency over which transitions will be kept ~0.01 - 0.05
                seem reasonable. Smaller or larger values may cause problems, but I leave this as a parameter for 
                the user to modify as needed

        notes:
            - if spectra are not making sense below, may need to mess around with some of the args and kwargs
            - need to add removing points outside of the frequency range desired.
        """
        I0 = self.I0
        dim = self.dim
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz
        Iplus = self.Iplus
        Iminus = self.Iminus
        I2 = self.I2
        II3 = self.II3
        Ivec = self.Ivec
        
        if va==None and vb==None and eta!=None:
            va,vb = self.vx_vy_from_vz_eta(vc,eta)

        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        r, ri, SR, SRi  = rm_SRm_tuple

        # Knight shift tensor and rotations
        KtensorInit = np.array([[Ka/100.0, 0, 0],
                                [0, Kb/100.0, 0],
                                [0, 0, Kc/100.0]])
        Ktensor = r@KtensorInit@ri

        ## Zeeman Hamiltonian
        H0_array=H0
        H0vec=np.zeros(shape=(H0_array.shape[0],3,1)) # the extra dimension at the end is to make sure that
                                                      # the @ operator in the line below KdotH0 = eyePlusK@H0vec
                                                      # works properly to perform matrix multiplication (dot product)
                                                      # on the the stacked matrices:eyePlusK times the stacked 
                                                      # vectors:H0vec, then we will have to reshape again later
        H0vec[:,2,0]=H0_array # populate the z directions of the H0vec from the input array
        # print('H0vec:')
        # print(H0vec.shape)
        # print(H0vec)
        eyePlusK = (II3 + Ktensor)
        # print('eyePlusK:')
        # print(eyePlusK.shape)
        # print(eyePlusK)
        KdotH0 = eyePlusK@H0vec
        KdotH0 = np.reshape(KdotH0,(H0_array.shape[0],3))
        # print('KdotH0')
        # print(KdotH0)
        Hze = gamma*(KdotH0[:,0,np.newaxis,np.newaxis]*Ix + KdotH0[:,1,np.newaxis,np.newaxis]*Iy + KdotH0[:,2,np.newaxis,np.newaxis]*Iz)

        ## allow for an internal magnetic field due to hyperfine coupling to magnetic order
        # may want to modify this to actually be S@A@Ivec in the future 
        HintVecInit = np.array([Hinta,Hintb,Hintc])
        HintVec = r@HintVecInit
        HzeInt =  gamma*(HintVec[:,0,np.newaxis,np.newaxis]*Ix + HintVec[:,1,np.newaxis,np.newaxis]*Iy + HintVec[:,2,np.newaxis,np.newaxis]*Iz)

        ## Quadrupole Hamiltonian
        try:
            HqInit = (vc/6.0)*(3.0*Iz@Iz - I2 + ((va - vb)/vc)*(Ix@Ix - Iy@Iy))
            Hq = SR@HqInit@SRi

            ## total Hamiltonian
            Htot = Hze + Hq + HzeInt
        except ZeroDivisionError:
            print('HqInit zero division error')
            HqInit = 0.0
            Htot = Hze + HzeInt

        ## calculate eigenvales and eigenvectors
        evals, U = np.linalg.eig(Htot)
            
        ## build U and Ui which are the matricies of the eigenvectors which will be used to diagonalize the Hamiltonian
        Ui = np.transpose(np.conj(U),(0,2,1))

        ## use U and Ui on Ix to get the matrix elements
        H1x = Ui@Ix@U
        trans_probs_array = np.abs(H1x)**2
        allowed_trans_idxs = np.argwhere(trans_probs_array > mtx_elem_min)
        #print('allowed_trans_idxs:')
        #print(allowed_trans_idxs)
        #need to index the H0 wave to figure out which field value actually has a transition within the tolerance
        # maybe can use the argwhere above here (or rather the first dimension thereof)

        allowed_eval_m_idxs = allowed_trans_idxs[:,[0,1]]
        #print('allowed_eval_m_idxs:')
        #print(allowed_eval_m_idxs)
        allowed_eval_mPlus1_idxs = allowed_trans_idxs[:,[0,2]]
        #print('allowed_eval_mPlus1_idxs:')
        #print(allowed_eval_mPlus1_idxs)
        allowed_eval_m = evals[allowed_eval_m_idxs[:,0],allowed_eval_m_idxs[:,1]]
        #print('allowed_eval_m:')
        #print(allowed_eval_m)
        allowed_eval_mPlus1 = evals[allowed_eval_mPlus1_idxs[:,0], allowed_eval_mPlus1_idxs[:,1]]
        #print('allowed_eval_mPlus1:')
        #print(allowed_eval_mPlus1)

        #print('allowed_eval_m_idxs[:,0]')
        #print(allowed_eval_m_idxs[:,0])
        H0_array = H0_array[allowed_eval_m_idxs[:,0]]
        freq_array = allowed_eval_mPlus1 - allowed_eval_m
        prob_array = trans_probs_array[trans_probs_array > mtx_elem_min]
        #remove zero frequency transitions
        H0_array_out = H0_array[freq_array > 0]
        prob_array = prob_array[freq_array > 0]
        freq_array = np.abs(freq_array[freq_array > 0])

        # use U and Ui to get the corresponding transition numbers (absolute value 0=cent, 1=1st sat, 2=2nd sat, to indicate simply which satellite it is) for convolution of delta_vq
        m_val_mtcs = Ui@Iz@U
        trans = (m_val_mtcs - 0.5).real
        #print('trans before diag:')
        #print(trans)
        trans = np.diagonal(trans, axis1=1, axis2=2)
        #print('trans after diag:')
        #print(trans)
        trans = np.abs(np.around(trans[np.where(trans>-I0)]))
        trans_array = trans.astype(int)
        print('trans.shape, trans:')
        print(trans.shape, trans)
        print('freq_array.shape, freq_array:')
        print(freq_array.shape, freq_array)
        print('prob_array.shape, prob_array')
        print(prob_array.shape, prob_array)

        # in the following block of code, there is the possibility that the following
        # error might arise for values of mtx_elem_min that are small (smaller than ~0.5 so far):
        # Traceback (most recent call last):
        #   File "plot_field_spectrum_multisite.py", line 254, in <module>
        #     out_filename=sim_export_file
        #   File "/Users/apd/gd/code/python/IFW/pySimNMR/v0.6/pySimNMR.py", line 829, in field_spec_edpp
        #     delta_f0=delta_f0
        #   File "/Users/apd/gd/code/python/IFW/pySimNMR/v0.6/pySimNMR.py", line 657, in freq_prob_trans_ed_HS
        #     trans_array = trans_array[f0_close_bool_array]
        # IndexError: boolean index did not match indexed array along dimension 0; dimension is 3000 but corresponding boolean dimension is 3387
        # It seems that one possible solution here would be to catch this error and increase 
        # mtx_elem_min automatically until the error goes away... may try to implement this.
        # the following line searches for values within the freq_array that are within delta_f0/2 of 
        # the observed frequency f0 and returns an array of the same shape as freq_array with boolean values
        # to be used in the following 4 lines to select the elements that we are after (that is, we need a small
        # range of resonant frequencies to keep)
        f0_close_bool_array = np.isclose(f0, freq_array, rtol=0.0, atol=delta_f0/2.0)
        print('f0_close_bool_array.shape')
        print(f0_close_bool_array.shape)
        print('prob_array.shape')
        print(prob_array.shape)
        prob_array = prob_array[f0_close_bool_array]
        print('prob_array.shape after')
        print(prob_array.shape)
        print('trans_array.shape')
        print(trans_array.shape)
        trans_array = trans_array[f0_close_bool_array]  
        print('trans_array.shape after')
        print(trans_array.shape)

        H0_array_out = H0_array_out[f0_close_bool_array]

        freq_array = freq_array[f0_close_bool_array]

        return (H0_array_out, freq_array, prob_array, trans_array)


    def freq_spec_edpp(self,
                       H0,
                       Ka,
                       Kb,
                       Kc,
                       va,
                       vb,
                       vc,
                       eta,
                       rm_SRm_tuple,
                       Hinta=0.0,
                       Hintb=0.0,
                       Hintc=0.0,
                       mtx_elem_min=0.1,
                       min_freq=None,
                       max_freq=None,
                       FWHM_MHz=0.02,
                       FWHM_dvQ_MHz=0.0,
                       broadening_func='gauss',
                       baseline=0.1,
                       nbins=1000,
                       save_files_bool=False,
                       out_filename=''):
        """
        This method calculates the resonant frequencies for a given nucleus for the input parameters:
        args:
            - Ka, Kb, and Kc are the components of the Knight shift (or rather just shift tensor) in percent
            - vc and eta are the quadrupole parameters of the EFG tensor in MHz
            - r_SR_tuple expects a four element tuple of the form:
                (r, ri, SR, SRi) so these do not need to be recalculated 
                each iteration of the function during curve fitting.
                progressive rotations about those axes (in the code, the rotations are
                performed in the order z, xprime, zprime euler angle convention)
        kwargs:
            - mtx_elem_min is the minimum value a matrix element for a given transition must
                have before being discarded after matrix diagonalization
            - min_freq is the lower frequency limit of the calculation, default = 0.1 MHz
            - max_freq is the highest frequency limit of the calculation, default = 500 MHz
        notes:
            - if spectra are not making sense below, may need to mess around with some of the args and kwargs
        """
        I0 = self.I0
        
        (freq_array,
         prob_array,
         trans_array) = self.freq_prob_trans_ed(H0=H0, 
                                                Ka=Ka, 
                                                Kb=Kb, 
                                                Kc=Kc, 
                                                va=va,
                                                vb=vb,
                                                vc=vc,
                                                eta=eta,
                                                rm_SRm_tuple=rm_SRm_tuple,
                                                Hinta=Hinta,
                                                Hintb=Hintb,
                                                Hintc=Hintc,
                                                mtx_elem_min=mtx_elem_min,
                                                min_freq=None,
                                                max_freq=None)

        if min_freq == None:
            min_freq = freq_array.min() - baseline
        if max_freq == None:
            max_freq = freq_array.max() + baseline

        if FWHM_dvQ_MHz < 1e-6:
            hist, bin_edges = np.histogram(freq_array,
                                           bins=int(nbins),
                                           range=(min_freq, max_freq),
                                           weights=prob_array)
            bin_edges=bin_edges[:-1]
            histogram = np.column_stack((bin_edges, hist))
            convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                             y_array=hist,
                                                             FWHM=FWHM_MHz,
                                                             mode=broadening_func)
            convolved_histogram = np.column_stack((bin_edges, convolved_hist_intensity))
        else:
            conv_hist_intensity_sum = np.zeros(int(nbins))
            for trans in range(int(I0 + 0.5)):
                trans_freqs = np.array(freq_array[np.where(trans_array==trans)])
                trans_probs = np.array(prob_array[np.where(trans_array==trans)])

                # make histograms
                hist, bin_edges = np.histogram(trans_freqs,
                                              bins=int(nbins),
                                              range=(min_freq,max_freq),
                                              weights=trans_probs)
                bin_edges=bin_edges[:-1]

                # convolve with df and dvQ*trans as necessary
                if trans == 0:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_MHz,
                                                                mode=broadening_func)
                else:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_MHz,
                                                                mode=broadening_func)
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=conv_hist_intensity,
                                                                FWHM=FWHM_dvQ_MHz*float(trans),
                                                                mode=broadening_func)
                conv_hist_intensity_sum = conv_hist_intensity_sum + conv_hist_intensity
            convolved_histogram = np.column_stack((bin_edges,conv_hist_intensity_sum))

        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_pp.txt'
                out_filename_hist = out_filename + '_sim_pp_hist.txt'
                out_filename_hist_conv = out_filename + '_sim_pp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_pp.txt'
                out_filename_hist = 'sim_pp_hist.txt'
                out_filename_hist_conv = 'sim_pp_hist_conv.txt'
            np.savetxt(out_filename_hist_conv, convolved_histogram)

        return convolved_histogram


    def field_spec_edpp(self,
                        f0,
                        H0_array,
                        Ka,
                        Kb,
                        Kc,
                        va,
                        vb,
                        vc,
                        eta,
                        rm_SRm_tuple,
                        Hinta=0.0,
                        Hintb=0.0,
                        Hintc=0.0,
                        mtx_elem_min=0.1, 
                        min_field=None, 
                        max_field=None,
                        delta_f0=0.05,
                        FWHM_T=0.02,
                        FWHM_dvQ_T=0.0,
                        broadening_func='gauss',
                        baseline=0.1,
                        nbins=1000,
                        save_files_bool=False,
                        out_filename=''):
        """
        man page text here
        """
        I0 = self.I0

        (H0_array_out,
         freq_array,
         prob_array,
         trans_array) = self.freq_prob_trans_ed_HS(H0=H0_array,
                                                   Ka=Ka,
                                                   Kb=Kb,
                                                   Kc=Kc,
                                                   va=va,
                                                   vb=vb,
                                                   vc=vc,
                                                   eta=eta,
                                                   rm_SRm_tuple=rm_SRm_tuple,
                                                   Hinta=Hinta,
                                                   Hintb=Hintb,
                                                   Hintc=Hintc,
                                                   mtx_elem_min=mtx_elem_min,
                                                   f0=f0,
                                                   delta_f0=delta_f0)
        #print(freq_array)
        if min_field==None:
            min_field = H0_array_out.min()-baseline
        if max_field==None:
            max_field = H0_array_out.max()+baseline

        if FWHM_dvQ_T<1e-6:
            hist, bin_edges = np.histogram(H0_array_out,
                                           bins=int(nbins),
                                           range=(min_field, max_field),
                                           weights=prob_array
                                          )
            bin_edges=bin_edges[:-1]
            histogram = np.column_stack((bin_edges,hist))
            convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                             y_array=hist,
                                                             FWHM=FWHM_T,
                                                             mode=broadening_func)
            convolved_histogram = np.column_stack((bin_edges,convolved_hist_intensity))
        else:
            conv_hist_intensity_sum = np.zeros(int(nbins))
            for trans in range(int(I0 + 0.5)):
                trans_fields = np.array(H0_array_out[np.where(trans_array==trans)])
                trans_probs = np.array(prob_array[np.where(trans_array==trans)])

                # make histograms
                hist,bin_edges = np.histogram(trans_fields,
                                              bins=int(nbins),
                                              range=(min_field,max_field),
                                              weights=trans_probs
                                             )                       
                bin_edges=bin_edges[:-1]
                
                # convolve with df and dvQ*trans as necessary
                if trans==0:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_T,
                                                                mode=broadening_func)
                else:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_T,
                                                                mode=broadening_func)
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=conv_hist_intensity,
                                                                FWHM=FWHM_dvQ_T*float(trans),
                                                                mode=broadening_func)
                conv_hist_intensity_sum = conv_hist_intensity_sum + conv_hist_intensity
            convolved_histogram = np.column_stack((bin_edges,conv_hist_intensity_sum))

        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_pp.txt'
                out_filename_hist = out_filename + '_sim_pp_hist.txt'
                out_filename_hist_conv = out_filename + '_sim_pp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_pp.txt'
                out_filename_hist = 'sim_pp_hist.txt'
                out_filename_hist_conv = 'sim_pp_hist_conv.txt'
            np.savetxt(out_filename_hist_conv, convolved_histogram)

        return convolved_histogram


    # 2nd order pert functions:
    # following J. F. Baugher, et al., J. Chem. Phys. 50, 4914 (1969). https://doi.org/10.1063/1.1670988
    # and P. C. Taylor, et al., Chem. Rev. 75, 203 (1975). https://doi.org/10.1021/cr60294a003
    def R(self,vQ,I0):
        R = vQ**2*(I0*(I0 + 1) - 3./4)
        return R


    def A(self,eta,phi):
        A_var = -27/8 - 9/4*eta*np.cos(2*phi) - 3/8*eta**2*np.cos(2*phi)**2
        return A_var


    def B(self,eta,phi):
        B_var = 30/8 - eta**2/2 + 2*eta*np.cos(2*phi) + 3/4*eta**2*np.cos(2*phi)**2
        return B_var


    def C(self,eta,phi):
        C_var = -3/8 + eta**2/3 + eta/4*np.cos(2*phi) - 3/8*eta**2*np.cos(2*phi)**2
        return C_var


    def prob(self, m, I0):
        pulse_NMR_probability = abs(I0*(I0 + 1) - m*(m - 1))
        return pulse_NMR_probability


    def I0_one_half_freq(
        self,
        gamma,
        H0,
        Ka,
        Kb,
        Kc,
        theta_ar,
        phi_ar
        ):
        """
        I=1/2 vectorized friendly for numpy arrays, function that returns 
        an array of resonance frequencies given the input parameters. Input parameters with _ar after 
        them are meant to be numpy arrays for calculation of powder patterns. The parameter meanings 
        are detailed here:
        I0 = spin of the nucleus to by simulated (only tested half integer spins so far)
        gamma = gyromagnetic ratio
        H0 = external applied field
        Ka,b,c = Diagonal elements of the shift tensor
        theta_ar = array of polar angles theta in radians. if randomly choosing values, they shoud be 
                   generated by first choosing random values of cosine(theta) and then taking the arccos, eg 
                   rand_costheta_array = np.random.uniform(-1,1,size=nrands)
                   rand_theta_array = np.arccos(rand_costheta_array)
        phi_ar = array of azimuthal angles phi in radians from 0 -- 2pi, eg
                 (rand_phi_array = np.random.uniform(0,np.pi*2,size=nrands))
        """
        Ka=Ka/100.0
        Kb=Kb/100.0
        Kc=Kc/100.0

        # Reference for following: P. C. Taylor, J. F. Baugher, and H. M. Kriz, Chem. Rev. 75, 203 (1975).
        # eqn 28
        freq = gamma*H0*(((1 + Ka)**2*np.sin(theta_ar)**2*np.sin(phi_ar)**2
                         + (1 + Kb)**2*np.sin(theta_ar)**2*np.cos(phi_ar)**2
                         + (1 + Kc)**2*np.cos(theta_ar)**2)**0.5)
        # or equivalently:
    #     Kiso = 1./3*(Ka + Kb + Kc)
    #     Kani = 0.5*(Kb - Ka)
    #     Kax = 1./6*(2*Kc - Ka - Kb)
    #     freq = gamma*H0*(1 + Kiso + Kax*(3*np.cos(theta_ar)**2 - 1)
    #                      + Kani*np.sin(theta_ar)**2*np.cos(2*phi_ar))
        return freq


    def sec_ord_freq(self,
                     I0,
                     gamma,
                     H0,
                     Ka,
                     Kb,
                     Kc,
                     vQ,
                     eta,
                     m_ar,
                     theta_ar,
                     phi_ar):
        """
        Second order perturbation theory, vectorized friendly for numpy arrays, function that returns 
        an array of resonance frequencies given the input parameters. Input parameters with _ar after 
        them are meant to be numpy arrays for calculation of powder patterns. The parameter meanings 
        are detailed here:
        I0 = spin of the nucleus to by simulated (only tested half integer spins so far)
        gamma = gyromagnetic ratio
        H0 = external applied field
        Ka,b,c = Diagonal elements of the shift tensor [ahoy! NOTE: That Bauger stipulates K3>K2>K1 (Kc>Kb>Ka) and therefore I may need to re-derive these equations for the general case, OR just stipulate this in the code for other users...]
        vQ = principle component of the EFG tensor (not the actual NQR freq, should probably be called vc)
             the function was calculated by Baugher et al. assuming the shift and EFG tensors principle axes
             are coincident
        eta = asymmetry parameter of the EFG tensor

        the following parameters are expected to be 1D arrays all with the same length (could be length 1):
        m_ar = array of m (nuclear spin state index) values from -I0+1 to I0. generated by the powder 
               pattern simulation function
        theta_ar = array of polar angles theta in radians. if randomly choosing values, they shoud be 
                   generated by first choosing random values of cosine(theta) and then taking the arccos, eg 
                   rand_costheta_array = np.random.uniform(-1,1,size=nrands)
                   rand_theta_array = np.arccos(rand_costheta_array)
        phi_ar = array of azimuthal angles phi in radians from 0 -- 2pi, eg
                 (rand_phi_array = np.random.uniform(0,np.pi*2,size=nrands))
        """
        print('inside sec_ord_freq')
        print('input parameters:')
        print('I0 =', I0)
        print('gamma =', gamma)
        print('H0 =', H0)
        print('Ka =', Ka)
        print('Kb =', Kb)
        print('Kc =', Kc)
        print('vQ =', vQ)
        print('eta =', eta)
        print('m_ar =', m_ar)
        print('theta_ar =', theta_ar)
        print('phi_ar =', phi_ar)
        Kiso = 1./3*(Ka/100 + Kb/100 + Kc/100)
        print('Kiso =', Kiso)
        Kani = 1./2*(Kb/100 - Ka/100)
        print('Kani =', Kani)
        Kax = 1./6*(2*Kc/100 - Ka/100 - Kb/100)
        print('Kax =', Kax)
        cent_ar = np.where(m_ar==0.5, 1, 0)
        print('cent_ar =', cent_ar)
        print('self.A(eta, phi_ar) =', self.A(eta, phi_ar))
        print('self.B(eta, phi_ar) =', self.B(eta, phi_ar))
        print('self.C(eta, phi_ar) =', self.C(eta, phi_ar))
        freq_ar = (gamma*H0*((1 + Kiso + Kax*(3*np.cos(theta_ar)**2 - 1)
                              + Kani*np.sin(theta_ar)**2*np.cos(2*phi_ar)))
                            - (cent_ar*self.R(vQ, I0)/(6*gamma*H0)*(self.A(eta, phi_ar)*np.cos(theta_ar)**4 
                                                                 + self.B(eta, phi_ar)*np.cos(theta_ar)**2 
                                                                 + self.C(eta, phi_ar)))
                   - ((2*m_ar - 1.0)*vQ/4*(((3*np.cos(theta_ar)**2 - 1))
                    - eta*np.sin(theta_ar)**2*np.cos(2*phi_ar)))
                  )

        return freq_ar


    def convolveGaussLor(self,
                         x_array,
                         y_array,
                         FWHM,
                         mode='gauss'):
        """
        Performs a convolution of the input y_array with a normalized gaussian or lorentzian function. The x_array 
        is expected to be the frequency in MHz/field in Tesla (independent variable) data, and the y_array is 
        expected to be the NMR response (dependent variable) data. The FWHM is converted properly below and is expected
        to be in the corresponding dependent variable units (MHz or T). The mode keyword argument can be either 'gauss' or
        'lor' strings, and if anything else is passed here the function just returns the input y_array data.
        """
        if mode=='gauss':
            mu = np.mean(x_array)
            sigma = FWHM/(2*(2*np.log(2))**0.5)
            gaussian_array = 1/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((x_array - mu)/sigma)**2)
            sum_norm = np.sum(gaussian_array)
            if sum_norm == 0.0:
                print('there was a problem normalizing the covolution, increasing nbins may help')
                return y_array
            else:
                gaussian_array = gaussian_array/sum_norm
                gauss_convolved_spectrum = np.convolve(y_array, gaussian_array, mode='same')
                return gauss_convolved_spectrum
        elif mode == 'lor':
            x0 = np.mean(x_array)
            Gamma = FWHM
            lorentzian_array = (1/np.pi)*((0.5*Gamma)/((x_array - x0)**2 + (0.5*Gamma)**2))
            sum_norm = np.sum(lorentzian_array)
            if sum_norm==0.0:
                print('there was a problem normalizing the convolution, increasing nbins may help')
                return y_array
            else:
                lorentzian_array = lorentzian_array/sum_norm
                lor_convolved_spectrum = np.convolve(y_array, lorentzian_array, mode='same')
                return lor_convolved_spectrum
        else:
            print("invalid mode; please select mode='gauss' or 'lor'")
            return y_array
        

    def sec_ord_freq_spec(self,
                          I0,
                          gamma,
                          H0,
                          Ka,
                          Kb,
                          Kc,
                          vQ,
                          eta,
                          theta_array,
                          phi_array,
                          nbins=1000,
                          min_freq=None,
                          max_freq=None,
                          broadening_func='gauss',
                          FWHM_MHz=0.01,
                          save_files_bool=False,
                          out_filename='',
                          baseline=0.25):
        print('inside sec_ord_freq_spec')
        gamma = abs(gamma)
        if I0==0.5: #ahoy! this should be boolean...
            #print("Running I=1/2 calc...")
            freq_array=self.I0_one_half_freq(
                                             gamma=gamma,
                                             H0=H0,
                                             Ka=Ka,
                                             Kb=Kb,
                                             Kc=Kc,
                                             theta_ar=theta_array,
                                             phi_ar=phi_array
                                            )

            if min_freq==None or max_freq==None:
                min_range = freq_array.min() - baseline
                max_range = freq_array.max() + baseline
            else:
                min_range = min_freq
                max_range = max_freq
            hist, bin_edges = np.histogram(freq_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range)
                                          )
        else:
            #print("Running second order calc...")
            dim = int((I0 + 1/2)*2) #ahoy! check this int algebra to make sure it works properly...
            print('dim =', dim)
            m_values = np.array([I0-i for i in range(dim-1)])
            print('m_values =', m_values)

            freq_array = self.sec_ord_freq(                
                                           I0=I0,
                                           gamma=gamma,
                                           H0=H0,
                                           Ka=Ka,
                                           Kb=Kb,
                                           Kc=Kc,
                                           vQ=vQ,
                                           eta=eta,
                                           m_ar=m_values,
                                           theta_ar=theta_array,
                                           phi_ar=phi_array
                                          )
            print('freq_array', freq_array)
            prob_array = self.prob(m_values, I0)
            print('prob_array', prob_array)
            if min_freq==None or max_freq==None:
                min_range = freq_array.min() - baseline
                max_range = freq_array.max() + baseline
            else:
                min_range = min_freq
                max_range = max_freq

            hist, bin_edges = np.histogram(freq_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range),
                                           weights=prob_array
                                          )
        bin_edges = bin_edges[:-1]
        histogram = np.column_stack((bin_edges,hist))
        convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                    y_array=hist,
                                                    FWHM=FWHM_MHz,
                                                    mode=broadening_func)
        #ahoy! need to implement quadrupolar broadening of the satellites
        convolved_histogram = np.column_stack((bin_edges,convolved_hist_intensity))
        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_ppp.txt'
                out_filename_hist = out_filename + '_sim_ppp_hist.txt'
                out_filename_hist_conv = out_filename + '_sim_ppp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_ppp.txt'
                out_filename_hist = 'sim_ppp_hist.txt'
                out_filename_hist_conv = 'sim_ppp_hist_conv.txt'
            np.savetxt(out_filename_hist_conv, convolved_histogram)

        return convolved_histogram


    def I0_one_half_field(self,
                          gamma,
                          f0,
                          Ka,
                          Kb,
                          Kc,
                          theta_ar,
                          phi_ar):
        """
        I=1/2 vectorized friendly for numpy arrays, function that returns 
        an array of resonance frequencies given the input parameters. Input parameters with _ar after 
        them are meant to be numpy arrays for calculation of powder patterns. The parameter meanings 
        are detailed here:
        I0 = spin of the nucleus to by simulated (only tested half integer spins so far)
        gamma = gyromagnetic ratio
        H0 = external applied field
        Ka,b,c = Diagonal elements of the shift tensor
        theta_ar = array of polar angles theta in radians. if randomly choosing values, they shoud be 
                   generated by first choosing random values of cosine(theta) and then taking the arccos, eg 
                   rand_costheta_array = np.random.uniform(-1,1,size=nrands)
                   rand_theta_array = np.arccos(rand_costheta_array)
        phi_ar = array of azimuthal angles phi in radians from 0 -- 2pi, eg
                 (rand_phi_array = np.random.uniform(0,np.pi*2,size=nrands))
        """
        Ka=Ka/100.0
        Kb=Kb/100.0
        Kc=Kc/100.0
        field = f0/gamma*(((1 + Ka)**2*np.sin(theta_ar)**2*np.sin(phi_ar)**2
                          + (1 + Kb)**2*np.sin(theta_ar)**2*np.cos(phi_ar)**2
                          + (1 + Kc)**2*np.cos(theta_ar)**2)**-0.5)
        # or equivalently:
    #     Kiso = 1./3*(Ka + Kb + Kc)
    #     Kani = 0.5*(Kb - Ka)
    #     Kax = 1./6*(2*Kc - Ka - Kb)
    #     field = f0/(gamma*(1 + Kiso + Kax*(3*np.cos(theta_ar)**2 - 1)
    #                        + Kani*np.sin(theta_ar)**2*np.cos(2*phi_ar)))
        return field


    def sec_ord_field(self,
                      I0,
                      gamma,
                      f0,
                      Ka,
                      Kb,
                      Kc,
                      vQ,
                      eta,
                      m_ar,
                      theta_ar,
                      phi_ar):
        """
        Ahoy! There may be a problem here because the shift tensor principle components are defined such that K3>K2>K1, may need to re-solve these equations for the general case, where the shift tensor elements do not need to be ordered and are instead assigned to specific crystalline axes!

        This function was calculated by solving for H0 (now called Hres) in the function sec_ord_freq. It is also
        vectorized for use with numpy arrays. See docstring for sec_ord_freq for details on variables
        The solution used below is one of two given by the quadratic equation. It seems to be correct,
        but both solutions are given here in case there is some problem with my algebra:
        Solve[f0 == C1*H0 + C2/H0 + C3, H0]
        H0 = (f0 - C3 - (-4*C1*C2 + C3**2 - 2*C3*f0 + f0**2)**0.5)/(2*C1)
        H0 = (f0 - C3 + (-4*C1*C2 + C3**2 - 2*C3*f0 + f0**2)**0.5)/(2*C1)
        Need to double check this solution for the case when the terms are zero...
        """
        Kiso = 1./3*(Ka/100 + Kb/100 + Kc/100)
        Kani = 1./2*(Kb/100 - Ka/100)
        Kax = 1./6*(2*Kc/100 - Ka/100 - Kb/100)
        cent_ar = np.where(m_ar==0.5,1.0,0.0)

        #f0 = C1*H0 + C2/H0 + C3
        C1 = gamma*(1 + Kiso + Kax*(3*np.cos(theta_ar)**2 - 1) + Kani*np.sin(theta_ar)**2*np.cos(2*phi_ar)) #ahoy! In Brauger et al. 1969 there seems to be a mistake in equation 20, where the Kax*(3*np.cos(theta_ar)**2 - 1) was written as Kax*(6*np.cos(theta_ar)**2 - 1). This seems to be the source of the factor of two shift error that was showing up. Should double check their math!
        C2 = -cent_ar*self.R(vQ,I0)/(6*gamma)*(self.A(eta,phi_ar)*np.cos(theta_ar)**4 + self.B(eta,phi_ar)*np.cos(theta_ar)**2 + self.C(eta,phi_ar))
        C3 = -(m_ar - 0.5)*vQ*(0.5*(3*np.cos(theta_ar)**2 - 1) - 0.5*eta*np.sin(theta_ar)**2*np.cos(2*phi_ar))
        Hres = (f0 - C3 + (-4*C1*C2 + C3**2 - 2*C3*f0 + f0**2)**0.5)/(2*C1)

        return Hres


    def sec_ord_field_spec(self,
                           I0,
                           gamma,
                           f0,
                           Ka,
                           Kb,
                           Kc,
                           vQ,
                           eta,
                           theta_array,
                           phi_array,
                           nbins=1000,
                           min_field=None,
                           max_field=None,
                           broadening_func='gauss',
                           FWHM_T=0.01,
                           save_files_bool=False,
                           out_filename='',
                           baseline=0.25):
        if I0==0.5: #ahoy! this should be boolean...
            #print("Running I=1/2 calc...")
            field_array=self.I0_one_half_field(gamma=gamma,
                                    f0=f0,
                                    Ka=Ka,
                                    Kb=Kb,
                                    Kc=Kc,
                                    theta_ar=rtheta_array,
                                    phi_ar=phi_array
                                    )
            if min_field==None or max_field==None:
                min_range=field_array.min()-baseline
                max_range=field_array.max()+baseline
            else:
                min_range=min_field
                max_range=max_field
            hist, bin_edges = np.histogram(field_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range)
                                          )
        else:
            #print("Running second order calc...")
            dim = int((I0 + 1/2)*2) #ahoy! check this int algebra to make sure it works properly...
            m_values = np.array([I0-i for i in range(dim-1)])

            field_array = self.sec_ord_field(I0=I0,
                                            gamma=gamma,
                                            f0=f0,
                                            Ka=Ka,
                                            Kb=Kb,
                                            Kc=Kc,
                                            vQ=vQ,
                                            eta=eta,
                                            m_ar=m_values,
                                            theta_ar=theta_array,
                                            phi_ar=phi_array)
            prob_array = self.prob(m_values,I0)

            if min_field==None or max_field==None:
                min_range=field_array.min()-baseline
                max_range=field_array.max()+baseline
            else:
                min_range=min_field
                max_range=max_field

            hist, bin_edges = np.histogram(field_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range),
                                           weights=prob_array)
        bin_edges=bin_edges[:-1]
        histogram = np.column_stack((bin_edges, hist))
        convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                         y_array=hist,
                                                         FWHM=FWHM_T,
                                                         mode=broadening_func)
        #ahoy! need to implement quadrupolar broadening of the satellites
        convolved_histogram = np.column_stack((bin_edges,convolved_hist_intensity))
        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_ppp.txt'
                out_filename_hist = out_filename + '_sim_ppp_hist.txt'
                out_filename_hist_conv = out_filename + '_sim_ppp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_ppp.txt'
                out_filename_hist = 'sim_ppp_hist.txt'
                out_filename_hist_conv = 'sim_ppp_hist_conv.txt'
            np.savetxt(out_filename_hist_conv, convolved_histogram)

        return convolved_histogram


    def sec_ord_freq_pp(self,
                        H0,
                        Ka,
                        Kb,
                        Kc,
                        vQ,
                        eta,
                        nrands,
                        nbins=1000,
                        min_freq=None,
                        max_freq=None,
                        broadening_func='gauss',
                        FWHM_MHz=0.02,
                        FWHM_dvQ_MHz=0.001,
                        save_files_bool=False,
                        out_filename='',
                        baseline=1.0):
        I0 = self.I0
        gamma = self.isotope_data_dict[self.isotope]["gamma"]
        
        if FWHM_MHz < 1.0e-6:
            FWHM_MHz=1.0e-6
        
        if FWHM_dvQ_MHz < 1.0e-6:
            FWHM_dvQ_MHz=1.0e-6

        nrands = int(nrands)
        rand_phi_array = np.random.uniform(0,np.pi*2,size=nrands)
        rand_costheta_array = np.random.uniform(-1,1,size=nrands)
        rand_theta_array = np.arccos(rand_costheta_array)

        if I0==0.5: #ahoy! this should be boolean...
            #print("Running I0 = 1/2...")
            freq_array = self.I0_one_half_freq(
                                    gamma=gamma,
                                    H0=H0,
                                    Ka=Ka,
                                    Kb=Kb,
                                    Kc=Kc,
                                    theta_ar=rand_theta_array,
                                    phi_ar=rand_phi_array
                                    )

            if min_freq==None:
                freq_array.min()-baseline
            if max_freq==None:
                freq_array.max()+baseline

            hist, bin_edges = np.histogram(freq_array,
                                           bins=int(nbins),
                                           range=(min_freq,max_freq)
                                          )

            bin_edges=bin_edges[:-1]
            # convolve with df and dvQ*trans as necessary
            convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                             y_array=hist,
                                                             FWHM=FWHM_MHz,
                                                             mode=broadening_func
                                                            )
            convolved_histogram = np.column_stack((bin_edges,convolved_hist_intensity))
        else:
            #print("Running second order...")
            dim = int((I0+1/2)*2)
            m_values = np.array([I0-i for i in range(dim-1)])
            rand_m_array = np.random.choice(m_values,size=nrands)
            freq_array = self.sec_ord_freq(                
                            I0=I0,
                            gamma=gamma,
                            H0=H0,
                            Ka=Ka,
                            Kb=Kb,
                            Kc=Kc,
                            vQ=vQ,
                            eta=eta,
                            m_ar=rand_m_array,
                            theta_ar=rand_theta_array,
                            phi_ar=rand_phi_array
                            )

            if min_freq==None:
                freq_array.min()-baseline
            if max_freq==None:
                freq_array.max()+baseline

            prob_array = self.prob(rand_m_array,I0)
            trans_array = np.abs(rand_m_array-0.5)
            #print('trans_array:')
            #print(trans_array)
            conv_hist_intensity_sum = np.zeros(int(nbins))
            for trans in range(int(I0 + 0.5)):
                trans_freqs = np.array(freq_array[np.where(trans_array==trans)])
                trans_probs = np.array(prob_array[np.where(trans_array==trans)])
                #print('trans_prob:')
                #print(trans_prob)
                # make histograms
                hist,bin_edges = np.histogram(trans_freqs,
                                              bins=int(nbins),
                                              range=(min_freq,max_freq),
                                              weights=trans_probs
                                             )                       
                bin_edges=bin_edges[:-1]
                #print('raw hist:')
                #plt.plot(bin_edges,hist)
                #plt.show()
                
                # convolve with df and dvQ*trans as necessary
                if trans==0:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_MHz,
                                                                mode=broadening_func
                                                               )
                    #print('conv hist central:')
                    #plt.plot(bin_edges,conv_hist_intensity)
                    #plt.show()
                else:
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=hist,
                                                                FWHM=FWHM_MHz,
                                                                mode=broadening_func
                                                               )
                    #print('conv hist sats magnetic:')
                    #plt.plot(bin_edges,conv_hist_intensity)
                    #plt.show()
                    conv_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                                y_array=conv_hist_intensity,
                                                                FWHM=FWHM_dvQ_MHz*float(trans),
                                                                mode=broadening_func
                                                               )
                    #print('conv hist sats quadrupolar:')
                    #plt.plot(bin_edges,conv_hist_intensity)
                    #plt.show()
                conv_hist_intensity_sum = conv_hist_intensity_sum + conv_hist_intensity
            #print('out hist:')
            #plt.plot(bin_edges,conv_hist_intensity_sum)
            #plt.show()
            convolved_histogram = np.column_stack((bin_edges,conv_hist_intensity_sum))
                
        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_ppp.txt'
                out_filename_hist_conv = out_filename + '_sim_ppp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_ppp.txt'
                out_filename_hist_conv = 'sim_ppp_hist_conv.txt'

            #np.savetxt(out_filename_spec, freqs_probs)
            np.savetxt(out_filename_hist_conv, convolved_histogram)

        return convolved_histogram
    

    def sec_ord_field_pp(self,
                         I0,
                         gamma,
                         f0,
                         Ka,
                         Kb,
                         Kc,
                         vQ,
                         eta,
                         nrands,
                         nbins=1000,
                         min_field=0.1,
                         max_field=16.0,
                         broadening_func='gauss',
                         FWHM_T=0.001,
                         save_files_bool=False,
                         out_filename='',
                         baseline=0.25,
                         rand_theta_array=None,
                         rand_phi_array=None):
        nrands = int(nrands)

        if rand_theta_array==None:
            rand_costheta_array = np.random.uniform(-1,1,size=nrands)
            rand_theta_array = np.arccos(rand_costheta_array)

        if rand_phi_array==None:
            rand_phi_array = np.random.uniform(0,np.pi*2,size=nrands)

        if I0==0.5: #ahoy! this should be boolean...
            #print("Running I=1/2 calc...")
            field_array=self.I0_one_half_field(gamma=gamma,
                                    f0=f0,
                                    Ka=Ka,
                                    Kb=Kb,
                                    Kc=Kc,
                                    theta_ar=rand_theta_array,
                                    phi_ar=rand_phi_array
                                    )
            if min_field==None or max_field==None:
                min_range=field_array.min()-baseline
                max_range=field_array.max()+baseline
            else:
                min_range=min_field
                max_range=max_field
            hist, bin_edges = np.histogram(field_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range)
                                          )
        else:
            #print("Running second order calc...")
            dim = int((I0+1/2)*2)
            m_values = np.array([I0-i for i in range(dim-1)])
            rand_m_array = np.random.choice(m_values,size=nrands)
            field_array = self.sec_ord_field(                
                            I0=I0,
                            gamma=gamma,
                            f0=f0,
                            Ka=Ka,
                            Kb=Kb,
                            Kc=Kc,
                            vQ=vQ,
                            eta=eta,
                            m_ar=rand_m_array,
                            theta_ar=rand_theta_array,
                            phi_ar=rand_phi_array
                            )

            prob_array = self.prob(rand_m_array,I0)

            if min_field==None or max_field==None:
                min_range=field_array.min()-baseline
                max_range=field_array.max()+baseline
            else:
                min_range=min_field
                max_range=max_field

            hist, bin_edges = np.histogram(field_array,
                                           bins=int(nbins),
                                           range=(min_range, max_range),
                                           weights=prob_array
                                          )
        bin_edges=bin_edges[:-1]
        histogram = np.column_stack((bin_edges,hist))
        convolved_hist_intensity = self.convolveGaussLor(x_array=bin_edges,
                                                    y_array=hist,
                                                    FWHM=FWHM_T,
                                                    mode=broadening_func)
        convolved_histogram = np.column_stack((bin_edges,convolved_hist_intensity))
        if save_files_bool==True:
            if out_filename!='':
                out_filename_spec = out_filename + '_sim_ppp.txt'
                out_filename_hist = out_filename + '_sim_ppp_hist.txt'
                out_filename_hist_conv = out_filename + '_sim_ppp_hist_conv.txt'
            else:
                out_filename_spec = 'sim_ppp.txt'
                out_filename_hist = 'sim_ppp_hist.txt'
                out_filename_hist_conv = 'sim_ppp_hist_conv.txt'

    #         np.savetxt(out_filename_spec, freqs_probs)
    #         np.savetxt(out_filename_hist, histogram)
            np.savetxt(out_filename_hist_conv, convolved_histogram)

    #     plt.plot(histogram[:,0],histogram[:,1])
    #     plt.plot(convolved_histogram[:,0],convolved_histogram[:,1])
    #     plt.show()

        return convolved_histogram
