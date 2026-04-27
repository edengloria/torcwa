from collections import OrderedDict
import weakref
import warnings
import torch
from .torch_eig import Eig
from .v2.linalg import diag_post_multiply, lu_factor_left, lu_solve_left, solve_left, solve_left_many, solve_right

pi = 3.141592652589793

class rcwa:
    _material_conv_cache = OrderedDict()
    _material_conv_cache_max = 32
    _material_conv_cache_policy = {}

    @classmethod
    def clear_material_cache(cls):
        cls._material_conv_cache.clear()
        cls._material_conv_cache_policy.clear()

    @classmethod
    def register_material_cache_policy(cls,material,*,cache_key=None,cache=True):
        if not torch.is_tensor(material):
            return
        try:
            material_ref = weakref.ref(material)
        except TypeError:
            return
        cls._material_conv_cache_policy[id(material)] = (material_ref,cache_key,bool(cache))

    @classmethod
    def unregister_material_cache_policy(cls,material):
        if torch.is_tensor(material):
            cls._material_conv_cache_policy.pop(id(material),None)

    # Simulation setting
    def __init__(self,freq,order,L,*,
            dtype=torch.complex64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            stable_eig_grad=True,
            avoid_Pinv_instability=False,
            max_Pinv_instability=0.005
        ):

        '''
            Rigorous Coupled Wave Analysis
            - Lorentz-Heaviside units
            - Speed of light: 1
            - Time harmonics notation: exp(-jωt)

            Parameters
            - freq: simulation frequency (unit: length^-1)
            - order: Fourier order [x_order (int), y_order (int)]
            - L: Lattice constant [Lx, Ly] (unit: length)

            Keyword Parameters
            - dtype: simulation data type (only torch.complex64 and torch.complex128 are allowed.)
            - device: simulation device (only torch.device('cpu') and torch.device('cuda') are allowed.)
            - stable_eig_grad: stabilize gradient calculation of eigendecompsition (default as True)
            - avoid_Pinv_instability: avoid instability of P inverse (P: H to E) (default as False)
            - max_Pinv_instability: allowed maximum instability value for P inverse (default as 0.005 if avoid_Pinv_instability is True)
        '''

        # Hardware
        if dtype != torch.complex64 and dtype != torch.complex128:
            warnings.warn('Invalid simulation data type. Set as torch.complex64.',UserWarning)
            self._dtype = torch.complex64
        else:
            self._dtype = dtype
        self._device = device

        # Stabilize the gradient of eigendecomposition
        self.stable_eig_grad = True if stable_eig_grad else False
        self.memory_mode = 'balanced'
        self.store_fields = True

        # Stability setting for inverse matrix of P and Q
        if avoid_Pinv_instability is True:
            self.avoid_Pinv_instability = True
            self.max_Pinv_instability = max_Pinv_instability
            self.Pinv_instability = []
            self.Qinv_instability = []
        else:
            self.avoid_Pinv_instability = False
            self.max_Pinv_instability = None
            self.Pinv_instability = None
            self.Qinv_instability = None

        # Simulation parameters
        self.freq = torch.as_tensor(freq,dtype=self._dtype,device=self._device) # unit^-1
        self.omega = 2*pi*freq # same as k0a
        self.L = torch.as_tensor(L,dtype=self._dtype,device=self._device)

        # Fourier order
        self.order = order
        self.order_x = torch.linspace(-self.order[0],self.order[0],2*self.order[0]+1,dtype=torch.int64,device=self._device)
        self.order_y = torch.linspace(-self.order[1],self.order[1],2*self.order[1]+1,dtype=torch.int64,device=self._device)
        self.order_N = len(self.order_x)*len(self.order_y)

        # Lattice vector
        self.L = L  # unit
        self.Gx_norm, self.Gy_norm = 1/(L[0]*self.freq), 1/(L[1]*self.freq)

        # Input and output layer (Default: free space)
        self.eps_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_in = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.eps_out = torch.tensor(1.,dtype=self._dtype,device=self._device)
        self.mu_out = torch.tensor(1.,dtype=self._dtype,device=self._device)

        # Internal layers
        self.layer_N = 0  # total number of layers
        self.thickness = []
        self.eps_conv, self.mu_conv = [], []

        # Internal layer eigenmodes
        self.P, self.Q = [], []
        self.kz_norm, self.E_eigvec, self.H_eigvec = [], [], []

        # Internal layer mode coupling coefficiencts
        self.Cf, self.Cb = [], []

        # Single layer scattering matrices
        self.layer_S11, self.layer_S21, self.layer_S12, self.layer_S22 = [], [], [], []

    def add_input_layer(self,eps=1.,mu=1.):
        '''
            Add input layer
            - If this function is not used, simulation will be performed under free space input layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_in = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_in = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sin = []

    def add_output_layer(self,eps=1.,mu=1.):
        '''
            Add output layer
            - If this function is not used, simulation will be performed under free space output layer.

            Parameters
            - eps: relative permittivity
            - mu: relative permeability
        '''

        self.eps_out = torch.as_tensor(eps,dtype=self._dtype,device=self._device)
        self.mu_out = torch.as_tensor(mu,dtype=self._dtype,device=self._device)
        self.Sout = []

    def set_incident_angle(self,inc_ang,azi_ang,angle_layer='input'):
        '''
            Set incident angle

            Parameters
            - inc_ang: incident angle (unit: radian)
            - azi_ang: azimuthal angle (unit: radian)
            - angle_layer: reference layer to calculate angle ('i', 'in', 'input' / 'o', 'out', 'output')
        '''

        self.inc_ang = torch.as_tensor(inc_ang,dtype=self._dtype,device=self._device)
        self.azi_ang = torch.as_tensor(azi_ang,dtype=self._dtype,device=self._device)

        if angle_layer in ['i', 'in', 'input']:
            self.angle_layer = 'input'
        elif angle_layer in ['o', 'out', 'output']:
            self.angle_layer = 'output'
        else:
            warnings.warn('Invalid angle layer. Set as input layer.',UserWarning)
            self.angle_layer = 'input'

        self._kvectors()

    def add_layer(self,thickness,eps=1.,mu=1.):
        '''
            Add internal layer

            Parameters
            - thickness: layer thickness (unit: length)
            - eps: relative permittivity
            - mu: relative permeability
        '''

        is_eps_homogenous = (type(eps) == float) or (type(eps) == complex) or (eps.dim() == 0) or ((eps.dim() == 1) and eps.shape[0] == 1)
        is_mu_homogenous = (type(mu) == float) or (type(mu) == complex) or (mu.dim() == 0) or ((mu.dim() == 1) and mu.shape[0] == 1)
        
        self.eps_conv.append(eps*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_eps_homogenous else self._material_conv(eps))
        self.mu_conv.append(mu*torch.eye(self.order_N,dtype=self._dtype,device=self._device) if is_mu_homogenous else self._material_conv(mu))

        self.layer_N += 1
        self.thickness.append(thickness)

        if is_eps_homogenous and is_mu_homogenous:
            self._eigen_decomposition_homogenous(eps,mu)
        else:
            self._eigen_decomposition()

        self._solve_layer_smatrix()

    # Solve simulation
    def solve_global_smatrix(self):
        '''
            Solve global S-matrix
        '''

        # Initialization
        retain_fields = getattr(self,'store_fields',True)

        if self.layer_N > 0:
            S11 = self.layer_S11[0]
            S21 = self.layer_S21[0]
            S12 = self.layer_S12[0]
            S22 = self.layer_S22[0]
            C = [[self.Cf[0]], [self.Cb[0]]] if retain_fields else [[], []]
        else:
            S11 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            S21 = torch.zeros([2*self.order_N,2*self.order_N],dtype=self._dtype,device=self._device)
            S12 = torch.zeros([2*self.order_N,2*self.order_N],dtype=self._dtype,device=self._device)
            S22 = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
            C = [[], []]

        # Connection
        for i in range(self.layer_N-1):
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.layer_S11[i+1], self.layer_S21[i+1], self.layer_S12[i+1], self.layer_S22[i+1]],
                Cm=C, Cn=[[self.Cf[i+1]], [self.Cb[i+1]]], retain_c=retain_fields)

        if hasattr(self,'Sin'):
            # input layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[self.Sin[0], self.Sin[1], self.Sin[2], self.Sin[3]],
                Sn=[S11, S21, S12, S22],
                Cm=[[],[]], Cn=C, retain_c=retain_fields)

        if hasattr(self,'Sout'):
            # output layer coupling
            [S11, S21, S12, S22], C = self._RS_prod(Sm=[S11, S21, S12, S22],
                Sn=[self.Sout[0], self.Sout[1], self.Sout[2], self.Sout[3]],
                Cm=C, Cn=[[],[]], retain_c=retain_fields)

        self.S = [S11, S21, S12, S22]
        self.C = C

    # Returns
    def diffraction_angle(self,orders,*,layer='output',unit='radian'):
        '''
            Diffraction angles for the selected orders

            Parameters
            - orders: selected diffraction orders (Recommended shape: Nx2)
            - layer: selected layer ('i', 'in', 'input' / 'o', 'out', 'output')
            - unit: unit of the output angles ('r', 'rad', 'radian' / 'd', 'deg', 'degree')

            Return
            - inclination angle (torch.Tensor), azimuthal angle (torch.Tensor)
        '''

        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if layer in ['i', 'in', 'input']:
            layer = 'input'
        elif layer in ['o', 'out', 'output']:
            layer = 'output'
        else:
            warnings.warn('Invalid layer. Set as output layer.',UserWarning)
            layer = 'output'

        if unit in ['r', 'rad', 'radian']:
            unit = 'radian'
        elif unit in ['d', 'deg', 'degree']:
            unit = 'degree'
        else:
            warnings.warn('Invalid unit. Set as radian.',UserWarning)
            unit = 'radian'

        # Matching indices
        order_indices = self._matching_indices(orders)

        eps = self.eps_in if layer == 'input' else self.eps_out
        mu = self.mu_in if layer == 'input' else self.mu_out

        kx_norm = self.Kx_norm_dn[order_indices]
        ky_norm = self.Ky_norm_dn[order_indices]
        Kt_norm_dn = torch.sqrt(kx_norm**2 + ky_norm**2)
        kz_norm = torch.sqrt(eps*mu - kx_norm**2 - ky_norm**2)
        inc_angle = torch.atan2(torch.real(Kt_norm_dn),torch.real(kz_norm))
        azi_angle = torch.atan2(torch.real(ky_norm),torch.real(kx_norm))

        if unit == 'degree':
            inc_angle = (180./pi) * inc_angle
            azi_angle = (180./pi) * azi_angle

        return inc_angle, azi_angle

    def return_layer(self,layer_num,nx=100,ny=100):
        '''
            Return spatial distributions of eps and mu for the selected layer.
            The eps and mu are recovered from the trucated Fourier orders.

            Parameters
            - layer_num: selected layer (int)
            - nx: x-direction grid number (int)
            - ny: y-direction grid number (int)

            Return
            - eps_recover (torch.Tensor), mu_recover (torch.Tensor)
        '''

        eps_fft = torch.zeros([nx,ny],dtype=self._dtype,device=self._device)
        mu_fft = torch.zeros([nx,ny],dtype=self._dtype,device=self._device)
        for i in range(-2*self.order[0],2*self.order[0]+1):
            for j in range(-2*self.order[1],2*self.order[1]+1):
                if i >= 0 and j >= 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][i*(2*self.order[1]+1)+j,0]
                    mu_fft[i,j] = self.mu_conv[layer_num][i*(2*self.order[1]+1)+j,0]
                elif i >= 0 and j < 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][i*(2*self.order[1]+1),-j]
                    mu_fft[i,j] = self.mu_conv[layer_num][i*(2*self.order[1]+1),-j]
                elif i < 0 and j >= 0:
                    eps_fft[i,j] = self.eps_conv[layer_num][j,-i*(2*self.order[1]+1)]
                    mu_fft[i,j] = self.mu_conv[layer_num][j,-i*(2*self.order[1]+1)]
                else:
                    eps_fft[i,j] = self.eps_conv[layer_num][0,-i*(2*self.order[1]+1)-j]
                    mu_fft[i,j] = self.mu_conv[layer_num][0,-i*(2*self.order[1]+1)-j]

        eps_recover = torch.fft.ifftn(eps_fft)*nx*ny
        mu_recover = torch.fft.ifftn(mu_fft)*nx*ny

        return eps_recover, mu_recover
    
    def S_parameters(self,orders,*,direction='forward',port='transmission',polarization='xx',ref_order=[0,0],power_norm=True,evanescent=1e-3,evanscent=None):
        '''
            Return S-parameters.

            Parameters
            - orders: selected orders (Recommended shape: Nx2)

            - direction: set the direction of light propagation ('f', 'forward' / 'b', 'backward')
            - port: set the direction of light propagation ('t', 'transmission' / 'r', 'reflection')
            - polarization: set the input and output polarization of light ((output,input) xy-pol: 'xx' / 'yx' / 'xy' / 'yy' , ps-pol: 'pp' / 'sp' / 'ps' / 'ss' )
            - ref_order: set the reference for calculating S-parameters (Recommended shape: Nx2)
            - power_norm: if set as True, the absolute square of S-parameters are corresponds to the ratio of power
            - evanescent: Criteria for judging the evanescent field. If power_norm=True and real(kz_norm)/imag(kz_norm) < evanescent, function returns 0 (default = 1e-3)
            - evanscent: Deprecated alias for evanescent.

            Return
            - S-parameters (torch.Tensor)
        '''

        if evanscent is not None:
            warnings.warn('Parameter "evanscent" is deprecated. Use "evanescent" instead.',DeprecationWarning)
            evanescent = evanscent

        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if direction in ['f', 'forward']:
            direction = 'forward'
        elif direction in ['b', 'backward']:
            direction = 'backward'
        else:
            warnings.warn('Invalid propagation direction. Set as forward.',UserWarning)
            direction = 'forward'

        if port in ['t', 'transmission']:
            port = 'transmission'
        elif port in ['r', 'reflection']:
            port = 'reflection'
        else:
            warnings.warn('Invalid port. Set as tramsmission.',UserWarning)
            port = 'transmission'

        if polarization not in ['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']:
            warnings.warn('Invalid polarization. Set as xx.',UserWarning)
            polarization = 'xx'

        ref_order = torch.as_tensor(ref_order,dtype=torch.int64,device=self._device).reshape([1,2])

        # Matching order indices
        order_indices = self._matching_indices(orders)
        ref_order_index = self._matching_indices(ref_order)

        if polarization in ['xx', 'yx', 'xy', 'yy']:
            # Matching order indices with polarization
            if polarization == 'yx' or polarization == 'yy':
                order_indices = order_indices + self.order_N
            if polarization == 'xy' or polarization == 'yy':
                ref_order_index = ref_order_index + self.order_N

            # power normalization factor
            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
                is_evanescent_in = torch.abs(torch.real(Kz_norm_dn_in_complex) / torch.imag(Kz_norm_dn_in_complex)) < evanescent
                Kz_norm_dn_in = torch.where(is_evanescent_in,torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),torch.real(Kz_norm_dn_in_complex))
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in,Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
                is_evanescent_out = torch.abs(torch.real(Kz_norm_dn_out_complex) / torch.imag(Kz_norm_dn_out_complex)) < evanescent
                Kz_norm_dn_out = torch.where(is_evanescent_out,torch.real(torch.zeros_like(Kz_norm_dn_out_complex)),torch.real(Kz_norm_dn_out_complex))
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out,Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack((torch.real(self.Kx_norm_dn),torch.real(self.Kx_norm_dn)))
                Ky_norm_dn = torch.hstack((torch.real(self.Ky_norm_dn),torch.real(self.Ky_norm_dn)))

                if polarization == 'xx':
                    numerator_pol, denominator_pol = Kx_norm_dn, Kx_norm_dn
                elif polarization == 'xy':
                    numerator_pol, denominator_pol = Kx_norm_dn, Ky_norm_dn
                elif polarization == 'yx':
                    numerator_pol, denominator_pol = Ky_norm_dn, Kx_norm_dn
                elif polarization == 'yy':
                    numerator_pol, denominator_pol = Ky_norm_dn, Ky_norm_dn

                if direction == 'forward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'forward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'backward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == 'backward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                normalization = torch.sqrt((1+(numerator_pol[order_indices]/numerator_kz[order_indices])**2)/(1+(denominator_pol[ref_order_index]/denominator_kz[ref_order_index])**2))
                normalization = normalization * torch.sqrt(numerator_kz[order_indices]/denominator_kz[ref_order_index])
            else:
                normalization = 1.

            # Get S-parameters
            if direction == 'forward' and port == 'transmission':
                S = self.S[0][order_indices,ref_order_index] * normalization
            elif direction == 'forward' and port == 'reflection':
                S = self.S[1][order_indices,ref_order_index] * normalization
            elif direction == 'backward' and port == 'reflection':
                S = self.S[2][order_indices,ref_order_index] * normalization
            elif direction == 'backward' and port == 'transmission':
                S = self.S[3][order_indices,ref_order_index] * normalization

            S = torch.where(torch.isinf(S),torch.zeros_like(S),S)
            S = torch.where(torch.isnan(S),torch.zeros_like(S),S)

            return S
        
        elif polarization in ['pp', 'sp', 'ps', 'ss']:
            if direction == 'forward' and port == 'transmission':
                idx = 0
                order_sign, ref_sign = 1, 1
                order_k0_norm2 = self.eps_out * self.mu_out
                ref_k0_norm2 = self.eps_in * self.mu_in
            elif direction == 'forward' and port == 'reflection':
                idx = 1
                order_sign, ref_sign = -1, 1
                order_k0_norm2 = self.eps_in * self.mu_in
                ref_k0_norm2 = self.eps_in * self.mu_in
            elif direction == 'backward' and port == 'reflection':
                idx = 2
                order_sign, ref_sign = 1, -1
                order_k0_norm2 = self.eps_out * self.mu_out
                ref_k0_norm2 = self.eps_out * self.mu_out
            elif direction == 'backward' and port == 'transmission':
                idx = 3
                order_sign, ref_sign = -1, -1
                order_k0_norm2 = self.eps_in * self.mu_in
                ref_k0_norm2 = self.eps_out * self.mu_out

            order_Kx_norm_dn = self.Kx_norm_dn[order_indices]
            order_Ky_norm_dn = self.Ky_norm_dn[order_indices]
            order_Kt_norm_dn = torch.sqrt(order_Kx_norm_dn**2 + order_Ky_norm_dn**2)
            order_Kz_norm_dn = order_sign*torch.abs(torch.real(torch.sqrt(order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2)))
            order_Kz_norm_dn_complex = torch.sqrt(order_k0_norm2 - order_Kx_norm_dn**2 - order_Ky_norm_dn**2)
            order_is_evanescent = torch.abs(torch.real(order_Kz_norm_dn_complex) / torch.imag(order_Kz_norm_dn_complex)) < evanescent

            order_inc_angle = torch.atan2(torch.real(order_Kt_norm_dn),order_Kz_norm_dn)
            order_azi_angle = torch.atan2(torch.real(order_Ky_norm_dn),torch.real(order_Kx_norm_dn))

            ref_Kx_norm_dn = self.Kx_norm_dn[ref_order_index]
            ref_Ky_norm_dn = self.Ky_norm_dn[ref_order_index]
            ref_Kt_norm_dn = torch.sqrt(ref_Kx_norm_dn**2 + ref_Ky_norm_dn**2)
            ref_Kz_norm_dn = ref_sign*torch.abs(torch.real(torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)))
            ref_Kz_norm_dn_complex = torch.sqrt(ref_k0_norm2 - ref_Kx_norm_dn**2 - ref_Ky_norm_dn**2)
            ref_is_evanescent = torch.abs(torch.real(ref_Kz_norm_dn_complex) / torch.imag(ref_Kz_norm_dn_complex)) < evanescent

            ref_inc_angle = torch.atan2(torch.real(ref_Kt_norm_dn),ref_Kz_norm_dn)
            ref_azi_angle = torch.atan2(torch.real(ref_Ky_norm_dn),torch.real(ref_Kx_norm_dn))

            xx = self.S[idx][order_indices,ref_order_index]
            xy = self.S[idx][order_indices,ref_order_index+self.order_N]
            yx = self.S[idx][order_indices+self.order_N,ref_order_index]
            yy = self.S[idx][order_indices+self.order_N,ref_order_index+self.order_N]

            xx = torch.where(order_is_evanescent,torch.zeros_like(xx),xx)
            xy = torch.where(order_is_evanescent,torch.zeros_like(xy),xy)
            yx = torch.where(order_is_evanescent,torch.zeros_like(yx),yx)
            yy = torch.where(order_is_evanescent,torch.zeros_like(yy),yy)

            if ref_is_evanescent:
                S = torch.zeros_like(xx)
                return S

            if polarization == 'pp':
                S = torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * xx +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * yx +\
                    torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * xy +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * yy
            elif polarization == 'ps':
                S = torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * (-1)*torch.sin(ref_azi_angle) * xx +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * (-1)*torch.sin(ref_azi_angle) * yx +\
                    torch.cos(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_azi_angle) * xy +\
                    torch.sin(order_azi_angle)/torch.cos(order_inc_angle) * torch.cos(ref_azi_angle) * yy
            elif polarization == 'sp':
                S = -torch.sin(order_azi_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * xx +\
                    torch.cos(order_azi_angle) * torch.cos(ref_inc_angle)*torch.cos(ref_azi_angle) * yx +\
                    -torch.sin(order_azi_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * xy +\
                    torch.cos(order_azi_angle) * torch.cos(ref_inc_angle)*torch.sin(ref_azi_angle) * yy
            elif polarization == 'ss':
                S = -torch.sin(order_azi_angle) * (-1)*torch.sin(ref_azi_angle) * xx +\
                    torch.cos(order_azi_angle) * (-1)*torch.sin(ref_azi_angle) * yx +\
                    -torch.sin(order_azi_angle) * torch.cos(ref_azi_angle) * xy +\
                    torch.cos(order_azi_angle) * torch.cos(ref_azi_angle) * yy

            if power_norm:
                Kz_norm_dn_in_complex = torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
                is_evanescent_in = torch.abs(torch.real(Kz_norm_dn_in_complex) / torch.imag(Kz_norm_dn_in_complex)) < evanescent
                Kz_norm_dn_in = torch.where(is_evanescent_in,torch.real(torch.zeros_like(Kz_norm_dn_in_complex)),torch.real(Kz_norm_dn_in_complex))
                Kz_norm_dn_in = torch.hstack((Kz_norm_dn_in,Kz_norm_dn_in))

                Kz_norm_dn_out_complex = torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
                is_evanescent_out = torch.abs(torch.real(Kz_norm_dn_out_complex) / torch.imag(Kz_norm_dn_out_complex)) < evanescent
                Kz_norm_dn_out = torch.where(is_evanescent_out,torch.abs(torch.real(Kz_norm_dn_out_complex)),torch.real(Kz_norm_dn_out_complex))
                Kz_norm_dn_out = torch.hstack((Kz_norm_dn_out,Kz_norm_dn_out))

                Kx_norm_dn = torch.hstack((torch.real(self.Kx_norm_dn),torch.real(self.Kx_norm_dn)))
                Ky_norm_dn = torch.hstack((torch.real(self.Ky_norm_dn),torch.real(self.Ky_norm_dn)))

                if direction == 'forward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'forward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_in
                elif direction == 'backward' and port == 'reflection':
                    numerator_kz = Kz_norm_dn_out
                    denominator_kz = Kz_norm_dn_out
                elif direction == 'backward' and port == 'transmission':
                    numerator_kz = Kz_norm_dn_in
                    denominator_kz = Kz_norm_dn_out

                normalization = torch.sqrt(numerator_kz[order_indices]/denominator_kz[ref_order_index])
            else:
                normalization = 1.

            S = torch.where(torch.isinf(S),torch.zeros_like(S),S)
            S = torch.where(torch.isnan(S),torch.zeros_like(S),S)

            return S * normalization
        
        else:
            return None

    def source_planewave(self,*,amplitude=[1.,0.],direction='forward',notation='xy'):
        '''
            Generate planewave

            Paramters
            - amplitude: amplitudes at the matched diffraction orders ([Ex_amp, Ey_amp] for 'xy' notation, [Ep_amp, Es_amp] for 'ps' notation)
              (list / np.ndarray / torch.Tensor) (Recommended shape: 1x2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
            - notation: amplitude notation (xy-pol: 'xy' / ps-pol: 'ps')
        '''

        self.source_fourier(amplitude=amplitude,orders=[0,0],direction=direction,notation=notation)

    def source_fourier(self,*,amplitude,orders,direction='forward',notation='xy'):
        '''
            Generate Fourier source

            Paramters
            - amplitude: amplitudes at the matched diffraction orders [([Ex_amp, Ey_amp] at orders[0]), ..., ...]
                (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - orders: diffraction orders (list / np.ndarray / torch.Tensor) (Recommended shape: Nx2)
            - direction: incident direction ('f', 'forward' / 'b', 'backward')
            - notation: amplitude notation (xy-pol: 'xy' / ps-pol: 'ps')
        '''
        amplitude = torch.as_tensor(amplitude,dtype=self._dtype,device=self._device).reshape([-1,2])
        orders = torch.as_tensor(orders,dtype=torch.int64,device=self._device).reshape([-1,2])

        if direction in ['f', 'forward']:
            direction = 'forward'
        elif direction in ['b', 'backward']:
            direction = 'backward'
        else:
            warnings.warn('Invalid source direction. Set as forward.',UserWarning)
            direction = 'forward'

        if notation not in ['xy', 'ps']:
            warnings.warn('Invalid amplitude notation. Set as xy notation.',UserWarning)
            notation = 'xy'

        # Matching indices
        order_indices = self._matching_indices(orders)

        self.source_direction = direction

        E_i = torch.zeros([2*self.order_N,1],dtype=self._dtype,device=self._device)
        E_i[order_indices,0] = amplitude[:,0]
        E_i[order_indices+self.order_N,0] = amplitude[:,1]

        # Convert ps-pol to xy-pol
        if notation == 'ps':
            if direction == 'forward':
                eps, mu = self.eps_in, self.mu_in
                sign = 1
            else:
                eps, mu = self.eps_out, self.mu_out
                sign = -1
            
            Kt_norm_dn = torch.sqrt(self.Kx_norm_dn**2 + self.Ky_norm_dn**2)
            Kz_norm_dn = sign*torch.abs(torch.real(torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)))

            inc_angle = torch.atan2(torch.real(Kt_norm_dn),Kz_norm_dn)
            azi_angle = torch.atan2(torch.real(self.Ky_norm_dn),torch.real(self.Kx_norm_dn))

            p_amp = E_i[:self.order_N]
            s_amp = E_i[self.order_N:]
            E_x = (torch.cos(inc_angle)*torch.cos(azi_angle)).reshape([-1,1]) * p_amp - torch.sin(azi_angle).reshape([-1,1]) * s_amp
            E_y = (torch.cos(inc_angle)*torch.sin(azi_angle)).reshape([-1,1]) * p_amp + torch.cos(azi_angle).reshape([-1,1]) * s_amp
            E_i = torch.vstack((E_x,E_y)).to(self._dtype)

        self.E_i = E_i

    def field_xz(self,x_axis,z_axis,y):
        '''
            XZ-plane field distribution.
            Returns the field at the specific y point.

            Paramters
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - z_axis: z-direction sampling coordinates (torch.Tensor)
            - y: selected y point

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if not getattr(self,'store_fields',True):
            raise RuntimeError('Field reconstruction is unavailable because store_fields=False.')

        if type(x_axis) != torch.Tensor or type(z_axis) != torch.Tensor:
            warnings.warn('x and z axis must be torch.Tensor type. Return None.',UserWarning)
            return None

        return self._field_xz_yz('xz',x_axis.reshape([-1]),z_axis.reshape([-1]),y,chunk_size=getattr(self,'field_chunk_size',None))
    
    def field_yz(self,y_axis,z_axis,x):
        '''
            YZ-plane field distribution.
            Returns the field at the specific x point.

            Parameters
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_axis: z-direction sampling coordinates (torch.Tensor)
            - x: selected x point

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if not getattr(self,'store_fields',True):
            raise RuntimeError('Field reconstruction is unavailable because store_fields=False.')

        if type(y_axis) != torch.Tensor or type(z_axis) != torch.Tensor:
            warnings.warn('y and z axis must be torch.Tensor type. Return None.',UserWarning)
            return None

        return self._field_xz_yz('yz',y_axis.reshape([-1]),z_axis.reshape([-1]),x,chunk_size=getattr(self,'field_chunk_size',None))

    def _field_xz_yz(self,plane,axis,z_axis,fixed,chunk_size=None):
        axis = axis.to(device=self._device)
        z_axis = z_axis.to(device=self._device)
        fixed = torch.as_tensor(fixed,dtype=axis.dtype,device=self._device)

        layer_num,zp,zm = self._field_layer_numbers(z_axis)
        z_chunk = self._field_auto_chunk(len(z_axis),chunk_size)
        axis_chunk = self._field_auto_chunk(len(axis),chunk_size)

        electric = [torch.empty([len(axis),len(z_axis)],dtype=self._dtype,device=self._device) for _ in range(3)]
        magnetic = [torch.empty([len(axis),len(z_axis)],dtype=self._dtype,device=self._device) for _ in range(3)]

        for layer_value in torch.unique(layer_num):
            layer_id = int(layer_value.item())
            indices = torch.nonzero(layer_num == layer_id,as_tuple=False).reshape([-1])
            for start in range(0,len(indices),z_chunk):
                chunk_indices = indices[start:start+z_chunk]
                z_sel = z_axis[chunk_indices]
                z_prop = self._field_z_propagation(layer_id,z_sel,zp,zm)
                components = self._field_fourier_components(layer_id,z_prop)
                for axis_start in range(0,len(axis),axis_chunk):
                    axis_indices = slice(axis_start,axis_start+axis_chunk)
                    phase = self._field_transverse_phase(plane,axis[axis_indices],fixed)
                    for ci in range(3):
                        electric[ci][axis_indices,chunk_indices] = torch.matmul(phase,components[ci])
                        magnetic[ci][axis_indices,chunk_indices] = torch.matmul(phase,components[ci+3])

        return electric, magnetic

    def _field_layer_numbers(self,z_axis):
        zp = torch.zeros(len(self.thickness),device=self._device)
        zm = torch.zeros(len(self.thickness),device=self._device)
        layer_num = torch.zeros([len(z_axis)],dtype=torch.int64,device=self._device)
        layer_num[z_axis<0.] = -1

        for ti in range(len(self.thickness)):
            zp[ti:] += self.thickness[ti]
        zm[1:] = zp[0:-1]

        for bi in range(len(zp)):
            layer_num[z_axis>zp[bi]] += 1

        return layer_num,zp,zm

    def _field_z_propagation(self,layer_id,z_axis,zp,zm):
        if layer_id == -1:
            return torch.minimum(z_axis,torch.zeros_like(z_axis))
        if layer_id == self.layer_N:
            if len(zp) == 0:
                return z_axis
            return torch.maximum(z_axis-zp[-1],torch.zeros_like(z_axis))
        return z_axis - zm[layer_id]

    def _field_transverse_phase(self,plane,axis,fixed):
        axis = axis.reshape([-1,1])
        if plane == 'xz':
            phase_arg = self.Kx_norm_dn.reshape([1,-1])*axis + self.Ky_norm_dn.reshape([1,-1])*fixed
        elif plane == 'yz':
            phase_arg = self.Kx_norm_dn.reshape([1,-1])*fixed + self.Ky_norm_dn.reshape([1,-1])*axis
        else:
            raise ValueError("plane must be 'xz' or 'yz'")
        return torch.exp(1.j*self.omega*phase_arg)

    def _field_fourier_components(self,layer_id,z_prop):
        z_prop = z_prop.reshape([1,-1])

        if layer_id == -1 or layer_id == self.layer_N:
            Kx_norm_dn = self.Kx_norm_dn
            Ky_norm_dn = self.Ky_norm_dn

            if layer_id == -1:
                eps = self.eps_in if hasattr(self,'eps_in') else 1.
                mu = self.mu_in if hasattr(self,'mu_in') else 1.
                V = self.__dict__.get('_Vi',self._Vf)
                Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)>0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])
            else:
                eps = self.eps_out if hasattr(self,'eps_in') else 1.
                mu = self.mu_out if hasattr(self,'mu_in') else 1.
                V = self.__dict__.get('_Vo',self._Vf)
                Kz_norm_dn = torch.sqrt(eps*mu - Kx_norm_dn**2 - Ky_norm_dn**2)
                Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn).reshape([-1,1])

            Kz_norm_dn = torch.vstack((Kz_norm_dn,Kz_norm_dn))
            z_phase = torch.exp(1.j*self.omega*Kz_norm_dn*z_prop)

            if layer_id == -1 and self.source_direction == 'forward':
                Exy_p = self.E_i*z_phase
                Hxy_p = self._homogeneous_matmul(V,Exy_p)
                Exy_m = torch.matmul(self.S[1],self.E_i)*torch.conj(z_phase)
                Hxy_m = -self._homogeneous_matmul(V,Exy_m)
            elif layer_id == -1 and self.source_direction == 'backward':
                Exy_p = torch.zeros([2*self.order_N,z_phase.shape[-1]],dtype=self._dtype,device=self._device)
                Hxy_p = torch.zeros_like(Exy_p)
                Exy_m = torch.matmul(self.S[3],self.E_i)*torch.conj(z_phase)
                Hxy_m = -self._homogeneous_matmul(V,Exy_m)
            elif layer_id == self.layer_N and self.source_direction == 'forward':
                Exy_p = torch.matmul(self.S[0],self.E_i)*z_phase
                Hxy_p = self._homogeneous_matmul(V,Exy_p)
                Exy_m = torch.zeros_like(Exy_p)
                Hxy_m = torch.zeros_like(Exy_p)
            elif layer_id == self.layer_N and self.source_direction == 'backward':
                Exy_p = torch.matmul(self.S[2],self.E_i)*z_phase
                Hxy_p = self._homogeneous_matmul(V,Exy_p)
                Exy_m = self.E_i*torch.conj(z_phase)
                Hxy_m = -self._homogeneous_matmul(V,Exy_m)
            else:
                raise RuntimeError('Invalid field source direction.')

            Ex_mn = Exy_p[:self.order_N] + Exy_m[:self.order_N]
            Ey_mn = Exy_p[self.order_N:] + Exy_m[self.order_N:]
            Hz_mn = self._dn_pre_multiply(self.Kx_norm_dn,Ey_mn)/mu - self._dn_pre_multiply(self.Ky_norm_dn,Ex_mn)/mu
            Hx_mn = Hxy_p[:self.order_N] + Hxy_m[:self.order_N]
            Hy_mn = Hxy_p[self.order_N:] + Hxy_m[self.order_N:]
            Ez_mn = self._dn_pre_multiply(self.Ky_norm_dn,Hx_mn)/eps - self._dn_pre_multiply(self.Kx_norm_dn,Hy_mn)/eps

            return [Ex_mn,Ey_mn,Ez_mn,Hx_mn,Hy_mn,Hz_mn]

        if self.source_direction == 'forward':
            C = torch.matmul(self.C[0][layer_id],self.E_i)
        elif self.source_direction == 'backward':
            C = torch.matmul(self.C[1][layer_id],self.E_i)
        else:
            raise RuntimeError('Invalid field source direction.')

        kz_norm = self.kz_norm[layer_id]
        E_eigvec = self.E_eigvec[layer_id]
        H_eigvec = self.H_eigvec[layer_id]
        Cp = C[:2*self.order_N,0]
        Cm = C[2*self.order_N:,0]

        z_phase_p = torch.exp(1.j*self.omega*kz_norm.reshape([-1,1])*z_prop)
        z_phase_m = torch.exp(1.j*self.omega*kz_norm.reshape([-1,1])*(self.thickness[layer_id]-z_prop))

        Exy_p = E_eigvec.unsqueeze(-1)*z_phase_p.unsqueeze(0)
        Ex_p = Exy_p[:self.order_N,:,:]
        Ey_p = Exy_p[self.order_N:,:,:]
        Hz_p_rhs = self._dn_pre_multiply(self.Kx_norm_dn,Ey_p.reshape([self.order_N,-1])) - self._dn_pre_multiply(self.Ky_norm_dn,Ex_p.reshape([self.order_N,-1]))

        Exy_m = E_eigvec.unsqueeze(-1)*z_phase_m.unsqueeze(0)
        Ex_m = Exy_m[:self.order_N,:,:]
        Ey_m = Exy_m[self.order_N:,:,:]
        Hz_m_rhs = self._dn_pre_multiply(self.Kx_norm_dn,Ey_m.reshape([self.order_N,-1])) - self._dn_pre_multiply(self.Ky_norm_dn,Ex_m.reshape([self.order_N,-1]))
        Hz_p_flat, Hz_m_flat = self._solve_left_many_policy(self.mu_conv[layer_id],[Hz_p_rhs,Hz_m_rhs])
        Hz_p = Hz_p_flat.reshape_as(Ex_p)
        Hz_m = Hz_m_flat.reshape_as(Ex_m)

        Hxy_p = H_eigvec.unsqueeze(-1)*z_phase_p.unsqueeze(0)
        Hx_p = Hxy_p[:self.order_N,:,:]
        Hy_p = Hxy_p[self.order_N:,:,:]
        Ez_p_rhs = self._dn_pre_multiply(self.Ky_norm_dn,Hx_p.reshape([self.order_N,-1])) - self._dn_pre_multiply(self.Kx_norm_dn,Hy_p.reshape([self.order_N,-1]))

        Hxy_m = -H_eigvec.unsqueeze(-1)*z_phase_m.unsqueeze(0)
        Hx_m = Hxy_m[:self.order_N,:,:]
        Hy_m = Hxy_m[self.order_N:,:,:]
        Ez_m_rhs = self._dn_pre_multiply(self.Ky_norm_dn,Hx_m.reshape([self.order_N,-1])) - self._dn_pre_multiply(self.Kx_norm_dn,Hy_m.reshape([self.order_N,-1]))
        Ez_p_flat, Ez_m_flat = self._solve_left_many_policy(self.eps_conv[layer_id],[Ez_p_rhs,Ez_m_rhs])
        Ez_p = Ez_p_flat.reshape_as(Hx_p)
        Ez_m = Ez_m_flat.reshape_as(Hx_m)

        Cp = Cp.reshape([1,-1,1])
        Cm = Cm.reshape([1,-1,1])
        Ex_mn = torch.sum(Ex_p*Cp + Ex_m*Cm,dim=1)
        Ey_mn = torch.sum(Ey_p*Cp + Ey_m*Cm,dim=1)
        Ez_mn = torch.sum(Ez_p*Cp + Ez_m*Cm,dim=1)
        Hx_mn = torch.sum(Hx_p*Cp + Hx_m*Cm,dim=1)
        Hy_mn = torch.sum(Hy_p*Cp + Hy_m*Cm,dim=1)
        Hz_mn = torch.sum(Hz_p*Cp + Hz_m*Cm,dim=1)

        return [Ex_mn,Ey_mn,Ez_mn,Hx_mn,Hy_mn,Hz_mn]

    def _field_xy_from_components(self,x_axis,y_axis,components,chunk_size=None):
        x_chunk = self._field_auto_chunk(len(x_axis),chunk_size)
        y_chunk = self._field_auto_chunk(len(y_axis),chunk_size)
        fields = [torch.empty([len(x_axis),len(y_axis)],dtype=self._dtype,device=self._device) for _ in range(6)]

        for x_start in range(0,len(x_axis),x_chunk):
            x_sel = x_axis[x_start:x_start+x_chunk].reshape([-1,1,1])
            for y_start in range(0,len(y_axis),y_chunk):
                y_sel = y_axis[y_start:y_start+y_chunk].reshape([1,-1,1])
                xy_phase = torch.exp(1.j*self.omega*(self.Kx_norm_dn.reshape([1,1,-1])*x_sel + self.Ky_norm_dn.reshape([1,1,-1])*y_sel))
                x_slice = slice(x_start,x_start+len(x_sel))
                y_slice = slice(y_start,y_start+len(y_sel.reshape([-1])))
                for ci,component in enumerate(components):
                    fields[ci][x_slice,y_slice] = torch.sum(component.reshape([1,1,-1])*xy_phase,dim=2)

        return fields[:3], fields[3:]

    def field_xy(self,layer_num,x_axis,y_axis,z_prop=0.):
        '''
            XY-plane field distribution at the selected layer.
            Returns the field at z_prop away from the lower boundary of the layer.
            For the input layer, z_prop is the distance from the upper boundary and should be negative (calculate z_prop=0 if positive value is entered).

            Parameters
            - layer_num: selected layer (int)
            - x_axis: x-direction sampling coordinates (torch.Tensor)
            - y_axis: y-direction sampling coordinates (torch.Tensor)
            - z_prop: z-direction distance from the lower boundary of the layer (layer_num>-1),
                or the distance from the upper boundary of the layer and should be negative (layer_num=-1).

            Return
            - [Ex, Ey, Ez] (list[torch.Tensor]), [Hx, Hy, Hz] (list[torch.Tensor])
        '''

        if not getattr(self,'store_fields',True):
            raise RuntimeError('Field reconstruction is unavailable because store_fields=False.')

        if type(layer_num) != int:
            warnings.warn('Parameter "layer_num" must be int type. Return None.',UserWarning)
            return None

        if layer_num < -1 or layer_num > self.layer_N:
            warnings.warn('Layer number is out of range. Return None.',UserWarning)
            return None

        if type(x_axis) != torch.Tensor or type(y_axis) != torch.Tensor:
            warnings.warn('x and y axis must be torch.Tensor type. Return None.',UserWarning)
            return None

        real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
        x_axis = x_axis.to(device=self._device,dtype=real_dtype).reshape([-1])
        y_axis = y_axis.to(device=self._device,dtype=real_dtype).reshape([-1])
        if layer_num == -1:
            z_prop = z_prop if z_prop <= 0. else 0.
        elif layer_num == self.layer_N:
            z_prop = z_prop if z_prop >= 0. else 0.
        z_axis = torch.as_tensor([z_prop],dtype=real_dtype,device=self._device)
        components = [component[:,0] for component in self._field_fourier_components(layer_num,z_axis)]

        return self._field_xy_from_components(x_axis,y_axis,components,chunk_size=getattr(self,'field_chunk_size',None))

    # Internal functions
    @property
    def Kx_norm(self):
        return torch.diag(self.Kx_norm_dn)

    @property
    def Ky_norm(self):
        return torch.diag(self.Ky_norm_dn)

    @property
    def Vf(self):
        if '_Vf' not in self.__dict__:
            raise AttributeError('Vf has not been initialized')
        return self._homogeneous_dense(self.__dict__['_Vf'])

    @property
    def Vi(self):
        if '_Vi' not in self.__dict__:
            raise AttributeError('Vi has not been initialized')
        return self._homogeneous_dense(self.__dict__['_Vi'])

    @property
    def Vo(self):
        if '_Vo' not in self.__dict__:
            raise AttributeError('Vo has not been initialized')
        return self._homogeneous_dense(self.__dict__['_Vo'])

    def _dn_pre_multiply(self,diagonal,tensor):
        shape = [diagonal.shape[0]] + [1]*(tensor.dim()-1)
        return diagonal.reshape(shape) * tensor

    def _solve_left_many_policy(self,A,rhs_list):
        if getattr(self,'memory_mode','balanced') == 'speed':
            return solve_left_many(A,rhs_list)
        return [solve_left(A,rhs) for rhs in rhs_list]

    def _field_auto_chunk(self,total,chunk_size=None):
        if chunk_size is not None:
            return max(1,int(chunk_size))
        mode = getattr(self,'memory_mode','balanced')
        if mode == 'speed':
            return max(1,int(total))
        if mode == 'memory':
            return max(1,min(int(total),16))
        return max(1,min(int(total),64))

    def _homogeneous_transform(self,kz_norm_dn):
        kx = self.Kx_norm_dn
        ky = self.Ky_norm_dn
        return (
            -ky*kx/kz_norm_dn,
            -kz_norm_dn - ky**2/kz_norm_dn,
            kz_norm_dn + kx**2/kz_norm_dn,
            kx*ky/kz_norm_dn,
        )

    def _homogeneous_add(self,left,right,alpha=1):
        return tuple(l + alpha*r for l,r in zip(left,right))

    def _homogeneous_dense(self,transform):
        a,b,c,d = transform
        return torch.hstack((torch.vstack((torch.diag(a),torch.diag(c))),torch.vstack((torch.diag(b),torch.diag(d)))))

    def _homogeneous_matmul(self,transform,matrix):
        a,b,c,d = transform
        top = matrix[:self.order_N]
        bottom = matrix[self.order_N:]
        return torch.vstack((a.reshape([-1,1])*top + b.reshape([-1,1])*bottom,
            c.reshape([-1,1])*top + d.reshape([-1,1])*bottom))

    def _homogeneous_solve(self,transform,rhs):
        a,b,c,d = transform
        top = rhs[:self.order_N]
        bottom = rhs[self.order_N:]
        det = a*d - b*c
        return torch.vstack(((d.reshape([-1,1])*top - b.reshape([-1,1])*bottom)/det.reshape([-1,1]),
            (-c.reshape([-1,1])*top + a.reshape([-1,1])*bottom)/det.reshape([-1,1])))

    def _homogeneous_solve_transform_dense(self,left,right):
        a,b,c,d = left
        e,f,g,h = right
        det = a*d - b*c
        result = (
            (d*e - b*g)/det,
            (d*f - b*h)/det,
            (-c*e + a*g)/det,
            (-c*f + a*h)/det,
        )
        return self._homogeneous_dense(result)

    def _matching_indices(self,orders):
        orders[orders[:,0]<-self.order[0],0] = int(-self.order[0])
        orders[orders[:,0]>self.order[0],0] = int(self.order[0])
        orders[orders[:,1]<-self.order[1],1] = int(-self.order[1])
        orders[orders[:,1]>self.order[1],1] = int(self.order[1])
        order_indices = len(self.order_y)*(orders[:,0]+int(self.order[0])) + orders[:,1]+int(self.order[1])

        return order_indices

    def _kvectors(self):
        if self.angle_layer == 'input':
            self.kx0_norm = torch.real(torch.sqrt(self.eps_in*self.mu_in)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(self.eps_in*self.mu_in)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)
        else:
            self.kx0_norm = torch.real(torch.sqrt(self.eps_out*self.mu_out)) * torch.sin(self.inc_ang) * torch.cos(self.azi_ang)
            self.ky0_norm = torch.real(torch.sqrt(self.eps_out*self.mu_out)) * torch.sin(self.inc_ang) * torch.sin(self.azi_ang)

        # Free space k-vectors and E to H transformation matrix
        self.kx_norm = self.kx0_norm + self.order_x * self.Gx_norm
        self.ky_norm = self.ky0_norm + self.order_y * self.Gy_norm

        kx_norm_grid, ky_norm_grid = torch.meshgrid(self.kx_norm,self.ky_norm,indexing='ij')

        self.Kx_norm_dn = torch.reshape(kx_norm_grid,(-1,))
        self.Ky_norm_dn = torch.reshape(ky_norm_grid,(-1,))

        Kz_norm_dn = torch.sqrt(1. - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
        self._Vf = self._homogeneous_transform(Kz_norm_dn)

        if hasattr(self,'Sin'):
            # Input layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_in*self.mu_in - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            self._Vi = self._homogeneous_transform(Kz_norm_dn)

            Vtmp1 = self._homogeneous_add(self._Vf,self._Vi)
            Vtmp2 = self._homogeneous_add(self._Vf,self._Vi,alpha=-1)
            Vtmp1_Vi = self._homogeneous_solve_transform_dense(Vtmp1,self._Vi)
            Vtmp1_Vtmp2 = self._homogeneous_solve_transform_dense(Vtmp1,Vtmp2)
            Vtmp1_Vf = self._homogeneous_solve_transform_dense(Vtmp1,self._Vf)

            # Input layer S-matrix
            self.Sin.append(2*Vtmp1_Vi)      # Tf S11
            self.Sin.append(-Vtmp1_Vtmp2)    # Rf S21
            self.Sin.append(Vtmp1_Vtmp2)     # Rb S12
            self.Sin.append(2*Vtmp1_Vf)      # Tb S22

        if hasattr(self,'Sout'):
            # Output layer k-vectors and E to H transformation matrix
            Kz_norm_dn = torch.sqrt(self.eps_out*self.mu_out - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
            Kz_norm_dn = torch.where(torch.imag(Kz_norm_dn)<0,torch.conj(Kz_norm_dn),Kz_norm_dn)
            self._Vo = self._homogeneous_transform(Kz_norm_dn)

            Vtmp1 = self._homogeneous_add(self._Vf,self._Vo)
            Vtmp2 = self._homogeneous_add(self._Vf,self._Vo,alpha=-1)
            Vtmp1_Vf = self._homogeneous_solve_transform_dense(Vtmp1,self._Vf)
            Vtmp1_Vtmp2 = self._homogeneous_solve_transform_dense(Vtmp1,Vtmp2)
            Vtmp1_Vo = self._homogeneous_solve_transform_dense(Vtmp1,self._Vo)

            # Output layer S-matrix
            self.Sout.append(2*Vtmp1_Vf)      # Tf S11
            self.Sout.append(Vtmp1_Vtmp2)     # Rf S21
            self.Sout.append(-Vtmp1_Vtmp2)    # Rb S12
            self.Sout.append(2*Vtmp1_Vo)      # Tb S22

    def _material_conv(self,material):
        cache_key, cache_ref = self._material_conv_cache_key(material)
        if cache_key is not None:
            cached = self._material_conv_cache.get(cache_key)
            if cached is not None:
                cached_ref, cached_value = cached
                if cached_ref() is not None:
                    self._material_conv_cache.move_to_end(cache_key)
                    return cached_value
                del self._material_conv_cache[cache_key]

        material_N = material.shape[0]*material.shape[1]

        # Matching indices
        order_x_grid, order_y_grid = torch.meshgrid(self.order_x,self.order_y,indexing='ij')
        ox = order_x_grid.to(torch.int64).reshape([-1])
        oy = order_y_grid.to(torch.int64).reshape([-1])

        ind = torch.arange(len(self.order_x)*len(self.order_y),device=self._device)
        indx, indy = torch.meshgrid(ind.to(torch.int64),ind.to(torch.int64),indexing='ij')

        material_fft = torch.fft.fft2(material)/material_N

        material_fft_real = torch.real(material_fft)
        material_fft_imag = torch.imag(material_fft)
        
        material_convmat_real = (material_fft_real[ox[indx]-ox[indy],oy[indx]-oy[indy]])
        material_convmat_imag = (material_fft_imag[ox[indx]-ox[indy],oy[indx]-oy[indy]])

        material_convmat = torch.complex(material_convmat_real,material_convmat_imag)

        if cache_key is not None:
            self._material_conv_cache[cache_key] = (cache_ref,material_convmat)
            self._material_conv_cache.move_to_end(cache_key)
            while len(self._material_conv_cache) > self._material_conv_cache_max:
                self._material_conv_cache.popitem(last=False)
        
        return material_convmat

    def _material_conv_cache_key(self,material):
        if not torch.is_tensor(material) or material.requires_grad:
            return None,None
        user_cache_key = None
        policy = self._material_conv_cache_policy.get(id(material))
        if policy is not None:
            material_ref,policy_cache_key,cache_enabled = policy
            if material_ref() is material:
                if not cache_enabled:
                    return None,None
                user_cache_key = policy_cache_key
            else:
                del self._material_conv_cache_policy[id(material)]
        try:
            material_ref = weakref.ref(material)
        except TypeError:
            return None,None
        if user_cache_key is not None:
            try:
                hash(user_cache_key)
            except TypeError:
                user_cache_key = repr(user_cache_key)
        is_conj = material.is_conj() if hasattr(material,'is_conj') else False
        is_neg = material.is_neg() if hasattr(material,'is_neg') else False
        key = (
            user_cache_key,
            id(material),
            material.data_ptr(),
            getattr(material,'_version',0),
            tuple(material.shape),
            tuple(material.stride()),
            int(material.storage_offset()),
            str(material.dtype),
            str(material.device),
            material.device.index if material.device.type != 'cpu' else None,
            str(material.layout),
            bool(is_conj),
            bool(is_neg),
            bool(material.requires_grad),
            tuple(int(v) for v in self.order),
            len(self.order_x),
            len(self.order_y),
        )
        return key,material_ref
    
    def _eigen_decomposition_homogenous(self,eps,mu):
        kx = self.Kx_norm_dn
        ky = self.Ky_norm_dn
        eye = torch.eye(self.order_N,dtype=self._dtype,device=self._device)

        # H to E transformation matirx
        P11 = torch.diag(kx*ky/eps)
        P12 = mu*eye - torch.diag(kx*kx/eps)
        P21 = -mu*eye + torch.diag(ky*ky/eps)
        P22 = -torch.diag(ky*kx/eps)
        self.P.append(torch.hstack((torch.vstack((P11,P21)),torch.vstack((P12,P22)))))

        # E to H transformation matrix
        Q11 = -torch.diag(kx*ky/mu)
        Q12 = -eps*eye + torch.diag(kx*kx/mu)
        Q21 = eps*eye - torch.diag(ky*ky/mu)
        Q22 = torch.diag(ky*kx/mu)
        self.Q.append(torch.hstack((torch.vstack((Q11,Q21)),torch.vstack((Q12,Q22)))))
        
        E_eigvec = torch.eye(self.P[-1].shape[-1],dtype=self._dtype,device=self._device)
        kz_norm = torch.sqrt(eps*mu - self.Kx_norm_dn**2 - self.Ky_norm_dn**2)
        kz_norm = torch.where(torch.imag(kz_norm)<0,torch.conj(kz_norm),kz_norm) # Normalized kz for positive mode
        kz_norm = torch.cat((kz_norm,kz_norm))

        self.kz_norm.append(kz_norm) 
        self.E_eigvec.append(E_eigvec)

    def _eigen_decomposition(self):
        Kx_norm = torch.diag(self.Kx_norm_dn)
        Ky_norm = torch.diag(self.Ky_norm_dn)

        # H to E transformation matirx
        P_tmp = solve_right(self.eps_conv[-1], torch.vstack((Kx_norm,Ky_norm)))
        P_tmp_x = P_tmp[:self.order_N,:]
        P_tmp_y = P_tmp[self.order_N:,:]
        self.P.append(torch.hstack((torch.vstack((torch.zeros_like(self.mu_conv[-1]),-self.mu_conv[-1])),
            torch.vstack((self.mu_conv[-1],torch.zeros_like(self.mu_conv[-1]))))) +
            torch.hstack((torch.vstack((diag_post_multiply(P_tmp_x,self.Ky_norm_dn),diag_post_multiply(P_tmp_y,self.Ky_norm_dn))),
                torch.vstack((-diag_post_multiply(P_tmp_x,self.Kx_norm_dn),-diag_post_multiply(P_tmp_y,self.Kx_norm_dn))))))

        # E to H transformation matrix
        Q_tmp = solve_right(self.mu_conv[-1], torch.vstack((Kx_norm,Ky_norm)))
        Q_tmp_x = Q_tmp[:self.order_N,:]
        Q_tmp_y = Q_tmp[self.order_N:,:]
        self.Q.append(torch.hstack((torch.vstack((torch.zeros_like(self.eps_conv[-1]),self.eps_conv[-1])),
            torch.vstack((-self.eps_conv[-1],torch.zeros_like(self.eps_conv[-1]))))) +
            torch.hstack((torch.vstack((-diag_post_multiply(Q_tmp_x,self.Ky_norm_dn),-diag_post_multiply(Q_tmp_y,self.Ky_norm_dn))),
                torch.vstack((diag_post_multiply(Q_tmp_x,self.Kx_norm_dn),diag_post_multiply(Q_tmp_y,self.Kx_norm_dn))))))
        
        # Eigen-decomposition
        if self.stable_eig_grad is True:
            kz_norm, E_eigvec = Eig.apply(torch.matmul(self.P[-1],self.Q[-1]))
        else:
            kz_norm, E_eigvec = torch.linalg.eig(torch.matmul(self.P[-1],self.Q[-1]))
        
        kz_norm = torch.sqrt(kz_norm)
        self.kz_norm.append(torch.where(torch.imag(kz_norm)<0,-kz_norm,kz_norm)) # Normalized kz for positive mode
        self.E_eigvec.append(E_eigvec)

    def _solve_layer_smatrix(self):
        phase = torch.exp(1.j*self.omega*self.kz_norm[-1]*self.thickness[-1])
        E_kz = diag_post_multiply(self.E_eigvec[-1],self.kz_norm[-1])
        P_lu, P_pivots = lu_factor_left(self.P[-1])

        if self.avoid_Pinv_instability == True:
            eye = torch.eye(self.P[-1].shape[-1],dtype=self._dtype,device=self._device)
            Q_lu, Q_pivots = lu_factor_left(self.Q[-1])
            Pinv_tmp = lu_solve_left(P_lu,P_pivots,eye)
            Qinv_tmp = lu_solve_left(Q_lu,Q_pivots,eye)
            
            Pinv_ins_tmp1 = torch.max(torch.abs( torch.matmul(self.P[-1].detach(),Pinv_tmp.detach())-eye.to(self.P[-1]) ))
            Pinv_ins_tmp2 = torch.max(torch.abs( torch.matmul(Pinv_tmp.detach(),self.P[-1].detach())-eye.to(self.P[-1]) ))
            Qinv_ins_tmp1 = torch.max(torch.abs( torch.matmul(self.Q[-1].detach(),Qinv_tmp.detach())-eye.to(self.Q[-1]) ))
            Qinv_ins_tmp2 = torch.max(torch.abs( torch.matmul(Qinv_tmp.detach(),self.Q[-1].detach())-eye.to(self.Q[-1]) ))

            self.Pinv_instability.append(torch.maximum(Pinv_ins_tmp1,Pinv_ins_tmp2))
            self.Qinv_instability.append(torch.maximum(Qinv_ins_tmp1,Qinv_ins_tmp2))

            if self.Pinv_instability[-1] < self.max_Pinv_instability:
                self.H_eigvec.append(lu_solve_left(P_lu,P_pivots,E_kz))
            else:
                self.H_eigvec.append(torch.matmul(self.Q[-1],diag_post_multiply(self.E_eigvec[-1],1/self.kz_norm[-1])))
        else:
            self.H_eigvec.append(lu_solve_left(P_lu,P_pivots,E_kz))

        Vf_inv_H = self._homogeneous_solve(self._Vf,self.H_eigvec[-1])
        E_plus = self.E_eigvec[-1] + Vf_inv_H
        E_minus = self.E_eigvec[-1] - Vf_inv_H
        E_minus_phase = diag_post_multiply(E_minus,phase)
        E_phase = diag_post_multiply(self.E_eigvec[-1],phase)

        # Mode coupling coefficients.  The coupling matrix has the exact
        # block-symmetric form [[A, B], [B, A]], so diagonalizing that block
        # structure avoids the larger 4M x 4M solve in balanced/memory modes.
        eye = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        if getattr(self,'memory_mode','balanced') == 'speed':
            Ctmp1 = torch.vstack((E_plus, E_minus_phase))
            Ctmp2 = torch.vstack((E_minus_phase, E_plus))
            Ctmp = torch.hstack((Ctmp1,Ctmp2))
            Cf_rhs = torch.vstack((2*eye,torch.zeros_like(eye)))
            Cb_rhs = torch.vstack((torch.zeros_like(eye),2*eye))
            Cf, Cb = solve_left_many(Ctmp,[Cf_rhs,Cb_rhs])
        else:
            U = solve_left(E_plus + E_minus_phase,eye)
            V = solve_left(E_plus - E_minus_phase,eye)
            Cf = torch.vstack((U + V,U - V))
            Cb = torch.vstack((U - V,U + V))
        self.Cf.append(Cf)
        self.Cb.append(Cb)

        self.layer_S11.append(torch.matmul(E_phase, self.Cf[-1][:2*self.order_N,:]) + torch.matmul(self.E_eigvec[-1],self.Cf[-1][2*self.order_N:,:]))
        self.layer_S21.append(torch.matmul(self.E_eigvec[-1], self.Cf[-1][:2*self.order_N,:]) + torch.matmul(E_phase,self.Cf[-1][2*self.order_N:,:])
            - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))
        self.layer_S12.append(torch.matmul(E_phase, self.Cb[-1][:2*self.order_N,:]) + torch.matmul(self.E_eigvec[-1],self.Cb[-1][2*self.order_N:,:])
            - torch.eye(2*self.order_N,dtype=self._dtype,device=self._device))
        self.layer_S22.append(torch.matmul(self.E_eigvec[-1], self.Cb[-1][:2*self.order_N,:]) + torch.matmul(E_phase,self.Cb[-1][2*self.order_N:,:]))

    def _RS_prod(self,Sm,Sn,Cm,Cn,retain_c=True):
        # S11 = S[0] / S21 = S[1] / S12 = S[2] / S22 = S[3]
        # Cf = C[0] / Cb = C[1]

        eye = torch.eye(2*self.order_N,dtype=self._dtype,device=self._device)
        tmp1_matrix = eye - torch.matmul(Sm[2],Sn[1])
        tmp2_matrix = eye - torch.matmul(Sn[1],Sm[2])
        tmp1_Sm0, tmp1_Sm2Sn3 = self._solve_left_many_policy(tmp1_matrix,[Sm[0],torch.matmul(Sm[2],Sn[3])])
        tmp2_Sn1Sm0, tmp2_Sn3 = self._solve_left_many_policy(tmp2_matrix,[torch.matmul(Sn[1],Sm[0]),Sn[3]])

        # Layer S-matrix
        S11 = torch.matmul(Sn[0],tmp1_Sm0)
        S21 = Sm[1] + torch.matmul(Sm[3],tmp2_Sn1Sm0)
        S12 = Sn[2] + torch.matmul(Sn[0],tmp1_Sm2Sn3)
        S22 = torch.matmul(Sm[3],tmp2_Sn3)

        if not retain_c:
            return [S11, S21, S12, S22], [[], []]

        # Mode coupling coefficients
        C = [[],[]]
        for m in range(len(Cm[0])):
            C[0].append(Cm[0][m] + torch.matmul(Cm[1][m],tmp2_Sn1Sm0))
            C[1].append(torch.matmul(Cm[1][m],tmp2_Sn3))

        for n in range(len(Cn[0])):
            C[0].append(torch.matmul(Cn[0][n],tmp1_Sm0))
            C[1].append(Cn[1][n] + torch.matmul(Cn[0][n],tmp1_Sm2Sn3))

        return [S11, S21, S12, S22], C
