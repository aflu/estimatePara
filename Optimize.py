from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import special
from scipy import interpolate
from scipy.integrate import quad
from scipy.special import erf
from scipy.interpolate import interp1d as interp1d
from scipy import optimize





# -------------------------------------------------------------------------
# BHE Data
# -------------------------------------------------------------------------

H = 100				# Length [m]
rb = 0.076			# Borehole radius [m]
lmS = 2.3			# thermal conductivty soil
cS = 2300000.		# pc soil
Rb = 0.088			# borehole resistance
Tungest = 13.		# undisturbed ground temperature
densF = 1045.		# density fluid
capF = 3800000.		# pc fluid



# -------------------------------------------------------------------------
# Measured Data
# -------------------------------------------------------------------------

with open('D:\\Benutzer\\Dueber\\V\\23_EONMPC\\16_ParameterOptimizer\\Data_Probe_3dt_30.0.txt', 'r') as file:
	file.readline() # skip the first line
	rows = [line.split('\t')[:] for line in file]
	cols = [list(col) for col in zip(*rows)]

T_in  = np.array([float(x) for x in cols[0]])			# [°C]
T_out = np.array([float(x) for x in cols[1]])			# [°C]
m_flow = np.array([float(x) for x in cols[2]])			# [kg/s]

groundload = (T_out-T_in)*(m_flow/densF)*capF/H			# [W/m]
groundload = np.nan_to_num(groundload)

# Calc Load increments
idxPlus = np.roll(np.linspace(0,groundload.size-1,groundload.size,dtype = 'int'),1)
delta_groundload = groundload - np.take(groundload,idxPlus)
delta_groundload[0] = groundload[0]

# Mean Fluid Temperature
T_fluid_measured = (T_in + T_out)/2

# Timestep of input data
dt_data = 30 #s
Time = np.linspace(dt_data,dt_data*groundload.size,groundload.size)

plt.plot(T_in)
plt.show()

# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def ierf(X):
	return X*special.erf(X)-(1/np.sqrt(np.pi))*(1-np.exp(-X**2))
	
def Ils(h,d):
	return 2*ierf(h) + 2*ierf(h+2*d) - ierf(2*h+2*d) - ierf(2*d)
	
def T_FLSjc_G(x,y,ro,z,H,lm,Cm,t):
	# Finite Line Sourve: Javed and Claesson 2011
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]	
	T = (1.0/4.0/np.pi/lm)*(integrate.quad(lambda s,r,H,Dt,t: np.exp(-r**2 * s**2)* Ils(H*s,1*s)/(H*s**2),1/np.sqrt(4*Dt*t),np.inf,args = (r,H,Dt,t))[0])
	return T

def T_ILS_G(x,y,ro,z,H,lm,Cm,t):
	# (ILS) Infinite Line Source : Carslaw & Jaeger 1959 
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]	
	T = (1.0/4.0/np.pi/lm)*special.expn(1,r**2.0/4.0/Dt/t)
	return T
	
def T_ICS_G(x,y,ro,z,H,lm,Cm,t):
	# (ICS) Infinite Cylindrical Source : Man et al. (2010) 	
	Dt = lm/Cm             	 # thermal diffusivity [m2/s]
	r = np.sqrt(x**2+y**2)   # radial distance [m]	
	def Func(f,r,ro,Dt,t):
		return (1/np.pi)*special.expn(1,((r**2+ro**2-2*r*ro*np.cos(f))/4.0/Dt/t))
	T = (1.0/4.0/np.pi/lm)*(integrate.quad(Func,0,np.pi,args=(r,ro,Dt,t))[0])
	return T
	
def T_FullScale(x,y,ro,z,H,lm,Cm,t):
	# Full Scale G-Function (Mi et al 2014)
	T = T_ICS_G(x,y,ro,z,H,lm,Cm,t) + T_FLSjc_G(x,y,ro,z,H,lm,Cm,t) - T_ILS_G(x,y,ro,z,H,lm,Cm,t)
	return T

# Laplace Transformation 
def laplace(a,sig,t):
	return np.fft.fft(a*np.exp(-sig*t))

def invLaplace(a,sig,t):
	return np.exp(sig*t)*np.fft.ifft(a)	
	
def optimizer(optPara):
	
	
	tlog = np.geomspace(1, dt_data*groundload.size+1, 200)	
	T_flSjc_Glog = np.zeros(tlog.size)	
	for i in range(0,tlog.size):
		T_flSjc_Glog[i] = eval('T_FullScale(rb,0,rb,H/2,H,optPara[2],cS,tlog[i])')
	lin_g = interpolate.interp1d(tlog,T_flSjc_Glog)	
	
	T_borehole = np.ones(Time.size)*optPara[1]		
	Lap_Gfunc = laplace(lin_g(Time),sigma,Time)		

	# Calc Tborehole
	T_borehole = T_borehole -1*np.real(invLaplace(Lap_deltaG*Lap_Gfunc,sigma,Time)) 
	T_Fluid_calc = T_borehole - groundload*optPara[0]

	# ignore values where flow is ~ 0
	T_Fluid_calc[m_flow < 0.3] = float('nan')
	T_fluid_measured[m_flow < 0.3] = float('nan')

	error = (np.nanmean(np.abs(T_Fluid_calc - T_fluid_measured)/T_fluid_measured*100))
	print('Tungest: ',optPara[1])
	print('Rb: ',optPara[0])
	print('lmS: ',optPara[2])
	print('error: ',error)
	print('-----------------')
	return error


	

# -------------------------------------------------------------------------
# Optimize
# -------------------------------------------------------------------------		


# some parameteres
Nt = Time.size 						
tmax = np.max(Time) 				
sigma = 2*np.log(Nt)/tmax
Lap_deltaG = laplace(delta_groundload ,sigma,Time)

# Boundaries
Rb = (0.07,0.1)
lmS = (2, 4)
Tungest = (10,15)

bounds = [Rb,Tungest,lmS]
initialGuess = [0.09,10,3]



# Brute
resbrute = optimize.brute(optimizer, bounds, full_output=False,finish=optimize.fmin)
print(resbrute)

'''
# basinhopping
minimizer_kwargs = {"method":"L-BFGS-B", "jac":False, "bounds": bounds}
results = dict()
results= optimize.basinhopping(optimizer, initialGuess,  minimizer_kwargs=minimizer_kwargs, niter = 10, niter_success = 2)
print(results)
'''
