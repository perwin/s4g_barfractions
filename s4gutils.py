# Miscellaneous code for analysis of S4G bar fractions

import copy
import math
import random
import numpy as np

random.seed()


# lower and upper bounds of 68.3% confidence interval:
ONESIGMA_LOWER = 0.1585
ONESIGMA_UPPER = 0.8415



def dtomm( distanceMpc ):
	"""converts distance in Mpc to distance modulus (M - m, in magnitudes)
	"""
	five_logD = 5.0 * np.log10(distanceMpc)
	return (25.0 + five_logD)


def HIMassToFlux( M_HI, dist_Mpc ):
	"""Converts H I mass (in solar masses) to equivalent H I flux (in Jy km/s)
	based on distance in Mpc.  Equation originally from Giovanelli & Haynes
	(1988, in Galactic and extragalactic radio astronomy (2nd edition), p.522),
	based on Roberts (1975, n A. Sandage, M. Sandage, and J. Kristian (eds.), 
	Galaxies and the Universe. Chicago:	University of Chicago Press; p.	309).
	"""
	
	return M_HI / (2.356e5 * dist_Mpc**2)


def GetRadialSampleFromSphere( rMin, rMax ):
	"""Get radius sample from spherical Euclidean volume (or spherical shell) using 
	the discarding method: generate a random point within a cube of half-width = rMax; 
	discard and re-generate if radius to that point is outside [rMin, rMax]
	"""
	rMin2 = rMin*rMin
	rMax2 = rMax*rMax
	done = False
	while not done:
		x = random.uniform(-rMax, rMax)
		y = random.uniform(-rMax, rMax)
		z = random.uniform(-rMax, rMax)
		r2 = x*x + y*y + z*z
		if (r2 >= rMin2) and (r2 <= rMax2):
			done = True
	return math.sqrt(r2)	


def ConfidenceInterval( vect ):
	
	nVals = len(vect)
	lower_ind = int(round(ONESIGMA_LOWER*nVals)) - 1
	upper_ind = int(round(ONESIGMA_UPPER*nVals))
	vect_sorted = copy.copy(vect)
	vect_sorted.sort()
	return (vect_sorted[lower_ind], vect_sorted[upper_ind])


def Binomial( n, n_tot, nsigma=1.0, conf_level=None, method="wilson" ):
	"""Computes fraction (aka frequency or rate) of occurances p = (n/n_tot).
	Also computes the lower and upper confidence limits using either the 
	Wilson (1927) or Agresti & Coull (1998) method (method="wilson" or method="agresti");
	default is to use Wilson method.
	Default is to calculate 68.26895% confidence limits (i.e., 1-sigma in the
	Gaussian approximation).
	
	Returns tuple of (p, sigma_minus, sigma_plus).
	"""
	
	# the following works only for scalar inputs
# 	if (n_tot == 0):
# 		return (0.0, 0.0, 0.0)
	
	p = (1.0 * n) / n_tot
	q = 1.0 - p
	
	if (conf_level is not None):
		print("Alternate values of nsigma or conf_limit not yet supported!")
		alpha = 1.0 - conf_level
		# R code would be the following:
		#z_alpha = qnorm(1.0 - alpha/2.0)
		return None
	else:
		z_alpha = nsigma   # e.g., z_alpha = nsigma = 1.0 for 68.26895% conf. limits
	
	if (method == "wald"):
		# Wald (aka asymptotic) method -- don't use except for testing purposes!
		sigma_minus = sigma_plus = z_alpha * np.sqrt(p*q/n_tot)
	else:
		z_alpha2 = z_alpha**2
		n_tot_mod = n_tot + z_alpha2
		p_mod = (n + 0.5*z_alpha2) / n_tot_mod
		if (method == "wilson"):
			# Wilson (1927) method
			sigma_mod = np.sqrt(z_alpha2 * n_tot * (p*q + z_alpha2/(4.0*n_tot))) / n_tot_mod
		elif (method == "agresti"):
			# Agresti=Coull method
			sigma_mod = np.sqrt(z_alpha2 * p_mod * (1.0 - p_mod) / n_tot_mod)
		else:
			print("ERROR: method \"%s\" not implemented in Binomial!" % method)
			return None
		p_upper = p_mod + sigma_mod
		p_lower = p_mod - sigma_mod
		sigma_minus = p - p_lower
		sigma_plus = p_upper - p
	
	return (p, sigma_minus, sigma_plus)


def ConfidenceInterval( vect ):
	
	nVals = len(vect)
	lower_ind = int(round(ONESIGMA_LOWER*nVals)) - 1
	upper_ind = int(round(ONESIGMA_UPPER*nVals))
	vect_sorted = copy.copy(vect)
	vect_sorted.sort()
	return (vect_sorted[lower_ind], vect_sorted[upper_ind])
	


