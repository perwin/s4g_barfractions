# Miscellaneous code for analysis of S4G bar fractions

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
	

