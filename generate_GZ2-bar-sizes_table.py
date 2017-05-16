# Python code for generating the combined GZ2/Hoyle+2011 data file
#    GalaxyZoo2_barlengths_alldata.txt
# which contains bar sizes from Hoyle+2011 (corrected to radii with h^-1 removed)
# and stellar masses estimated from the SDSS g and i magnitudes.
#
# This can be executed via
#    $ python generate_GZ2-bar-sizes_table.py
#
# WARNING: This code requires access to the gz2sample.csv data file, which is
# not included in the repository; it can be found at https://data.galaxyzoo.org

import math
from astropy.io import ascii
import s4gutils

baseDir = "./data/external/"

# Read in data from Hoyle+2011 table
ff = baseDir + "hoyle_barlengths.csv"
hoyle11data = ascii.read(ff)
hoyle11names = hoyle11data['SDSS objid']
nHoyle11 = len(hoyle11names)

# remove h^-1 effect:
h11_barsizes = 0.7 * hoyle11data['average bar length [kpc/h]']
h11_barsizes_std = 0.7 * hoyle11data['standard deviation of bar length [kpc/h]']
# covert to *radial* sizes
h11_barsizes = h11_barsizes / 2.0
h11_barsizes_std = h11_barsizes_std / 2.0


# Get g and i magnitudes from Galaxy Zoo 2 main sample table for galaxies in Hoyle+2011
# *** WARNING: this file is about 200 MB in size, and must be retrieved 
ff2 = baseDir + "gz2sample.csv"
try:
	gz2data = ascii.read(ff2)
except:
	print("\nUnable to find Galaxy Zoo 2 SDSS metadata file \"{0}\"!".format(ff))
	print("This may need to be downloaded from data.galaxyzoo.org first!\n")
	raise


# match on SDSS OBJID
gz2dict = {}
nGZ2all = len(gz2data['OBJID'])
for i in range(nGZ2all):
	gid = gz2data['OBJID'][i]
	if gid in hoyle11names:
		gz2dict[gid] = (gz2data['PETROMAG_MG'][i], gz2data['PETROMAG_MR'][i], 
						gz2data['PETROMAG_MI'][i], gz2data['REDSHIFT'][i])


# Generate stellar-mass estimates
def H11logMstar( M_g, M_i, z ):
	gmi = M_g - M_i
	M_star = s4gutils.AbsMagToStellarMass(M_i, 'i', 'g-i', gmi, mode='Zibetti')
	return math.log10(M_star)
	
h11_logmstar = []
for i in range(nHoyle11):
	gid = hoyle11names[i]
	gz2tuple = gz2dict[gid]
	M_g, M_i, z = gz2tuple[0], gz2tuple[2], gz2tuple[3]
	logmstar = H11logMstar(M_g, M_i, z)
	h11_logmstar.append(logmstar)


# Write out combined-data table
outf = open(baseDir+'GalaxyZoo2_barlengths_alldata.txt','w')
headerLine = "#SDSS_OBJID       sma_kpc err_sma_kpc logMstar  M_g       M_i      z\n"
templateLine = "%18d   %.3f   %.3f   %6.3f   %5.3f   %5.3f   %.5f"
outf.write(headerLine)
for i in range(nHoyle11):
	gid = hoyle11names[i]
	gz2tuple = gz2dict[gid]
	M_g, M_i, z = gz2tuple[0], gz2tuple[2], gz2tuple[3]
	line = templateLine % (gid, h11_barsizes[i], h11_barsizes_std[i], h11_logmstar[i], 
				M_g, M_i, z)
	outf.write(line + "\n")
outf.close()
