#!/usr/bin/env python

# Script for generating simulated surveys for measuring bar fractions as a 
# function of stellar mass or gas mass ratio. Outputs are stored in the
# directory specified by baseDirSurv.

from __future__ import print_function
import simulate_surveys

# output directory for simulation results
baseDirSurv = "./data/"


# column headers for output files
colHeaders_logmstar = "# logMstar	medFbar		sigma_low	sigma_high\n"
colHeaders_logfgas = "# log_fgas	medFbar		sigma_low	sigma_high\n"


# random-number-generator seed used for simulations in paper; change for
# different pseudorandom number sequences (set = None to have the seed 
# be based on the current time)
randomSeed = 100



# * Simulations measuring f_bar as a function of stellar mass for GZ2-style surveys
# GZoo -- M_star, D < 30 Mpc:
# 10,000 galaxies, to match typical sample sizes in GZ2 and other SDSS surveys
n = 10000
nFWHM = 2.0
zrange_mstar = [0.01, 0.05]

mstar_bincenters, fmed, flow, fhigh = simulate_surveys.GenerateAndObserveNTimes(200, simulate_surveys.dset_d30_sp, 
zrange_mstar, n, nFWHM*1.4, 9.0, 11.25, 0.25, useObservedSizes=False, randomSeed=randomSeed)
sigma_minus_2fwhm = fmed - flow
sigma_plus_2fwhm = fhigh - fmed
ff = baseDirSurv + "sim_logMstar_d30_sp_2xfwhm_SDSS_200_dp-sizes.txt"
outf = open(ff, 'w')
outf.write(colHeaders_logmstar)
for i in range(len(fmed)):
	outf.write("%.2f   %.4f   %.4f   %.4f\n" % (mstar_bincenters[i], fmed[i], sigma_minus_2fwhm[i], sigma_plus_2fwhm[i]))
outf.close()
print("Output saved to %s." % ff)


# * Simulations measuring f_bar as a function of gas mass ratio for GZ2-style surveys
# GZoo -- f_gas, D < 30 Mpc, log(M_star) > 9.5:
# 2000 galaxies, to match sample size in Masters+2012, Cervantes Sodi 2017
n = 2000
zrange_fgas = [0.01,0.05]

fgas_bincenters, fmed, flow, fhigh = simulate_surveys.GenerateAndObserveNTimes(200, simulate_surveys.dset_d30m95_sp, 
zrange_fgas, n, nFWHM*1.4, -2,1,0.5, value='fgas', useObservedSizes=False, useHILimit=False, randomSeed=randomSeed, debug=True)
sigma_minus_2fwhm = fmed - flow
sigma_plus_2fwhm = fhigh - fmed
ff = baseDirSurv + "sim_logfgas_d30_2xfwhm_SDSS_200_dp-sizes.txt"
outf = open(ff, 'w')
outf.write(colHeaders_logfgas)
for i in range(len(fmed)):
	outf.write("%.2f   %.4f   %.4f   %.4f\n" % (fgas_bincenters[i], fmed[i], sigma_minus_2fwhm[i], sigma_plus_2fwhm[i]))
outf.close()
print("Output saved to %s." % ff)

# Same, but now including simple H I detection threshold
fgas_bincenters, fmed, flow, fhigh = simulate_surveys.GenerateAndObserveNTimes(200, simulate_surveys.dset_d30m95_sp, 
zrange_fgas, n, nFWHM*1.4, -2,1,0.5, value='fgas', useObservedSizes=False, useHILimit=True, randomSeed=randomSeed, debug=True)
sigma_minus_2fwhm = fmed - flow
sigma_plus_2fwhm = fhigh - fmed
ff = baseDirSurv + "sim_logfgas_d30m95_sp_2xfwhm_SDSS_200_HI-limited.txt"
outf = open(ff, 'w')
outf.write(colHeaders_logfgas)
for i in range(len(fmed)):
	outf.write("%.2f   %.4f   %.4f   %.4f\n" % (fgas_bincenters[i], fmed[i], sigma_minus_2fwhm[i], sigma_plus_2fwhm[i]))
outf.close()
print("Output saved to %s." % ff)



# * Simulation for f_bar as a function of stellar mass in an 
# HST-style survey (e.g., Sheth+2008), assuming all galaxies are at z = 0.75
zrange_075 = [0.75,0.75]
n = 1000
mstar_bincenters, fmed, flow, fhigh = simulate_surveys.GenerateAndObserveNTimes(200, simulate_surveys.dset_d30_sp, 
zrange_075, n, nFWHM*0.1, 9.0, 11.5, 0.25, useObservedSizes=False, randomSeed=randomSeed)
sigma_minus_2fwhm = fmed - flow
sigma_plus_2fwhm = fhigh - fmed
ff = baseDirSurv + "sim_logMstar_d30_sp_2xfwhm_HST_z0.75_200_dp-sizes.txt"
outf = open(ff, 'w')
outf.write(colHeaders_logmstar)
for i in range(len(fmed)):
	outf.write("%.2f   %.4f   %.4f   %.4f\n" % (mstar_bincenters[i], fmed[i], sigma_minus_2fwhm[i], sigma_plus_2fwhm[i]))
outf.close()
print("Output saved to %s." % ff)

# Same, but now assuming all bars have half their z=0 size
mstar_bincenters, fmed, flow, fhigh = simulate_surveys.GenerateAndObserveNTimes(200, simulate_surveys.dset_d30_sp, 
zrange_075, n, nFWHM*0.1, 9.0, 11.5, 0.25, useObservedSizes=False, scaleSizes=0.5, randomSeed=randomSeed)
sigma_minus_2fwhm = fmed - flow
sigma_plus_2fwhm = fhigh - fmed
ff = baseDirSurv + "sim_logMstar_d30_sp_2xfwhm_HST_z0.75_200_dp-sizes_scale0.5.txt"
outf = open(ff, 'w')
outf.write(colHeaders_logmstar)
for i in range(len(fmed)):
	outf.write("%.2f   %.4f   %.4f   %.4f\n" % (mstar_bincenters[i], fmed[i], sigma_minus_2fwhm[i], sigma_plus_2fwhm[i]))
outf.close()
print("Output saved to %s." % ff)

