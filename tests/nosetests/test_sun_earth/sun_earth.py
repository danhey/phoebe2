#
# The Sun-Earth system
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import phoebe
phoebe.devel_on()
from phoebe import u, c
import libphoebe


BLACKBODY = True


def initiate_sun_earth_system(pb_str):
  
    b = phoebe.Bundle.default_binary()
  
    b.add_dataset('lc', times=[0.75,], dataset='lc01', passband=pb_str)

    b['pblum@primary'] = 1.*u.solLum #* 0.99 # 0.99 is bolometric correction
    b['teff@primary'] = 1.*u.solTeff
    b['rpole@primary'] = 1.*u.solRad
    b['syncpar@primary'] = 14.61

    b['period@orbit'] = 1.*u.yr
    b['q@orbit'] = 3.986004e14/1.3271244e20   # (GM)_E / (GM)_Sun
    b['sma@orbit'] = 1.*u.au

    b['teff@secondary'] = (300, 'K')
    b['rpole@secondary'] = 1.*c.R_earth
    b['syncpar@secondary'] = 365.25

    b['distance@system'] = (1, 'au')
    
    b.set_value_all('irrad_method', 'none')

    if BLACKBODY:
        b.set_value_all('atm', value='blackbody')
        b.set_value_all('ld_func', value='linear')
        b.set_value_all('ld_coeffs', value=[0.0])
    else:
        b.set_value_all('atm', component='secondary', value='blackbody')
        b.set_value_all('ld_func', component='primary', value='interp')
        b.set_value_all('ld_func', component='secondary', value='linear')
        b.set_value_all('ld_coeffs', component='secondary', value=[0.0])

    return b

def integrated_flux(b, pb):
  
  r = b['value@abs_intensities@primary']
  r *= b['areas@primary@pbmesh'].get_value(unit=u.m**2)
  r *= b['value@mus@primary@pbmesh']
  r *= b['value@visibilities@primary@pbmesh']
  
  return np.sum(r)*pb.ptf_area/b['value@distance@system']**2
  

def _planck(lam, Teff):
    return 2*c.h.si.value*c.c.si.value*c.c.si.value/lam**5 * 1./(np.exp(c.h.si.value*c.c.si.value/lam/c.k_B.si.value/Teff)-1)
    

def sun_earth_result():

  pb_str = 'Bolometric:900-40000'
  mypb = phoebe.atmospheres.passbands.get_passband(pb_str)

  # theoretical result: planck formula + passband
  sedptf = lambda w: _planck(w, 5772)*mypb.ptf(w)
  sb_flux = np.pi*integrate.quad(sedptf, mypb.ptf_table['wl'][0], mypb.ptf_table['wl'][-1])[0] # Stefan-Boltzmann flux
  iflux0 = sb_flux*(1*u.solRad).si.value**2/c.au.si.value**2

  # phoebe result for different mesh sizes
  b = initiate_sun_earth_system(pb_str)

  res=[]
  for Nt in 1000*(2**np.arange(8)):
    b['ntriangles@primary'] = Nt
    b['ntriangles@secondary'] = Nt

    b.run_compute(protomesh=False, pbmesh=True, mesh_offset=True)
    
    opts = (b['value@q@orbit'], b['value@syncpar@primary'], 1., b['value@pot@primary@component'])
    area0 = libphoebe.roche_area_volume(*opts)['larea']
    area0 *= b['value@sma@orbit']**2
    
    area = np.sum(b['value@areas@primary@pbmesh'])
    iflux = integrated_flux(b, mypb)
    
    res.append([Nt, area-area0, iflux - iflux0])
  
  return np.array(res)  


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    
    res = sun_earth_result()
    
    print res
    
    np.savetxt("res.txt", res)
