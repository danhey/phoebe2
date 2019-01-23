#from phoebe.c import h, c, k_B
#from phoebe import u
from phoebe import __version__ as phoebe_version

# NOTE: we'll import directly from astropy here to avoid
# circular imports BUT any changes to these units/constants
# inside phoebe will be ignored within passbands
from astropy.constants import h, c, k_B, sigma_sb
from astropy import units as u

import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import curve_fit as cfit
from datetime import datetime
import marshal
import pickle
import types
import libphoebe
import os
import sys
import glob
import shutil
import json
import time

try:
    # For Python 3.0 and later
    from urllib.request import urlopen, urlretrieve
    from urllib.error import URLError, HTTPError

except ImportError:
    # Fall back to Python 2's urllib, urllib2
    from urllib import urlretrieve
    from urllib2 import urlopen, URLError, HTTPError

from phoebe.utils import parse_json

import logging
logger = logging.getLogger("PASSBANDS")
logger.addHandler(logging.NullHandler())

# Global passband table. This dict should never be tinkered with outside
# of the functions in this module; it might be nice to make it read-only
# at some point.
_pbtable = {}

_initialized = False
_online_passbands = {}

_pbdir_global = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))+'/'

# if we're in a virtual environment then we want don't want to use the home directory
# this check may fail for Python 3
if hasattr(sys, 'real_prefix'):
    # then we're running in a virtualenv
    _pbdir_local = os.path.join(sys.prefix, '.phoebe/atmospheres/tables/passbands/')
else:
    _pbdir_local = os.path.abspath(os.path.expanduser('~/.phoebe/atmospheres/tables/passbands'))+'/'

if not os.path.exists(_pbdir_local):
    logger.info("creating directory {}".format(_pbdir_local))
    os.makedirs(_pbdir_local)

_pbdir_env = os.getenv('PHOEBE_PBDIR', None)

class Passband:
    def __init__(self, ptf=None, pbset='Johnson', pbname='V', effwl=5500.0,
                 wlunits=u.AA, calibrated=False, reference='', version=1.0,
                 comments='', oversampling=1, spl_order=3, from_file=False):
        """
        <phoebe.atmospheres.passbands.Passband> class holds data and tools for
        passband-related computations, such as blackbody intensity, model
        atmosphere intensity, etc.

        Step #1: initialize passband object

        ```py
        pb = Passband(ptf='JOHNSON.V', pbset='Johnson', pbname='V', effwl=5500.0, wlunits=u.AA, calibrated=True, reference='ADPS', version=1.0, comments='')
        ```

        Step #2: compute intensities for blackbody radiation:

        ```py
        pb.compute_blackbody_response()
        ```

        Step #3: compute Castelli & Kurucz (2004) intensities. To do this,
        the tables/ck2004 directory needs to be populated with non-filtered
        intensities available for download from %static%/ck2004.tar.

        ```py
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/ck2004'))
        pb.compute_ck2004_response(atmdir)
        ```

        Step #4: -- optional -- import WD tables for comparison. This can only
        be done if the passband is in the list of supported passbands in WD.
        The WD index of the passband is passed to the import_wd_atmcof()
        function below as the last argument.

        ```py
        from phoebe.atmospheres import atmcof
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
        atmcof.init(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')
        pb.import_wd_atmcof(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat', 7)
        ```

        Step #5: save the passband file:

        ```py
        atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/passbands'))
        pb.save(atmdir + '/johnson_v.ptf')
        ```

        From now on you can use `pbset`:`pbname` as a passband qualifier, i.e.
        Johnson:V for the example above. Further details on supported model
        atmospheres are available by issuing:

        ```py
        pb.content
        ```

        see <phoebe.atmospheres.passbands.content>

        Arguments
        ----------
        * `ptf` (string, optional, default=None): passband transmission file: a
            2-column file with wavelength in @wlunits and transmission in
            arbitrary units.
        * `pbset` (string, optional, default='Johnson'): name of the passband
            set (i.e. Johnson).
        * `pbname` (string, optional, default='V'): name of the passband name
            (i.e. V).
        * `effwl` (float, optional, default=5500.0): effective wavelength in
            `wlunits`.
        * `wlunits` (unit, optional, default=u.AA): wavelength units from
            astropy.units used in `ptf` and `effwl`.
        * `calibrated` (bool, optional, default=False): true if transmission is
            in true fractional light, false if it is in relative proportions.
        * `reference` (string, optional, default=''): passband transmission data
            reference (i.e. ADPS).
        * `version` (float, optional, default=1.0): file version.
        * `comments` (string, optional, default=''): any additional comments
            about the passband.
        * `oversampling` (int, optional, default=1): the multiplicative factor
            of PTF dispersion to attain higher integration accuracy.
        * `spl_order` (int, optional, default=3): spline order for fitting
            the passband transmission function.
        * `from_file` (bool, optional, default=False): a switch that instructs
            the class instance to skip all calculations and load all data from
            the file passed to the <phoebe.atmospheres.passbands.Passband.load>
            method.

        Returns
        ---------
        * an instatiated <phoebe.atmospheres.passbands.Passband> object.
        """
        self.h = h.value
        self.c = c.value
        self.k = k_B.value

        if from_file:
            return

        # Initialize content list; each method that adds any content
        # to the passband file needs to add a corresponding label to the
        # content list.
        self.content = []

        # Initialize atmosphere list; these names match the names of the
        # atmosphere models in the atm parameter. As above, when an atm
        # table is added, this list is appended.
        self.atmlist = []

        # Basic passband properties:
        self.pbset = pbset
        self.pbname = pbname
        self.effwl = effwl
        self.calibrated = calibrated
        self.reference = reference
        self.version = version
        self.comments = comments

        # Initialize an empty timestamp. This will get set by calling the save() method.
        self.timestamp = None

        # Passband transmission function table:
        ptf_table = np.loadtxt(ptf).T
        ptf_table[0] = ptf_table[0]*wlunits.to(u.m)
        self.ptf_table = {'wl': np.array(ptf_table[0]), 'fl': np.array(ptf_table[1])}

        # Working (optionally oversampled) wavelength array:
        self.wl = np.linspace(self.ptf_table['wl'][0], self.ptf_table['wl'][-1], oversampling*len(self.ptf_table['wl']))

        # Spline fit to the energy-weighted passband transmission function table:
        self.ptf_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl'], s=0, k=spl_order)
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)
        self.ptf_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_func, 0)

        # Spline fit to the photon-weighted passband transmission function table:
        self.ptf_photon_func = interpolate.splrep(self.ptf_table['wl'], self.ptf_table['fl']*self.ptf_table['wl'], s=0, k=spl_order)
        self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)
        self.ptf_photon_area = interpolate.splint(self.wl[0], self.wl[-1], self.ptf_photon_func, 0)

    def __repr__(self):
        return('<Passband: %s:%s>' % (self.pbset, self.pbname))

    def __str__(self):
        # old passband files do not have versions embedded, that is why we have to do this:
        if not hasattr(self, 'version'):
            self.version = 1.0
        return('Passband: %s:%s\nVersion:  %1.1f\nProvides: %s' % (self.pbset, self.pbname, self.version, self.content))

    def save(self, archive):
        """
        Save the <phoebe.atmospheres.passbands.Passband> to a file.

        Arguments
        ------------
        * `archive` (string): filename
        """
        struct = dict()

        struct['originating_phoebe_version'] = phoebe_version

        struct['content']         = self.content
        struct['atmlist']         = self.atmlist
        struct['pbset']           = self.pbset
        struct['pbname']          = self.pbname
        struct['effwl']           = self.effwl
        struct['calibrated']      = self.calibrated
        struct['version']         = self.version
        struct['comments']        = self.comments
        struct['reference']       = self.reference
        struct['ptf_table']       = self.ptf_table
        struct['ptf_wl']          = self.wl
        struct['ptf_func']        = self.ptf_func
        struct['ptf_area']        = self.ptf_area
        struct['ptf_photon_func'] = self.ptf_photon_func
        struct['ptf_photon_area'] = self.ptf_photon_area
        if 'blackbody' in self.content:
            struct['_bb_func_energy'] = self._bb_func_energy
            struct['_bb_func_photon'] = self._bb_func_photon
        if 'ck2004' in self.content:
            struct['_ck2004_axes'] = self._ck2004_axes
            struct['_ck2004_energy_grid'] = self._ck2004_energy_grid
            struct['_ck2004_photon_grid'] = self._ck2004_photon_grid
        if 'ck2004_all' in self.content:
            struct['_ck2004_intensity_axes'] = self._ck2004_intensity_axes
            struct['_ck2004_Imu_energy_grid'] = self._ck2004_Imu_energy_grid
            struct['_ck2004_Imu_photon_grid'] = self._ck2004_Imu_photon_grid
            struct['_ck2004_boosting_energy_grid'] = self._ck2004_boosting_energy_grid
            struct['_ck2004_boosting_photon_grid'] = self._ck2004_boosting_photon_grid
        if 'ck2004_ld' in self.content:
            struct['_ck2004_ld_energy_grid'] = self._ck2004_ld_energy_grid
            struct['_ck2004_ld_photon_grid'] = self._ck2004_ld_photon_grid
        if 'ck2004_ldint' in self.content:
            struct['_ck2004_ldint_energy_grid'] = self._ck2004_ldint_energy_grid
            struct['_ck2004_ldint_photon_grid'] = self._ck2004_ldint_photon_grid
        if 'extern_planckint' in self.content and 'extern_atmx' in self.content:
            struct['extern_wd_idx'] = self.extern_wd_idx

        # Finally, timestamp the file:
        struct['timestamp'] = self.timestamp = time.ctime()

        with open(archive, 'wb') as f:
            if sys.version_info[0] < 3:
                marshal.dump(struct, f)
            else:
                pickle.dump(struct, f, protocol=3)

    @classmethod
    def load(cls, archive):
        """
        Load the <phoebe.atmospheres.passbands.Passband> from a file.

        This is a constructor so should be called as:

        ```py
        pb = Passband.load(filename)
        ```

        Arguments
        ----------
        * `archive` (string): filename

        Returns
        ---------
        * an instatiated <phoebe.atmospheres.passbands.Passband> object.
        """
        logger.debug("loading passband from {}".format(archive))
        with open(archive, 'rb') as f:
            try:
                if sys.version_info[0] < 3:
                    struct = marshal.load(f)
                    marshaled = True
                else:
                    struct = pickle.load(f)
                    marshaled = False
            except Exception as e:
                print("failed to load passband from {}".format(archive))
                raise e

        self = cls(from_file=True)

        self.content = struct['content']
        self.atmlist = struct['atmlist']

        self.pbset = struct['pbset']
        self.pbname = struct['pbname']
        self.effwl = struct['effwl']
        self.calibrated = struct['calibrated']

        # these are new additions and not every pb file has them.
        self.opv = struct.get('originating_phoebe_version', None)
        self.version = struct.get('version', None)
        self.comments = struct.get('comments', None)
        self.reference = struct.get('reference', None)
        self.timestamp = struct.get('timestamp', None)

        self.ptf_table = struct['ptf_table']
        if marshaled:
            self.ptf_table['wl'] = np.fromstring(self.ptf_table['wl'], dtype='float64')
            self.ptf_table['fl'] = np.fromstring(self.ptf_table['fl'], dtype='float64')
            self.wl = np.fromstring(struct['ptf_wl'], dtype='float64')
        else:
            self.wl = struct['ptf_wl']
        self.ptf_area = struct['ptf_area']
        self.ptf_photon_area = struct['ptf_photon_area']

        if marshaled:
            self.ptf_func = list(struct['ptf_func'])
            self.ptf_func[0] = np.fromstring(self.ptf_func[0])
            self.ptf_func[1] = np.fromstring(self.ptf_func[1])
            self.ptf_func = tuple(self.ptf_func)
        else:
            self.ptf_func = struct['ptf_func']
        self.ptf = lambda wl: interpolate.splev(wl, self.ptf_func)

        if marshaled:
            self.ptf_photon_func = list(struct['ptf_photon_func'])
            self.ptf_photon_func[0] = np.fromstring(self.ptf_photon_func[0])
            self.ptf_photon_func[1] = np.fromstring(self.ptf_photon_func[1])
            self.ptf_photon_func = tuple(self.ptf_photon_func)
        else:
            self.ptf_photon_func = struct['ptf_photon_func']
        self.ptf_photon = lambda wl: interpolate.splev(wl, self.ptf_photon_func)

        if 'blackbody' in self.content:
            if marshaled:
                self._bb_func_energy = list(struct['_bb_func_energy'])
                self._bb_func_energy[0] = np.fromstring(self._bb_func_energy[0])
                self._bb_func_energy[1] = np.fromstring(self._bb_func_energy[1])
                self._bb_func_energy = tuple(self._bb_func_energy)
            else:
                self._bb_func_energy = struct['_bb_func_energy']
            self._log10_Inorm_bb_energy = lambda Teff: interpolate.splev(Teff, self._bb_func_energy)

            if marshaled:
                self._bb_func_photon = list(struct['_bb_func_photon'])
                self._bb_func_photon[0] = np.fromstring(self._bb_func_photon[0])
                self._bb_func_photon[1] = np.fromstring(self._bb_func_photon[1])
                self._bb_func_photon = tuple(self._bb_func_photon)
            else:
                self._bb_func_photon = struct['_bb_func_photon']
            self._log10_Inorm_bb_photon = lambda Teff: interpolate.splev(Teff, self._bb_func_photon)

        if 'extern_atmx' in self.content and 'extern_planckint' in self.content:
            atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))

            planck = (atmdir+'/atmcofplanck.dat').encode('utf8')
            atm = (atmdir+'/atmcof.dat').encode('utf8')

            self.wd_data = libphoebe.wd_readdata(planck, atm)
            self.extern_wd_idx = struct['extern_wd_idx']

        if 'ck2004' in self.content:
            # CASTELLI & KURUCZ (2004):
            if marshaled:
                # Axes needs to be a tuple of np.arrays, and grid a np.array:
                self._ck2004_axes  = tuple(map(lambda x: np.fromstring(x, dtype='float64'), struct['_ck2004_axes']))
                self._ck2004_energy_grid = np.fromstring(struct['_ck2004_energy_grid'], dtype='float64')
                self._ck2004_energy_grid = self._ck2004_energy_grid.reshape(len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1)
                self._ck2004_photon_grid = np.fromstring(struct['_ck2004_photon_grid'], dtype='float64')
                self._ck2004_photon_grid = self._ck2004_photon_grid.reshape(len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1)
            else:
                self._ck2004_axes = struct['_ck2004_axes']
                self._ck2004_energy_grid = struct['_ck2004_energy_grid']
                self._ck2004_photon_grid = struct['_ck2004_photon_grid']

        if 'ck2004_all' in self.content:
            # CASTELLI & KURUCZ (2004) all intensities:
            if marshaled:
                # Axes needs to be a tuple of np.arrays, and grid a np.array:
                self._ck2004_intensity_axes  = tuple(map(lambda x: np.fromstring(x, dtype='float64'), struct['_ck2004_intensity_axes']))
                self._ck2004_Imu_energy_grid = np.fromstring(struct['_ck2004_Imu_energy_grid'], dtype='float64')
                self._ck2004_Imu_energy_grid = self._ck2004_Imu_energy_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1)
                self._ck2004_Imu_photon_grid = np.fromstring(struct['_ck2004_Imu_photon_grid'], dtype='float64')
                self._ck2004_Imu_photon_grid = self._ck2004_Imu_photon_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1)
                self._ck2004_boosting_energy_grid = np.fromstring(struct['_ck2004_boosting_energy_grid'], dtype='float64')
                self._ck2004_boosting_energy_grid = self._ck2004_boosting_energy_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1)
                self._ck2004_boosting_photon_grid = np.fromstring(struct['_ck2004_boosting_photon_grid'], dtype='float64')
                self._ck2004_boosting_photon_grid = self._ck2004_boosting_photon_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1)
            else:
                self._ck2004_intensity_axes = struct['_ck2004_intensity_axes']
                self._ck2004_Imu_energy_grid = struct['_ck2004_Imu_energy_grid']
                self._ck2004_Imu_photon_grid = struct['_ck2004_Imu_photon_grid']
                self._ck2004_boosting_energy_grid = struct['_ck2004_boosting_energy_grid']
                self._ck2004_boosting_photon_grid = struct['_ck2004_boosting_photon_grid']

        if 'ck2004_ld' in self.content:
            if marshaled:
                self._ck2004_ld_energy_grid = np.fromstring(struct['_ck2004_ld_energy_grid'], dtype='float64')
                self._ck2004_ld_energy_grid = self._ck2004_ld_energy_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11)
                self._ck2004_ld_photon_grid = np.fromstring(struct['_ck2004_ld_photon_grid'], dtype='float64')
                self._ck2004_ld_photon_grid = self._ck2004_ld_photon_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11)
            else:
                self._ck2004_ld_energy_grid = struct['_ck2004_ld_energy_grid']
                self._ck2004_ld_photon_grid = struct['_ck2004_ld_photon_grid']

        if 'ck2004_ldint' in self.content:
            if marshaled:
                self._ck2004_ldint_energy_grid = np.fromstring(struct['_ck2004_ldint_energy_grid'], dtype='float64')
                self._ck2004_ldint_energy_grid = self._ck2004_ldint_energy_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 1)
                self._ck2004_ldint_photon_grid = np.fromstring(struct['_ck2004_ldint_photon_grid'], dtype='float64')
                self._ck2004_ldint_photon_grid = self._ck2004_ldint_photon_grid.reshape(len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 1)
            else:
                self._ck2004_ldint_energy_grid = struct['_ck2004_ldint_energy_grid']
                self._ck2004_ldint_photon_grid = struct['_ck2004_ldint_photon_grid']

        return self

    def _planck(self, lam, Teff):
        """
        Computes monochromatic blackbody intensity in W/m^3 using the
        Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * monochromatic blackbody intensity
        """

        return 2*self.h*self.c*self.c/lam**5 * 1./(np.exp(self.h*self.c/lam/self.k/Teff)-1)

    def _planck_deriv(self, lam, Teff):
        """
        Computes the derivative of the monochromatic blackbody intensity using
        the Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * the derivative of monochromatic blackbody intensity
        """

        expterm = np.exp(self.h*self.c/lam/self.k/Teff)
        return 2*self.h*self.c*self.c/self.k/Teff/lam**7 * (expterm-1)**-2 * (self.h*self.c*expterm-5*lam*self.k*Teff*(expterm-1))

    def _planck_spi(self, lam, Teff):
        """
        Computes the spectral index of the monochromatic blackbody intensity
        using the Planck function. The spectral index is defined as:

            B(lambda) = 5 + d(log I)/d(log lambda),

        where I is the Planck function.

        Arguments
        -----------
        * `lam` (float/array): wavelength in m
        * `Teff` (float/array): effective temperature in K

        Returns
        --------
        * the spectral index of monochromatic blackbody intensity
        """

        hclkt = self.h*self.c/lam/self.k/Teff
        expterm = np.exp(hclkt)
        return hclkt * expterm/(expterm-1)

    def _bb_intensity(self, Teff, photon_weighted=False):
        """
        Computes mean passband intensity using blackbody atmosphere:

        I_pb^E = \int_\lambda I(\lambda) P(\lambda) d\lambda / \int_\lambda P(\lambda) d\lambda
        I_pb^P = \int_\lambda \lambda I(\lambda) P(\lambda) d\lambda / \int_\lambda \lambda P(\lambda) d\lambda

        Superscripts E and P stand for energy and photon, respectively.

        Arguments
        -----------
        * `Teff` (float/array): effective temperature in K
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ------------
        * mean passband intensity using blackbody atmosphere.
        """

        if photon_weighted:
            pb = lambda w: w*self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(pb, self.wl[0], self.wl[-1])[0]/self.ptf_photon_area
        else:
            pb = lambda w: self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(pb, self.wl[0], self.wl[-1])[0]/self.ptf_area

    def _bindex_blackbody(self, Teff, photon_weighted=False):
        """
        Computes the mean boosting index using blackbody atmosphere:

        B_pb^E = \int_\lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda I(\lambda) P(\lambda) d\lambda
        B_pb^P = \int_\lambda \lambda I(\lambda) P(\lambda) B(\lambda) d\lambda / \int_\lambda \lambda I(\lambda) P(\lambda) d\lambda

        Superscripts E and P stand for energy and photon, respectively.

        Arguments
        ----------
        * `Teff` (float/array): effective temperature in K
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ------------
        * mean boosting index using blackbody atmosphere.
        """

        if photon_weighted:
            num   = lambda w: w*self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: w*self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]
        else:
            num   = lambda w: self._planck(w, Teff)*self.ptf(w)*self._planck_spi(w, Teff)
            denom = lambda w: self._planck(w, Teff)*self.ptf(w)
            return integrate.quad(num, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-8)[0]/integrate.quad(denom, self.wl[0], self.wl[-1], epsabs=1e10, epsrel=1e-6)[0]

    def compute_blackbody_response(self, Teffs=None):
        """
        Computes blackbody intensities across the entire range of
        effective temperatures. It does this for two regimes, energy-weighted
        and photon-weighted. It then fits a cubic spline to the log(I)-Teff
        values and exports the interpolation functions _log10_Inorm_bb_energy
        and _log10_Inorm_bb_photon.

        Arguments
        ----------
        * `Teffs` (array, optional, default=None): an array of effective
            temperatures. If None, a default array from ~300K to ~500000K with
            97 steps is used. The default array is uniform in log10 scale.
        """

        if Teffs is None:
            log10Teffs = np.linspace(2.5, 5.7, 97) # this corresponds to the 316K-501187K range.
            Teffs = 10**log10Teffs

        # Energy-weighted intensities:
        log10ints_energy = np.array([np.log10(self._bb_intensity(Teff, photon_weighted=False)) for Teff in Teffs])
        self._bb_func_energy = interpolate.splrep(Teffs, log10ints_energy, s=0)
        self._log10_Inorm_bb_energy = lambda Teff: interpolate.splev(Teff, self._bb_func_energy)

        # Photon-weighted intensities:
        log10ints_photon = np.array([np.log10(self._bb_intensity(Teff, photon_weighted=True)) for Teff in Teffs])
        self._bb_func_photon = interpolate.splrep(Teffs, log10ints_photon, s=0)
        self._log10_Inorm_bb_photon = lambda Teff: interpolate.splev(Teff, self._bb_func_photon)

        self.content.append('blackbody')
        self.atmlist.append('blackbody')

    def compute_ck2004_response(self, path, verbose=False):
        """
        Computes Castelli & Kurucz (2004) intensities across the entire
        range of model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory containing ck2004 SEDs.
        * `verbose` (bool, optional, default=False): switch to determine whether
            computing progress should be printed on screen.
        """

        models = glob.glob(path+'/*M1.000*')
        Nmodels = len(models)

        # Store the length of the filename extensions for parsing:
        offset = len(models[0])-models[0].rfind('.')

        Teff, logg, abun = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)
        InormE, InormP = np.empty(Nmodels), np.empty(Nmodels)

        if verbose:
            print('Computing Castelli & Kurucz (2004) passband intensities for %s:%s. This will take a while.' % (self.pbset, self.pbname))

        for i, model in enumerate(models):
            #~ spc = np.loadtxt(model).T -- waaay slower
            spc = np.fromfile(model, sep=' ').reshape(-1,2).T

            Teff[i] = float(model[-17-offset:-12-offset])
            logg[i] = float(model[-11-offset:-9-offset])/10
            sign = 1. if model[-9-offset]=='P' else -1.
            abun[i] = sign*float(model[-8-offset:-6-offset])/10

            spc[0] /= 1e10 # AA -> m
            spc[1] *= 1e7  # erg/s/cm^2/A -> W/m^3
            wl = spc[0][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl = spc[1][(spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])]
            fl *= self.ptf(wl)
            flP = fl*wl
            InormE[i] = np.log10(fl.sum()/self.ptf_area*(wl[1]-wl[0]))             # energy-weighted intensity
            InormP[i] = np.log10(flP.sum()/self.ptf_photon_area*(wl[1]-wl[0]))     # photon-weighted intensity
            if verbose:
                if 100*i % (len(models)) == 0:
                    print('%d%% done.' % (100*i/(len(models)-1)))

        # Store axes (Teff, logg, abun) and the full grid of Inorm, with
        # nans where the grid isn't complete.
        self._ck2004_axes = (np.unique(Teff), np.unique(logg), np.unique(abun))

        self._ck2004_energy_grid = np.nan*np.ones((len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1))
        self._ck2004_photon_grid = np.nan*np.ones((len(self._ck2004_axes[0]), len(self._ck2004_axes[1]), len(self._ck2004_axes[2]), 1))
        for i, I0 in enumerate(InormE):
            self._ck2004_energy_grid[Teff[i] == self._ck2004_axes[0], logg[i] == self._ck2004_axes[1], abun[i] == self._ck2004_axes[2], 0] = I0
        for i, I0 in enumerate(InormP):
            self._ck2004_photon_grid[Teff[i] == self._ck2004_axes[0], logg[i] == self._ck2004_axes[1], abun[i] == self._ck2004_axes[2], 0] = I0

        # Tried radial basis functions but they were just terrible.
        #~ self._log10_Inorm_ck2004 = interpolate.Rbf(self._ck2004_Teff, self._ck2004_logg, self._ck2004_met, self._ck2004_Inorm, function='linear')
        self.content.append('ck2004')
        self.atmlist.append('ck2004')

    def compute_ck2004_intensities(self, path, particular=None, verbose=False):
        """
        Computes direction-dependent passband intensities using Castelli
        & Kurucz (2004) model atmospheres.

        Arguments
        -----------
        * `path` (string): path to the directory with SEDs.
        * `particular` (string, optional, default=None): particular file in
            `path` to be processed; if None, all files in the directory are
            processed.
        * `verbose` (bool, optional, default=False): set to True to display
            progress in the terminal.
        """
        models = os.listdir(path)
        if particular != None:
            models = [particular]
        Nmodels = len(models)

        # Store the length of the filename extensions for parsing:
        offset = len(models[0])-models[0].rfind('.')

        Teff, logg, abun, mu = np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels), np.empty(Nmodels)
        ImuE, ImuP = np.empty(Nmodels), np.empty(Nmodels)
        boostingE, boostingP = np.empty(Nmodels), np.empty(Nmodels)

        if verbose:
            print('Computing Castelli-Kurucz intensities for %s:%s. This will take a long while.' % (self.pbset, self.pbname))

        for i, model in enumerate(models):
            #spc = np.loadtxt(path+'/'+model).T -- waaay slower
            spc = np.fromfile(path+'/'+model, sep=' ').reshape(-1,2).T
            spc[0] /= 1e10 # AA -> m
            spc[1] *= 1e7  # erg/s/cm^2/A -> W/m^3

            Teff[i] = float(model[-17-offset:-12-offset])
            logg[i] = float(model[-11-offset:-9-offset])/10
            sign = 1. if model[-9-offset]=='P' else -1.
            abun[i] = sign*float(model[-8-offset:-6-offset])/10
            mu[i] = float(model[-5-offset:-offset])

            # trim the spectrum at passband limits:
            keep = (spc[0] >= self.ptf_table['wl'][0]) & (spc[0] <= self.ptf_table['wl'][-1])
            wl = spc[0][keep]
            fl = spc[1][keep]

            # make a log-scale copy for boosting and fit a Legendre
            # polynomial to the Imu envelope by way of sigma clipping;
            # then compute a Legendre series derivative to get the
            # boosting index; we only take positive fluxes to keep the
            # log well defined.

            lnwl = np.log(wl[fl > 0])
            lnfl = np.log(fl[fl > 0]) + 5*lnwl

            # First Legendre fit to the data:
            envelope = np.polynomial.legendre.legfit(lnwl, lnfl, 5)
            continuum = np.polynomial.legendre.legval(lnwl, envelope)
            diff = lnfl-continuum
            sigma = np.std(diff)
            clipped = (diff > -sigma)

            # Sigma clip to get the continuum:
            while True:
                Npts = clipped.sum()
                envelope = np.polynomial.legendre.legfit(lnwl[clipped], lnfl[clipped], 5)
                continuum = np.polynomial.legendre.legval(lnwl, envelope)
                diff = lnfl-continuum

                # clipping will sometimes unclip already clipped points
                # because the fit is slightly different, which can lead
                # to infinite loops. To prevent that, we never allow
                # clipped points to be resurrected, which is achieved
                # by the following bitwise condition (array comparison):
                clipped = clipped & (diff > -sigma)

                if clipped.sum() == Npts:
                    break

            derivative = np.polynomial.legendre.legder(envelope, 1)
            boosting_index = np.polynomial.legendre.legval(lnwl, derivative)

            # calculate energy (E) and photon (P) weighted fluxes and
            # their integrals.

            flE = self.ptf(wl)*fl
            flP = wl*flE
            flEint = flE.sum()
            flPint = flP.sum()

            # calculate mean boosting coefficient and use it to get
            # boosting factors for energy (E) and photon (P) weighted
            # fluxes.

            boostE = (flE[fl > 0]*boosting_index).sum()/flEint
            boostP = (flP[fl > 0]*boosting_index).sum()/flPint
            boostingE[i] = boostE
            boostingP[i] = boostP

            ImuE[i] = np.log10(flEint/self.ptf_area*(wl[1]-wl[0]))        # energy-weighted intensity
            ImuP[i] = np.log10(flPint/self.ptf_photon_area*(wl[1]-wl[0])) # photon-weighted intensity

            if verbose:
                if 100*i % (len(models)) == 0:
                    print('%d%% done.' % (100*i/(len(models)-1)))

        # Store axes (Teff, logg, abun, mu) and the full grid of Imu,
        # with nans where the grid isn't complete. Imu-s come in two
        # flavors: energy-weighted intensities and photon-weighted
        # intensities, based on the detector used.

        self._ck2004_intensity_axes = (np.unique(Teff), np.unique(logg), np.unique(abun), np.append(np.array(0.0,), np.unique(mu)))
        self._ck2004_Imu_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        self._ck2004_Imu_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        self._ck2004_boosting_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))
        self._ck2004_boosting_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), len(self._ck2004_intensity_axes[3]), 1))

        # Set the limb (mu=0) to 0; in log this actually means
        # flux=1W/m2, but for all practical purposes that is still 0.
        self._ck2004_Imu_energy_grid[:,:,:,0,:] = 0.0
        self._ck2004_Imu_photon_grid[:,:,:,0,:] = 0.0
        self._ck2004_boosting_energy_grid[:,:,:,0,:] = 0.0
        self._ck2004_boosting_photon_grid[:,:,:,0,:] = 0.0

        for i, Imu in enumerate(ImuE):
            self._ck2004_Imu_energy_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Imu
        for i, Imu in enumerate(ImuP):
            self._ck2004_Imu_photon_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Imu
        for i, Bavg in enumerate(boostingE):
            self._ck2004_boosting_energy_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg
        for i, Bavg in enumerate(boostingP):
            self._ck2004_boosting_photon_grid[Teff[i] == self._ck2004_intensity_axes[0], logg[i] == self._ck2004_intensity_axes[1], abun[i] == self._ck2004_intensity_axes[2], mu[i] == self._ck2004_intensity_axes[3], 0] = Bavg

        self.content.append('ck2004_all')

    def _ldlaw_lin(self, mu, xl):
        return 1.0-xl*(1-mu)

    def _ldlaw_log(self, mu, xl, yl):
        return 1.0-xl*(1-mu)-yl*mu*np.log(mu+1e-6)

    def _ldlaw_sqrt(self, mu, xl, yl):
        return 1.0-xl*(1-mu)-yl*(1.0-np.sqrt(mu))

    def _ldlaw_quad(self, mu, xl, yl):
        return 1.0-xl*(1.0-mu)-yl*(1.0-mu)*(1.0-mu)

    def _ldlaw_nonlin(self, mu, c1, c2, c3, c4):
        return 1.0-c1*(1.0-np.sqrt(mu))-c2*(1.0-mu)-c3*(1.0-mu*np.sqrt(mu))-c4*(1.0-mu*mu)

    def compute_ck2004_ldcoeffs(self, weighting='uniform', plot_diagnostics=False):
        """
        Computes limb darkening coefficients for linear, log, square root,
        quadratic and power laws.

        Arguments
        ----------
        * `weighting` (string, optional, default='uniform'): determines how data
            points should be weighted.
            * 'uniform':  do not apply any per-point weighting
            * 'interval': apply weighting based on the interval widths
        """
        if 'ck2004_all' not in self.content:
            print('Castelli & Kurucz (2004) intensities are not computed yet. Please compute those first.')
            return None

        self._ck2004_ld_energy_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11))
        self._ck2004_ld_photon_grid = np.nan*np.ones((len(self._ck2004_intensity_axes[0]), len(self._ck2004_intensity_axes[1]), len(self._ck2004_intensity_axes[2]), 11))
        mus = self._ck2004_intensity_axes[3] # starts with 0
        if weighting == 'uniform':
            sigma = np.ones(len(mus))
        elif weighting == 'interval':
            delta = np.concatenate( (np.array((mus[1]-mus[0],)), mus[1:]-mus[:-1]) )
            sigma = 1./np.sqrt(delta)
        else:
            print('Weighting scheme \'%s\' is unsupported. Please choose among [\'uniform\', \'interval\']')
            return None

        for Tindex in range(len(self._ck2004_intensity_axes[0])):
            for lindex in range(len(self._ck2004_intensity_axes[1])):
                for mindex in range(len(self._ck2004_intensity_axes[2])):
                    IsE = 10**self._ck2004_Imu_energy_grid[Tindex,lindex,mindex,:].flatten()
                    fEmask = np.isfinite(IsE)
                    if len(IsE[fEmask]) <= 1:
                        continue
                    IsE /= IsE[fEmask][-1]

                    cElin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5])
                    cElog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5])
                    cEnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fEmask], ydata=IsE[fEmask], sigma=sigma, p0=[0.5, 0.5, 0.5, 0.5])
                    self._ck2004_ld_energy_grid[Tindex, lindex, mindex] = np.hstack((cElin, cElog, cEsqrt, cEquad, cEnlin))

                    IsP = 10**self._ck2004_Imu_photon_grid[Tindex,lindex,mindex,:].flatten()
                    fPmask = np.isfinite(IsP)
                    IsP /= IsP[fPmask][-1]

                    cPlin,  pcov = cfit(f=self._ldlaw_lin,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5])
                    cPlog,  pcov = cfit(f=self._ldlaw_log,    xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPsqrt, pcov = cfit(f=self._ldlaw_sqrt,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPquad, pcov = cfit(f=self._ldlaw_quad,   xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5])
                    cPnlin, pcov = cfit(f=self._ldlaw_nonlin, xdata=mus[fPmask], ydata=IsP[fPmask], sigma=sigma, p0=[0.5, 0.5, 0.5, 0.5])
                    self._ck2004_ld_photon_grid[Tindex, lindex, mindex] = np.hstack((cPlin, cPlog, cPsqrt, cPquad, cPnlin))

                    if plot_diagnostics:
                        if Tindex == 10 and lindex == 9 and mindex == 5:
                            print(self._ck2004_intensity_axes[0][Tindex], self._ck2004_intensity_axes[1][lindex], self._ck2004_intensity_axes[2][mindex])
                            print(mus, IsE)
                            print(cElin, cElog, cEsqrt)
                            import matplotlib.pyplot as plt
                            plt.plot(mus[fEmask], IsE[fEmask], 'bo')
                            plt.plot(mus[fEmask], self._ldlaw_lin(mus[fEmask], *cElin), 'r-')
                            plt.plot(mus[fEmask], self._ldlaw_log(mus[fEmask], *cElog), 'g-')
                            plt.plot(mus[fEmask], self._ldlaw_sqrt(mus[fEmask], *cEsqrt), 'y-')
                            plt.plot(mus[fEmask], self._ldlaw_quad(mus[fEmask], *cEquad), 'm-')
                            plt.plot(mus[fEmask], self._ldlaw_nonlin(mus[fEmask], *cEnlin), 'k-')
                            plt.show()

        self.content.append('ck2004_ld')

    def export_legacy_ldcoeffs(self, models, filename=None, photon_weighted=True):
        """
        Exports CK2004 limb darkening coefficients to a PHOEBE legacy
        compatible format.

        Arguments
        -----------
        * `models` (string): the path (including the filename) of legacy's
            models.list
        * `filename` (string, optional, default=None): output filename for
            storing the table
        * `photon_weighted` (bool, optional, default=True): photon/energy switch
        """

        if photon_weighted:
            grid = self._ck2004_ld_photon_grid
        else:
            grid = self._ck2004_ld_energy_grid

        if filename is not None:
            import time
            f = open(filename, 'w')
            f.write('# PASS_SET  %s\n' % self.pbset)
            f.write('# PASSBAND  %s\n' % self.pbname)
            f.write('# VERSION   1.0\n\n')
            f.write('# Exported from PHOEBE-2 passband on %s\n' % (time.ctime()))
            f.write('# The coefficients are computed for the %s-weighted regime.\n\n' % ('photon' if photon_weighted else 'energy'))

        mods = np.loadtxt(models)
        for mod in mods:
            Tindex = np.argwhere(self._ck2004_intensity_axes[0] == mod[0])[0][0]
            lindex = np.argwhere(self._ck2004_intensity_axes[1] == mod[1]/10)[0][0]
            mindex = np.argwhere(self._ck2004_intensity_axes[2] == mod[2]/10)[0][0]
            if filename is None:
                print('%6.3f '*11 % tuple(grid[Tindex, lindex, mindex].tolist()))
            else:
                f.write(('%6.3f '*11+'\n') % tuple(self._ck2004_ld_photon_grid[Tindex, lindex, mindex].tolist()))

        if filename is not None:
            f.close()

    def compute_ck2004_ldints(self):
        """
        Computes integrated limb darkening profiles for ck2004 atmospheres.
        These are used for intensity-to-flux transformations. The evaluated
        integral is:

        ldint = 2 \pi \int_0^1 Imu mu dmu
        """

        if 'ck2004_all' not in self.content:
            print('Castelli & Kurucz (2004) intensities are not computed yet. Please compute those first.')
            return None

        ldaxes = self._ck2004_intensity_axes
        ldtable = self._ck2004_Imu_energy_grid
        pldtable = self._ck2004_Imu_photon_grid

        self._ck2004_ldint_energy_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))
        self._ck2004_ldint_photon_grid = np.nan*np.ones((len(ldaxes[0]), len(ldaxes[1]), len(ldaxes[2]), 1))

        mu = ldaxes[3]
        Imu = 10**ldtable[:,:,:,:]/10**ldtable[:,:,:,-1:]
        pImu = 10**pldtable[:,:,:,:]/10**pldtable[:,:,:,-1:]

        # To compute the fluxes, we need to evaluate \int_0^1 2pi Imu mu dmu.

        for a in range(len(ldaxes[0])):
            for b in range(len(ldaxes[1])):
                for c in range(len(ldaxes[2])):

                    ldint = 0.0
                    pldint = 0.0
                    for i in range(len(mu)-1):
                        ki = (Imu[a,b,c,i+1]-Imu[a,b,c,i])/(mu[i+1]-mu[i])
                        ni = Imu[a,b,c,i]-ki*mu[i]
                        ldint += ki/3*(mu[i+1]**3-mu[i]**3) + ni/2*(mu[i+1]**2-mu[i]**2)

                        pki = (pImu[a,b,c,i+1]-pImu[a,b,c,i])/(mu[i+1]-mu[i])
                        pni = pImu[a,b,c,i]-pki*mu[i]
                        pldint += pki/3*(mu[i+1]**3-mu[i]**3) + pni/2*(mu[i+1]**2-mu[i]**2)

                    self._ck2004_ldint_energy_grid[a,b,c] = 2*ldint
                    self._ck2004_ldint_photon_grid[a,b,c] = 2*pldint

        self.content.append('ck2004_ldint')

    def interpolate_ck2004_ldcoeffs(self, Teff=5772., logg=4.43, abun=0.0,
                                    atm='ck2004', ld_func='power',
                                    photon_weighted=False):
        """
        Interpolate the passband-stored table of LD model coefficients.

        Arguments
        ------------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ld_func`
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        --------
        * (list or None) list of limb-darkening coefficients or None if 'ck2004_ld'
            is not available in <phoebe.atmospheres.passbands.Passband.content>
            (see also <phoebe.atmospheres.passbands.Passband.compute_ck2004_ldcoeffs>)
            or if `ld_func` is not recognized.
        """
        # TODO: improve documentation for arguments above

        if 'ck2004_ld' not in self.content:
            print('Castelli & Kurucz (2004) limb darkening coefficients are not computed yet. Please compute those first.')
            return None

        if photon_weighted:
            table = self._ck2004_ld_photon_grid
        else:
            table = self._ck2004_ld_energy_grid

        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun),))
            ld_coeffs = libphoebe.interp(req, self._ck2004_intensity_axes[0:3], table)[0]
        else:
            req = np.vstack((Teff, logg, abun)).T
            ld_coeffs = libphoebe.interp(req, self._ck2004_intensity_axes[0:3], table).T

        if ld_func == 'linear':
            return ld_coeffs[0:1]
        elif ld_func == 'logarithmic':
            return ld_coeffs[1:3]
        elif ld_func == 'square_root':
            return ld_coeffs[3:5]
        elif ld_func == 'quadratic':
            return ld_coeffs[5:7]
        elif ld_func == 'power':
            return ld_coeffs[7:11]
        elif ld_func == 'all':
            return ld_coeffs
        else:
            print('ld_func=%s is invalid; please choose from [linear, logarithmic, square_root, quadratic, power, all].')
            return None

    def import_wd_atmcof(self, plfile, atmfile, wdidx, Nabun=19, Nlogg=11,
                        Npb=25, Nints=4):
        """
        Parses WD's atmcof and reads in all Legendre polynomials for the
        given passband.

        Arguments
        -----------
        * `plfile` (string): path and filename of atmcofplanck.dat
        * `atmfile` (string): path and filename of atmcof.dat
        * `wdidx` (int): WD index of the passed passband. This can be automated
            but it's not a high priority.
        * `Nabun` (int, optional, default=19): number of metallicity nodes in
            atmcof.dat. For the 2003 version the number of nodes is 19.
        * `Nlogg` (int, optional, default=11): number of logg nodes in
            atmcof.dat. For the 2003 version the number of nodes is 11.
        * `Nbp` (int, optional, default=25): number of passbands in atmcof.dat.
            For the 2003 version the number of passbands is 25.
        * `Nints` (int, optional, default=4): number of temperature intervals
            (input lines) per entry. For the 2003 version the number of lines
            is 4.
        """

        # Initialize the external atmcof module if necessary:
        # PERHAPS WD_DATA SHOULD BE GLOBAL??
        self.wd_data = libphoebe.wd_readdata(plfile, atmfile)

        # That is all that was necessary for *_extern_planckint() and
        # *_extern_atmx() functions. However, we also want to support
        # circumventing WD subroutines and use WD tables directly. For
        # that, we need to do a bit more work.

        # Store the passband index for use in planckint() and atmx():
        self.extern_wd_idx = wdidx

        # Break up the table along axes and extract a single passband data:
        atmtab = np.reshape(self.wd_data['atm_table'], (Nabun, Npb, Nlogg, Nints, -1))
        atmtab = atmtab[:, wdidx, :, :, :]

        # Finally, reverse the metallicity axis because it is sorted in
        # reverse order in atmcof:
        self.extern_wd_atmx = atmtab[::-1, :, :, :]
        self.content += ['extern_planckint', 'extern_atmx']
        self.atmlist += ['extern_planckint', 'extern_atmx']

    def _log10_Inorm_extern_planckint(self, Teff):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs blackbody approximation.

        @Teff: effective temperature in K

        Returns: log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_planckint(Teff, self.extern_wd_idx, self.wd_data["planck_table"])

        return log10_Inorm

    def _log10_Inorm_extern_atmx(self, Teff, logg, abun):
        """
        Internal function to compute normal passband intensities using
        the external WD machinery that employs model atmospheres and
        ramps.

        Arguments
        ----------
        * `Teff`: effective temperature in K
        * `logg`: surface gravity in cgs
        * `abun`: metallicity in dex, Solar=0.0

        Returns
        ----------
        * log10(Inorm)
        """

        log10_Inorm = libphoebe.wd_atmint(Teff, logg, abun, self.extern_wd_idx, self.wd_data["planck_table"], self.wd_data["atm_table"])

        return log10_Inorm

    def _log10_Inorm_ck2004(self, Teff, logg, abun, photon_weighted=False):
        #~ if not hasattr(Teff, '__iter__'):
            #~ req = np.array(((Teff, logg, abun),))
            #~ log10_Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid)[0][0]
        #~ else:
        req = np.vstack((Teff, logg, abun)).T
        log10_Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid).T[0]

        return log10_Inorm

    def _Inorm_ck2004(self, Teff, logg, abun, photon_weighted=False):
        #~ if not hasattr(Teff, '__iter__'):
            #~ req = np.array(((Teff, logg, abun),))
            #~ log10_Inorm = libphoebe.interp(req, self._ck2004_axes, self._ck2004_photon_grid if photon_weighted else self._ck2004_energy_grid)[0][0]
        #~ else:
        req = np.vstack((Teff, logg, abun)).T
        Inorm = libphoebe.interp(req, self._ck2004_axes, 10**self._ck2004_photon_grid if photon_weighted else 10**self._ck2004_energy_grid).T[0]

        return Inorm

    def _log10_Imu_ck2004(self, Teff, logg, abun, mu, photon_weighted=False):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            log10_Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            log10_Imu = libphoebe.interp(req, self._ck2004_intensity_axes, self._ck2004_Imu_photon_grid if photon_weighted else self._ck2004_Imu_energy_grid).T[0]

        return log10_Imu

    def _Imu_ck2004(self, Teff, logg, abun, mu, photon_weighted=False):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            Imu = libphoebe.interp(req, self._ck2004_intensity_axes, 10**self._ck2004_Imu_photon_grid if photon_weighted else 10**self._ck2004_Imu_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            Imu = libphoebe.interp(req, self._ck2004_intensity_axes, 10**self._ck2004_Imu_photon_grid if photon_weighted else 10**self._ck2004_Imu_energy_grid).T[0]

        return Imu

    def Inorm(self, Teff=5772., logg=4.43, abun=0.0, atm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """

        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ldint` (string, optional, default='ck2004'): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is to
            convert intensity to flux. In this method, however, it is only needed
            for blackbody atmospheres because they are not limb-darkened (i.e.
            the blackbody intensity is the same irrespective of \mu), so we need
            to *divide* by ldint to ascertain the correspondence between
            luminosity, effective temperature and fluxes once limb darkening
            correction is applied at flux integration time. If None, and if
            `atm=='blackbody'`, it will be computed from `ld_func` and
            `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) normal intensities.


        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring

        # convert scalars to vectors if necessary:
        if not hasattr(Teff, '__iter__'):
            Teff = np.array((Teff,))
        if not hasattr(logg, '__iter__'):
            logg = np.array((logg,))
        if not hasattr(abun, '__iter__'):
            abun = np.array((abun,))

        if atm == 'blackbody' and 'blackbody' in self.content:
            if photon_weighted:
                retval = 10**self._log10_Inorm_bb_photon(Teff)
            else:
                retval = 10**self._log10_Inorm_bb_energy(Teff)
            if ldint is None:
                ldint = self.ldint(Teff, logg, abun, atm, ld_func, ld_coeffs, photon_weighted)
            retval /= ldint

        elif atm == 'extern_planckint' and 'extern_planckint' in self.content:
            # -1 below is for cgs -> SI:
            retval = 10**(self._log10_Inorm_extern_planckint(Teff)-1)
            if ldint is None:
                ldint = self.ldint(Teff, logg, abun, atm, ld_func, ld_coeffs, photon_weighted)
            retval /= ldint

        elif atm == 'extern_atmx' and 'extern_atmx' in self.content:
            # -1 below is for cgs -> SI:
            retval = 10**(self._log10_Inorm_extern_atmx(Teff, logg, abun)-1)

        elif atm == 'ck2004' and 'ck2004' in self.content:
            retval = self._Inorm_ck2004(Teff, logg, abun, photon_weighted=photon_weighted)

        else:
            raise NotImplementedError('atm={} not supported by {}:{}'.format(atm, self.pbset, self.pbname))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('atmosphere parameters out of bounds: atm=%s, Teff=%s, logg=%s, abun=%s' % (atm, Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

    def Imu(self, Teff=5772., logg=4.43, abun=0.0, mu=1.0, atm='ck2004', ldint=None, ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ldint` (string, optional, default='ck2004'): integral of the limb
            darkening function, \int_0^1 \mu L(\mu) d\mu. Its general role is to
            convert intensity to flux. In this method, however, it is only needed
            for blackbody atmospheres because they are not limb-darkened (i.e.
            the blackbody intensity is the same irrespective of \mu), so we need
            to *divide* by ldint to ascertain the correspondence between
            luminosity, effective temperature and fluxes once limb darkening
            correction is applied at flux integration time. If None, and if
            `atm=='blackbody'`, it will be computed from `ld_func` and
            `ld_coeffs`.
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) projected intensities.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * ValueError: if `ld_func='interp'` but is not supported by the
            atmosphere table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring

        if ld_func == 'interp':
            # The 'interp' LD function works only for model atmospheres:
            if atm == 'ck2004' and 'ck2004' in self.content:
                retval = self._Imu_ck2004(Teff, logg, abun, mu, photon_weighted=photon_weighted)
                nanmask = np.isnan(retval)
                if np.any(nanmask):
                    raise ValueError('atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s, mu=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask], mu[nanmask]))
                return retval
            else:
                raise ValueError('atm={} not supported by {}:{} ld_func=interp'.format(atm, self.pbset, self.pbname))

        if ld_coeffs is None:
            # LD function can be passed without coefficients; in that
            # case we need to interpolate them from the tables.
            ld_coeffs = self.interpolate_ck2004_ldcoeffs(Teff, logg, abun, atm, ld_func, photon_weighted)

        if ld_func == 'linear':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_lin(mu, *ld_coeffs)
        elif ld_func == 'logarithmic':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_log(mu, *ld_coeffs)
        elif ld_func == 'square_root':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_sqrt(mu, *ld_coeffs)
        elif ld_func == 'quadratic':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_quad(mu, *ld_coeffs)
        elif ld_func == 'power':
            retval = self.Inorm(Teff=Teff, logg=logg, abun=abun, atm=atm, ldint=ldint, ld_func=ld_func, ld_coeffs=ld_coeffs, photon_weighted=photon_weighted) * self._ldlaw_nonlin(mu, *ld_coeffs)
        else:
            raise NotImplementedError('ld_func={} not supported'.format(ld_func))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s, mu=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask], mu[nanmask]))
        return retval

    def _ldint_ck2004(self, Teff, logg, abun, photon_weighted):
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun),))
            ldint = libphoebe.interp(req, self._ck2004_axes, self._ck2004_ldint_photon_grid if photon_weighted else self._ck2004_ldint_energy_grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun)).T
            ldint = libphoebe.interp(req, self._ck2004_axes, self._ck2004_ldint_photon_grid if photon_weighted else self._ck2004_ldint_energy_grid).T[0]

        return ldint

    def ldint(self, Teff=5772., logg=4.43, abun=0.0, atm='ck2004', ld_func='interp', ld_coeffs=None, photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `atm`
        * `ld_func` (string, optional, default='interp') limb darkening
            function.  One of: linear, sqrt, log, quadratic, power, interp.
        * `ld_coeffs` (list, optional, default=None): limb darkening coefficients
            for the corresponding limb darkening function, `ld_func`.
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) ldint.

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * ValueError: if `ld_func='interp'` but is not supported by the
            atmosphere table.
        * NotImplementedError: if `ld_func` is not supported.
        """
        # TODO: improve docstring
        if ld_func == 'interp':
            if atm == 'ck2004':
                retval = self._ldint_ck2004(Teff, logg, abun, photon_weighted=photon_weighted)
            else:
                raise ValueError('atm={} not supported with ld_func=interp'.format(atm))
            nanmask = np.isnan(retval)
            if np.any(nanmask):
                raise ValueError('atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
            return retval

        if ld_coeffs is None:
            ld_coeffs = self.interpolate_ck2004_ldcoeffs(Teff, logg, abun, atm, ld_func, photon_weighted)

        if ld_func == 'linear':
            retval = 1-ld_coeffs[0]/3
        elif ld_func == 'logarithmic':
            retval = 1-ld_coeffs[0]/3+2.*ld_coeffs[1]/9
        elif ld_func == 'square_root':
            retval = 1-ld_coeffs[0]/3-ld_coeffs[1]/5
        elif ld_func == 'quadratic':
            retval = 1-ld_coeffs[0]/3-ld_coeffs[1]/6
        elif ld_func == 'power':
            retval = 1-ld_coeffs[0]/5-ld_coeffs[1]/3-3.*ld_coeffs[2]/7-ld_coeffs[3]/2

        else:
            raise NotImplementedError('ld_func={} not supported'.format(ld_func))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

    def _bindex_ck2004(self, Teff, logg, abun, mu, atm, photon_weighted=False):
        grid = self._ck2004_boosting_photon_grid if photon_weighted else self._ck2004_boosting_energy_grid
        if not hasattr(Teff, '__iter__'):
            req = np.array(((Teff, logg, abun, mu),))
            bindex = libphoebe.interp(req, self._ck2004_intensity_axes, grid)[0][0]
        else:
            req = np.vstack((Teff, logg, abun, mu)).T
            bindex = libphoebe.interp(req, self._ck2004_intensity_axes, grid).T[0]

        return bindex

    def bindex(self, Teff=5772., logg=4.43, abun=0.0, mu=1.0, atm='ck2004', photon_weighted=False):
        """
        Arguments
        ----------
        * `Teff`
        * `logg`
        * `abun`
        * `mu`
        * `atm`
        * `photon_weighted` (bool, optional, default=False): photon/energy switch

        Returns
        ----------
        * (float/array) boosting index

        Raises
        ----------
        * ValueError: if atmosphere parameters are out of bounds for the table.
        * NotImplementedError: if `atm` is not supported (not one of 'ck2004'
            or 'blackbody').
        """
        if atm == 'ck2004':
            retval = self._bindex_ck2004(Teff, logg, abun, mu, atm, photon_weighted)
        elif atm == 'blackbody':
            retval = self._bindex_blackbody(Teff, photon_weighted=photon_weighted)
        else:
            raise NotImplementedError('atm={} not supported'.format(atm))

        nanmask = np.isnan(retval)
        if np.any(nanmask):
            raise ValueError('atmosphere parameters out of bounds: Teff=%s, logg=%s, abun=%s' % (Teff[nanmask], logg[nanmask], abun[nanmask]))
        return retval

def _timestamp_to_dt(timestamp):
    return datetime.strptime(timestamp, "%a %b %d %H:%M:%S %Y")

def _init_passband(fullpath):
    """
    """
    logger.info("initializing passband at {}".format(fullpath))
    pb = Passband.load(fullpath)
    passband = pb.pbset+':'+pb.pbname
    _pbtable[passband] = {'fname': fullpath, 'atms': pb.atmlist, 'timestamp': pb.timestamp, 'pb': None}

    if update_passband_available(passband):
        msg = 'passband "{}" has a newer version available.  Run phoebe.download_passband("{}") or phoebe.update_all_passbands() to update.'.format(passband, passband)
        # NOTE: logger probably not available yet, so we'll also use a print statement
        print('PHOEBE: {}'.format(msg))
        logger.warning(msg)

def _init_passbands(refresh=False):
    """
    This function should be called only once, at import time. It
    traverses the passbands directory and builds a lookup table of
    passband names qualified as 'pbset:pbname' and corresponding files
    and atmosphere content within.
    """
    global _initialized

    if not _initialized or refresh:
        # load information from online passbands first so that any that are
        # available locally will override
        online_passbands = list_online_passbands(full_dict=True, refresh=refresh)
        for pb, info in online_passbands.items():
            _pbtable[pb] = {'fname': None, 'atms': info['atms'], 'pb': None}

        # load global passbands (in install directory) next and then local
        # (in .phoebe directory) second so that local passbands override
        # global passbands whenever there is a name conflict
        for path in list_passband_directories():
            for f in os.listdir(path):
                if f=='README':
                    continue
                if sys.version_info[0] < 3 and f.split('.')[-1] == 'pb3':
                    # then this is a python3 passband but we're in python 2
                    continue
                elif sys.version_info[0] >=3 and f.split('.')[-1] == 'pb':
                    # then this is a python 2 passband but we're in python 3
                    continue
                _init_passband(path+f)

        _initialized = True

def install_passband(fname, local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.install_passband>.

    Install a passband from a local file.  This simply copies the file into the
    install path - but beware that clearing the installation will clear the
    passband as well.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.uninstall_all_passbands>

    Arguments
    ----------
    * `fname` (string) the filename of the local passband.
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    """
    pbdir = _pbdir_local if local else _pbdir_global
    shutil.copy(fname, pbdir)
    _init_passband(os.path.join(pbdir, fname))

def uninstall_all_passbands(local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.uninstall_all_passbands> (only after 2.1.1).

    Uninstall all passbands, either globally or locally (need to call twice to
    delete ALL passbands).

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.install_passband>

    Arguments
    ----------
    * `local` (bool, optional, default=True): whether to uninstall from the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.
    """
    pbdir = _pbdir_local if local else _pbdir_global
    for f in os.listdir(pbdir):
        pbpath = os.path.join(pbdir, f)
        logger.warning("deleting file: {}".format(pbpath))
        os.remove(pbpath)

def download_passband(passband, local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.download_passband>.

    Download and install a given passband from the
    [phoebe2-tables](https://github.com/phoebe-project/phoebe2-tables) repository.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    Arguments
    ----------
    * `passband` (string): name of the passband.  Must be one of the available
        passbands in the repository (see
        <phoebe.atmospheres.passbands.list_online_passbands>).
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.

    Raises
    --------
    * ValueError: if the value of `passband` is not one of
        <phoebe.atmospheres.passbands.list_online_passbands>.
    * IOError: if internet connection fails.
    """
    if passband not in list_online_passbands():
        raise ValueError("passband '{}' not available".format(passband))

    pbdir = _pbdir_local if local else _pbdir_global

    passband_fname = _online_passbands[passband]['fname']
    passband_fname_local = os.path.join(pbdir, passband_fname)
    url = 'http://github.com/phoebe-project/phoebe2-tables/raw/master/passbands/{}'.format(passband_fname)
    logger.info("downloading from {} and installing to {}...".format(url, passband_fname_local))
    try:
        urlretrieve(url, passband_fname_local)
    except IOError:
        raise IOError("unable to download {} passband - check connection".format(passband))
    else:
        _init_passband(passband_fname_local)

def update_passband_available(passband):
    """
    For convenience, this function is available at the top-level as
    <phoebe.update_passband_available>.

    Check if a newer version of a given passband is available from the online repository.

    If so, you can update by calling <phoebe.atmospheres.passbands.download_passband>.

    See also:
    * <phoebe.atmospheres.passbands.list_all_update_passbands_available>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_all_passbands>

    Arguments
    -----------
    * `passband` (string): name of the passband

    Returns
    -----------
    * (bool): whether a newer version is available
    """
    if passband not in list_online_passbands():
        return False

    if _pbtable[passband]['timestamp'] is None:
        if _online_passbands[passband]['timestamp'] is not None:
            return True

    elif _timestamp_to_dt(_pbtable[passband]['timestamp']) < _timestamp_to_dt(_online_passbands[passband]['timestamp']):
        return True

    return False

def list_all_update_passbands_available():
    """
    For convenicence, this function is available at the top-lelve as
    <phoebe.list_all_update_passbands_available>.

    See also:
    * <phoebe.atmospheres.passbands.update_passband_available>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.update_all_passbands>

    Returns
    ----------
    * (list of string): list of passbands with newer versions available online
    """

    return [p for p in list_installed_passbands() if update_passband_available(p)]

def update_all_passbands(local=True):
    """
    For convenience, this function is available at the top-level as
    <phoebe.update_all_passbands>.

    Download and install updates for all passbands from the
    [phoebe2-tables](https://github.com/phoebe-project/phoebe2-tables) repository.

    This will install into the directory dictated by `local`, regardless of the
    location of the original file.  `local`=True passbands always override
    `local=False`.

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    See also:
    * <phoebe.atmospheres.passbands.update_passband_available>


    Arguments
    ----------
    * `local` (bool, optional, default=True): whether to install to the local/user
        directory or the PHOEBE installation directory.  If `local=False`, you
        must have the necessary permissions to write to the installation
        directory.

    Raises
    --------
    * IOError: if internet connection fails.
    """
    for passband in list_all_update_passbands_available():
        download_passband(passband, local=local)

def list_passband_directories():
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_passband_directories>.

    List the global and local passband installation directories (in that order).

    The local and global installation directories can be listed by calling
    <phoebe.atmospheres.passbands.list_passband_directories>.  The local
    (`local=True`) directory is generally at
    `~/.phoebe/atmospheres/tables/passbands`, and the global (`local=False`)
    directory is in the PHOEBE installation directory.

    Returns
    --------
    * (list of strings): global and local passband installation directories.
    """
    return [p for p in [_pbdir_global, _pbdir_local, _pbdir_env] if p is not None]

def list_passbands(refresh=False, full_dict=False):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_passbands>.

    List all available passbands, both installed and available online.

    This is just a combination of
    <phoebe.atmospheres.passbands.list_installed_passbands> and
    <phoebe.atmospheres.passbands.list_online_passbands>.

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.

    Returns
    --------
    * (list of strings or dictionary)
    """
    if full_dict:
        d = list_online_passbands(refresh, True)
        for k in d.keys():
            d[k]['installed'] = False
        # installed passband always overrides online
        for k,v in list_installed_passbands(refresh, True).items():
            d[k] = v
            d[k]['installed'] = True
        return d
    else:
        return list(set(list_installed_passbands(refresh) + list_online_passbands(refresh)))

def list_installed_passbands(refresh=False, full_dict=False):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_installed_passbands>.

    List all installed passbands, both in the local and global directories.

    See also:
    * <phoebe.atmospheres.passbands.list_passband_directories>

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.

    Returns
    --------
    * (list of strings or dictionary)
    """
    if refresh:
        _init_passbands(True)

    if full_dict:
        return {k:v for k,v in _pbtable.items() if v['fname'] is not None}
    else:
        return [k for k,v in _pbtable.items() if v['fname'] is not None]

def list_online_passbands(refresh=False, full_dict=False):
    """
    For convenience, this function is available at the top-level as
    <phoebe.list_online_passbands>.

    List all passbands available for download from the
    [phoebe2-tables](https://github.com/phoebe-project/phoebe2-tables) repository.

    Arguments
    ---------
    * `refresh` (bool, optional, default=False): whether to refresh the list
        of fallback on cached values.  Passing `refresh=True` should only
        be necessary if new passbands have been installed or added to the
        online repository since importing PHOEBE.
    * `full_dict` (bool, optional, default=False): whether to return the full
        dictionary of information about each passband or just the list
        of names.

    Returns
    --------
    * (list of strings or dictionary)
    """
    global _online_passbands
    if os.getenv('PHOEBE_ENABLE_ONLINE_PASSBANDS', 'TRUE').upper() == 'TRUE' and (len(_online_passbands.keys())==0 or refresh):

        branch = 'master'
        branch = 'python3_and_versioning'  # REMOVE ONCE TESTED
        url = 'http://github.com/phoebe-project/phoebe2-tables/raw/{}/passbands/list_online_passbands_full'.format(branch)
        if sys.version_info[0] >= 3:
            url += "_pb3"

        try:
            resp = urlopen(url)
        except URLError:
            url_repo = 'http://github.com/phoebe-project/phoebe2-tables'
            logger.warning("connection to online passbands at {} could not be established".format(url_repo))
            if _online_passbands is not None:
                if full_dict:
                    return _online_passbands
                else:
                    return list(_online_passbands.keys())
            else:
                if full_dict:
                    return {}
                else:
                    return []
        else:
            _online_passbands = json.loads(resp.read(), object_pairs_hook=parse_json)

    if full_dict:
        return _online_passbands
    else:
        return list(_online_passbands.keys())

def get_passband(passband):
    """
    For convenience, this function is available at the top-level as
    <phoebe.get_passbands>.

    Access a passband object by name.  If the passband isn't installed, it`
    will be downloaded and installed locally.

    See also:
    * <phoebe.atmospheres.passbands.list_installed_passbands>
    * <phoebe.atmospheres.passbands.download_passband>
    * <phoebe.atmospheres.passbands.list_passband_directories>

    Arguments
    -----------
    * `passband` (string): name of the passband.  Must be one of the available
        passbands in the repository (see
        <phoebe.atmospheres.passbands.list_online_passbands>).

    Returns
    -----------
    * the passband object

    Raises
    --------
    * ValueError: if the passband cannot be found installed or online.
    * IOError: if needing to download the passband but the connection fails.
    """

    if passband not in list_installed_passbands():
        if passband in list_online_passbands():
            download_passband(passband)
        else:
            raise ValueError("passband: {} not found. Try one of: {} (local) or {} (available for download)".format(passband, list_installed_passbands(), list_online_passbands()))

    if _pbtable[passband]['pb'] is None:
        logger.info("loading {} passband".format(passband))
        pb = Passband.load(_pbtable[passband]['fname'])
        _pbtable[passband]['pb'] = pb

    return _pbtable[passband]['pb']

def Inorm_bol_bb(Teff=5772., logg=4.43, abun=0.0, atm='blackbody', photon_weighted=False):
    """
    Computes normal bolometric intensity using the Stefan-Boltzmann law,
    Inorm_bol_bb = 1/\pi \sigma T^4. If photon-weighted intensity is
    requested, Inorm_bol_bb is multiplied by a conversion factor that
    comes from integrating lambda/hc P(lambda) over all lambda.

    Input parameters mimick the <phoebe.atmospheres.passbands.Passband.Inorm>
    method for calling convenience.

    Arguments
    ------------
    * `Teff` (float/array, optional, default=5772):  value or array of effective
        temperatures.
    * `logg` (float/array, optional, default=4.43): IGNORED, for class
        compatibility only.
    * `abun` (float/array, optional, default=0.0): IGNORED, for class
        compatibility only.
    * `atm` (string, optional, default='blackbody'): atmosphere model, must be
        `'blackbody'`, otherwise exception is raised.
    * `photon_weighted` (bool, optional, default=False): must be `False`,
        otherwise exception is raised.

    Returns
    ---------
    * (float/array) float or array (depending on input types) of normal
        bolometric blackbody intensities.

    Raises
    --------
    * ValueError: if `atm` is anything other than `'blackbody'`.
    """
    # TODO: the docs say errors will be raised if photon_weighted is not False
    # but this doesn't seem to be the case.

    if atm != 'blackbody':
        raise ValueError('atmosphere must be set to blackbody for Inorm_bol_bb.')

    if photon_weighted:
        factor = 2.6814126821264836e22/Teff
    else:
        factor = 1.0

    # convert scalars to vectors if necessary:
    if not hasattr(Teff, '__iter__'):
        Teff = np.array((Teff,))

    return factor * sigma_sb.value * Teff**4 / np.pi

if __name__ == '__main__':

    # Testing LD stuff:
    #~ jV = Passband.load('tables/passbands/johnson_v.pb')
    #~ jV.compute_ck2004_ldcoeffs()
    #~ jV.save('johnson_V.new.pb')
    #~ exit()

    # Constructing a passband:

    #atmdir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables/wd'))
    #wd_data = libphoebe.wd_readdata(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat')

    jV = Passband('tables/ptf/JOHNSON.V', pbset='Johnson', pbname='V', effwl=5500.0, calibrated=True, wlunits=u.AA, reference='ADPS', version=1.0, comments='')
    jV.compute_blackbody_response()
    jV.compute_ck2004_response('tables/ck2004')
    jV.compute_ck2004_intensities('tables/ck2004i')
    jV.import_wd_atmcof(atmdir+'/atmcofplanck.dat', atmdir+'/atmcof.dat', 7)
    jV.save('tables/passbands/JOHNSON.V')

    pb = Passband('tables/ptf/KEPLER.PTF', pbset='Kepler', pbname='mean', effwl=5920.0, calibrated=True, wlunits=u.AA, reference='Bachtell & Peters (2008)', version=1.0, comments='')
    pb.compute_blackbody_response()
    pb.compute_ck2004_response('tables/ck2004')
    pb.save('tables/passbands/KEPLER.PTF')

    #~ jV = Passband.load('tables/passbands/johnson_v.pb')

    #~ teffs = np.arange(5000, 10001, 25)
    #~ req = np.vstack((teffs, 4.43*np.ones(len(teffs)), np.zeros(len(teffs)))).T

    #~ Teff_verts = axes[0][(axes[0] > 4999)&(axes[0]<10001)]
    #~ Inorm_verts1 = grid[(axes[0] >= 4999) & (axes[0] < 10001), axes[1] == 4.5, axes[2] == 0.0, 0]
    #~ Inorm_verts2 = grid[(axes[0] >= 4999) & (axes[0] < 10001), axes[1] == 4.0, axes[2] == 0.0, 0]

    #~ res = libphoebe.interp(req, axes, grid)
    #~ print res.shape

    #~ import matplotlib.pyplot as plt
    #~ plt.plot(teffs, res, 'b-')
    #~ plt.plot(Teff_verts, Inorm_verts1, 'ro')
    #~ plt.plot(Teff_verts, Inorm_verts2, 'go')
    #~ plt.show()
    #~ exit()

    print('blackbody:', jV.Inorm(Teff=5880., logg=4.43, abun=0.0, atm='blackbody', ld_func='linear', ld_coeffs=[0.0,]))
    print('planckint:', jV.Inorm(Teff=5880., logg=4.43, abun=0.0, atm='extern_planckint'))
    print('atmx:     ', jV.Inorm(Teff=5880., logg=4.43, abun=0.0, atm='extern_atmx'))
    print('kurucz:   ', jV.Inorm(Teff=5880., logg=4.43, abun=0.0, atm='ck2004'))

    # Testing arrays:

    print('blackbody:', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), atm='blackbody', ld_func='linear', ld_coeffs=[0.0,]))
    print('planckint:', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), atm='extern_planckint'))
    print('atmx:     ', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), logg=np.array((4.40, 4.43, 4.46)), abun=np.array((0.0, 0.0, 0.0)), atm='extern_atmx'))
    print('kurucz:   ', jV.Inorm(Teff=np.array((5550., 5770., 5990.)), logg=np.array((4.40, 4.43, 4.46)), abun=np.array((0.0, 0.0, 0.0)), atm='kurucz'))
