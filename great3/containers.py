"""
Containers for great3 data for easier use
"""
from __future__ import print_function
import numpy

from . import files

class Field(object):
    def __init__(self, **keys):
        """
        parameters
        ----------
        See files.get_file.
        
        In short, experiment, obs_type, shear_type, subid, and optionally the
        epoch.

        the gal image and cat, and star image and cat are read.
        """

        self.load_data(**keys)

    def get_ngal(self):
        """
        get the number of galaxies
        """
        return self.gal_cat.size

    def get_nstar(self):
        """
        get the number of galaxies
        """
        return self.star_cat.size

    def get_gal_image(self, index):
        """
        Get the cutout centered on the indicated galaxy.
        """
        return self.get_image(index, 'gal')

    def get_star_image(self, index):
        """
        Get the cutout centered on the indicated galaxy.
        """
        return self.get_image(index, 'star')


    def get_image(self, index, type):
        """
        Get the cutout for the indicated index and object type.
        """
        
        if type=='gal':
            cat = self.gal_cat
            im  = self.gal_image
        else:
            cat = self.star_cat
            im  = self.star_image

        row = cat['y'][index] 
        col = cat['x'][index] 
        row_low  = row - 23
        row_high = row + 25
        col_low  = col - 23
        col_high = col + 25

        cutout = im[row_low:row_high, col_low:col_high].astype('f8')

        cen=[23.5, 23.5]
        return cutout, cen

    def load_data(self, **keys):
        """
        Load 
            - gal image
            - gal catalog
            - star image
            - star catalog
        """

        self.gal_image  = files.read_gal_image(**keys)
        self.gal_cat    = files.read_gal_cat(**keys)
        self.star_image = files.read_star_image(**keys)
        self.star_cat   = files.read_star_cat(**keys)
