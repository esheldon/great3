"""
Generic class for fitting, to be inherited
"""
from __future__ import print_function
import numpy

from . import files
from .containers import Field, DeepField

class FitterBase(object):
    def __init__(self, **keys):
        """
        parameters
        ----------
        deep: bool, optional
            If True, load a deep field

        See files.get_file.

        
        In short, experiment, obs_type, shear_type, subid, and optionally the
        epoch.
        """

        self._set_field(**keys)
        self._set_obj_range(**keys)
        self._process_extra_keywords(**keys)

    def go(self):
        """
        run through all the objects to be processed
        """

        t0=time.time()

        last=self.index_list[-1]
        num=len(self.index_list)

        for dindex in xrange(num):
            if self.data['processed'][dindex]==1:
                # was checkpointed
                continue

            index = self.index_list[dindex]
            print('index: %d:%d' % (index,last) )
            self._do_fits(dindex)

            tm=time.time()-t0

            self._try_checkpoint(tm) # only at certain intervals

        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)

    def _do_fits(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        # for checkpointing
        self.data['processed'][dindex]=1

        index = self.index_list[dindex]

        # need to do this because we work on subset files
        self.data['id'][dindex] = self.gal_cat['id'][index]

        psf_res=self._fit_psf(dindex)
        if psf_res['flags'] != 0:
            print("failed to fit psf")
            self.data['flags'][dindex] = psf_res['flags']
            return
        
        gal_res=self._fit_gal(dindex, psf_res)

        self._copy_to_output(dindex, psf_res, gal_res)

    def _fit_psf(self, dindex):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _fit_gal(self, dindex, psf_res):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _copy_to_output(self, dindex, psf_res, gal_res):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")


    def _process_extra_keywords(self, **keys):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _set_obj_range(**keys):
        """
        The range of objects to process. 
        """

        # this is inclusive
        obj_range=keys.get('obj_range',None)
        if obj_range is None:
            obj_range = [0,self.field.get_ngal()-1]

        self.index_list = numpy.arange(obj_range[0],obj_range[1]+1)

    def _set_field(self, **keys):
        deep=keys.get('deep',False)
        if deep:
            self.field = DeepField(**keys)
        else:
            self.field = Field(**keys)


