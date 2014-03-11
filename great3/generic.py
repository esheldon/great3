"""
Generic class for fitting, to be inherited
"""
from __future__ import print_function
import numpy

from . import files
from .containers import Field, DeepField

_CHECKPOINTS_DEFAULT_MINUTES=[10,30,60,90]

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

        self.conf=keys
        self._set_field()
        self._set_obj_range()
        self._finish_setup()

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


    def _finish_setup(self):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _set_obj_range(self):
        """
        The range of objects to process. 
        """

        # this is inclusive
        obj_range=self.conf.get('obj_range',None)
        if obj_range is None:
            obj_range = [0,self.field.get_ngal()-1]

        self.index_list = numpy.arange(obj_range[0],obj_range[1]+1)

    def _set_field(self):
        deep=self.conf.get('deep',False)
        if deep:
            self.field = DeepField(**self.conf)
        else:
            self.field = Field(**self.conf)

    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data
        """
        self.checkpoints = self.conf.get('checkpoints',_CHECKPOINTS_DEFAULT_MINUTES)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [0]*self.n_checkpoint
        self.checkpoint_file = self.conf.get('checkpoint_file',None)

        self._set_checkpoint_data()

        if self.checkpoint_file is not None:
            self.do_checkpoint=True
        else:
            self.do_checkpoint=False

    def _set_checkpoint_data(self):
        """
        See if checkpoint data was sent
        """
        self._checkpoint_data=self.conf.get('checkpoint_data',None)
        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data['data']

    def _try_checkpoint(self, tm):
        """
        Checkpoint at certain intervals.  
        Potentially modified self.checkpointed
        """

        should_checkpoint, icheck = self._should_checkpoint(tm)

        if should_checkpoint:
            self._write_checkpoint(tm)
            self.checkpointed[icheck]=1

    def _should_checkpoint(self, tm):
        """
        Should we write a checkpoint file?
        """

        should_checkpoint=False
        icheck=-1

        if self.do_checkpoint:
            tm_minutes=tm/60

            for i in xrange(self.n_checkpoint):

                checkpoint=self.checkpoints[i]
                checkpointed=self.checkpointed[i]

                if tm_minutes > checkpoint and not checkpointed:
                    should_checkpoint=True
                    icheck=i

        return should_checkpoint, icheck


