"""
Generic class for fitting, to be inherited
"""
from __future__ import print_function
import numpy

from .containers import Field, DeepField

CHECKPOINTS_DEFAULT_MINUTES=[10,30,60,90]

# starting new values for these
DEFVAL      = -9999
PDEFVAL     =  9999
BIG_DEFVAL  = -9.999e9
BIG_PDEFVAL =  9.999e9

# main flags field.  Everything from bit 1-29 is usable by the specific fitters
# but most likely they will have their more specific flag fields

PSF_FIT_FAILURE=2**0
NO_ATTEMPT=2**30


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
        self._set_field_data()
        self._set_obj_range()
        self._finish_setup()
        self._make_struct()

    def go(self):
        """
        run through all the objects to be processed
        """

        t0=time.time()

        last=self.index_list[-1]
        num=len(self.index_list)

        for sub_index in xrange(num):
            # if 1, means we had a checkpoint
            if self.data['processed'][sub_index]==1:
                continue

            index = self.index_list[sub_index]
            print('index: %d:%d' % (index,last) )
            self._do_fits(sub_index)

            tm=time.time()-t0

            # checkpoint after specified intervals
            self._try_checkpoint(tm)

        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)

    def _do_fits(self, sub_index):
        """
        Process the indicated object through the requested fits
        """

        # for checkpointing
        self.data['processed'][sub_index]=1

        # index into main catalog
        index = self.index_list[sub_index]

        # the id from the original catalog
        self.data['id'][sub_index] = self.gal_cat['id'][index]

        res=self._process_object(sub_index)
        self._copy_to_output(sub_index, psf_res, gal_res)

    def _fit_psf(self, sub_index):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _fit_gal(self, sub_index, psf_res):
        """
        over-ride this
        """
        raise RuntimeError("over-ride me")

    def _copy_to_output(self, sub_index, psf_res, gal_res):
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

    def _set_field_data(self):
        deep=self.conf.get('deep',False)
        if deep:
            self.field = DeepField(**self.conf)
            all_skysig = files.read_deep_skynoise_file(**self.conf)
        else:
            self.field = Field(**self.conf)
            all_skysig = files.read_skynoise_file(**self.conf)

        self.skysig = all_skysig['skysig'][self.conf['subid']]
        print("skysig:",self.skysig)

    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data
        """
        self.checkpoints = self.conf.get('checkpoints',CHECKPOINTS_DEFAULT_MINUTES)
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

    def _get_default_dtype(self):
        """
        dtype for fields always part of the output
        """

        dt=[('id','i8'),
            ('processed','i2'),
            ('flags','i4')]

        return dt

    def _make_struct(self):
        """
        Create the output structure
        """
        raise RuntimeError("over-ride me")

def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)


