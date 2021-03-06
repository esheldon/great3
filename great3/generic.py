"""
Generic class for fitting, to be inherited
"""
from __future__ import print_function

import time
import numpy

from . import files
from .containers import Field, DeepField
from .constants import *

class PSFFailure(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)
class GalFailure(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

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

        self.conf={}
        self.conf.update(keys)
        self.conf['psf_flux_min'] = self.conf.get('psf_flux_min',0.0)
        self._setup_checkpoints()
        self._set_field_data()
        self._set_obj_range()
        self._finish_setup()

        if self._checkpoint_data is None:
            self._make_struct()

    def get_data(self):
        """
        Get the data structure
        """
        return self.data

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
            self._do_all_fits(sub_index)

            tm=time.time()-t0

            # checkpoint after specified intervals
            self._try_checkpoint(tm)

        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)

    def _do_all_fits(self, sub_index):
        """
        Process the indicated object through the requested fits

        First some setup, then call out for the actual processing,
        then copy the result
        """

        # for checkpointing
        self.data['processed'][sub_index]=1

        # index into main catalog
        index = self.index_list[sub_index]

        # the id from the original catalog
        self.data['id'][sub_index] = self.field.gal_cat['id'][index]

        # this should set the self.res object
        self._process_object(sub_index)

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

    def _copy_to_output(self, sub_index, res):
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

    def _set_fracdev_prior(self):

        self.fracdev_grid=self.conf.get('fracdev_grid',None)

        prun=self.conf.get('fracdev_prior',None)
        if prun is not None:
            from ngmix.gmix import GMixND

            if 'cgc' in prun:
                experiment="control"
            elif 'rgc' in prun:
                experiment="real_galaxy"
            else:
                raise ValueError("expected rgc or cgc in run")

            partype=self.conf.get('fracdev_prior_type','fracdev')
            pars=files.read_prior(experiment=experiment,
                                  obs_type="ground",
                                  shear_type="constant",
                                  run=prun,
                                  partype=partype,
                                  ext="fits")

            means=pars['means']
            if len(means.shape)==1:
                means=means.reshape(pars['means'].size,1)

            fracdev_prior=GMixND(pars['weights'],
                                 means,
                                 pars['covars'])

            s2n_max=\
                    self.conf.get('fracdev_prior_s2nmax',1.0e9)
            fracdev_range = self.conf.get('fracdev_range',[-2,2])
            fracdev_prior.s2n_max=s2n_max
            # not applied during prior evaluation, just stored
            # here for convenience
            fracdev_prior.fracdev_range=fracdev_range

        else:
            fracdev_prior=None

        self.fracdev_prior=fracdev_prior

    def _set_field_data(self):
        deep=self.conf.get('deep',False)
        if deep:
            self.field = DeepField(**self.conf)
            skysig_struct = files.read_deep_skynoise(**self.conf)
        else:
            self.field = Field(**self.conf)
            skysig_struct = files.read_skynoise(**self.conf)

        self.skysig = skysig_struct['skysig'][0]
        self.sky_ivar = 1.0/self.skysig**2

        print("skysig:",self.skysig)

    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data
        """
        self.checkpoints = self.conf.get('checkpoints',CHECKPOINTS_DEFAULT_MINUTES)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [0]*self.n_checkpoint
        self.checkpoint_file = self.conf.get('checkpoint_file',None)
        print("checkpoint file:",self.checkpoint_file)

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
            self.data=self._checkpoint_data

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

    def _write_checkpoint(self, tm):
        """
        Write out the current data structure to a temporary
        checkpoint file.
        """
        import fitsio

        print('checkpointing at',tm/60,'minutes')
        print(self.checkpoint_file)

        with fitsio.FITS(self.checkpoint_file,'rw',clobber=True) as fobj:
            fobj.write(self.data, extname="model_fits")


    def _get_default_dtype(self):
        """
        dtype for fields always part of the output
        """

        dt=[('id','i8'),
            ('processed','i2'),
            ('flags','i4'),
            ('psf_g','f8',2), # always fit 1 gauss with em to psf
            ('psf_T','f8'),
            ('psf_flux','f8'),
            ('psf_flux_err','f8')]

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

class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None):
        self.front=front
    def __call__(self, name):
        if self.front is None or self.front=='':
            return name
        else:
            return '%s_%s' % (self.front, name)


'''
t['com_T']
t['com_e']
t['com_flux']

t['comp_T']
t['comp_e']
t['comp_flux']

t['c_T']
t['c_e']
t['c_flux']

t['cmodel_T']
t['cmodel_e']
t['cmodel_flux']

t['cm_T']
t['cm_e']
t['cm_flux']

t['cmod_T']
t['cmod_e']
t['cmod_flux']

'''
