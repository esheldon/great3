from __future__ import print_function
from . import files
from numpy import array

import ngmix
from ngmix.joint_prior import JointPriorTF, JointPriorSimpleHybrid


def make_joint_prior_simple(run,
                            partype, # e.g. hybrid, hybrid-pflux-1.304
                            cen_width,
                            g_prior_during=True,
                            g_prior_type='great-des',
                            g_prior_pars=None,
                            with_TF_bounds=True):
    """
    Make a joint prior

    g_prior_during=False to just use a simple ZDisk2D
    """

    if 'rgc' in run:
        """
        e.g. g302-rgc-deep02

        this GPriorGreatDES is very noisy for recovering shear, might want to
        use essentially *anything* else, even BA (should test in my sims)

        """

        t=files.read_prior(experiment="real_galaxy",
                            obs_type="ground",
                            shear_type="constant",
                            run=run,
                            partype=partype,
                            ext="fits")

        TF_prior=JointPriorTF(t['weights'],
                              t['means'],
                              t['covars'])

        cen_prior=ngmix.priors.CenPrior(0.0, 0.0, cen_width, cen_width)

        if g_prior_during:
            g_prior = get_g_prior(g_prior_type, g_prior_pars)
        else:
            g_prior = ngmix.priors.ZDisk2D(1.0)

        p=JointPriorSimpleHybrid(cen_prior, g_prior, TF_prior)

    elif 'cgc' in run:
        """
        e.g. rg302-cgc-deep01
        """

        t=files.read_prior(experiment="control",
                            obs_type="ground",
                            shear_type="constant",
                            run=run,
                            partype=partype,
                            ext="fits")

        TF_prior=JointPriorTF(t['weights'],
                              t['means'],
                              t['covars'])

        cen_prior=ngmix.priors.CenPrior(0.0, 0.0, cen_width, cen_width)

        if g_prior_during:
            g_prior = get_g_prior(g_prior_type, g_prior_pars)
        else:
            g_prior = ngmix.priors.ZDisk2D(1.0)

        p=JointPriorSimpleHybrid(cen_prior, g_prior, TF_prior)

    else:
        raise ValueError("support different priors")
       
    '''
    elif type=="great3-rgc-exp-hybrid-cosmosg-deep03":
        # pretending we can separate out the shape prior
        t=files.read_prior(experiment="real_galaxy",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-rgc-deep03",
                           partype="hybrid",
                           ext="fits")

        if with_TF_bounds:
            print("using TF bounds")
            logT_bounds=[-1.5, 0.5]
            logF_bounds=[-0.7, 1.5]
        else:
            print("without TF bounds")
            logT_bounds=None
            logF_bounds=None
        TF_prior=JointPriorTF(t['weights'],
                              t['means'],
                              t['covars'],
                              T_bounds=logT_bounds,
                              F_bounds=logF_bounds)

        cen_prior=ngmix.priors.CenPrior(0.0, 0.0, cen_width, cen_width)

        if g_prior_during:
            g_prior = ngmix.priors.make_gprior_cosmos_sersic(type='erf')
        else:
            g_prior = ngmix.priors.ZDisk2D(1.0)

        p=JointPriorSimpleHybrid(cen_prior,
                                 g_prior,
                                 TF_prior)

    elif type=="great3-rgc-exp-hybrid-cosmosg-deep04":
        raise RuntimeError("adapt to new system")
        # pretending we can separate out the shape prior
        # this one used prior on g from cosmos during fitting of
        # deep fields
        t=files.read_prior(experiment="real_galaxy",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-rgc-deep04",
                           partype="hybrid",
                           ext="fits")

        g_prior = ngmix.priors.make_gprior_cosmos_sersic(type='erf')
        p=JointPriorSimpleHybrid(t['weights'],
                                 t['means'],
                                 t['covars'],
                                 g_prior)

      
    elif type == "great3-real_galaxy-ground-constant-exp-hybrid-deep03":
        raise RuntimeError("adapt to new system")
        # pretending we can separate out the shape prior
        t=files.read_prior(experiment="real_galaxy",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-rgc-deep03",
                           partype="hybrid",
                           ext="fits")

        g_prior = ngmix.priors.make_gprior_great3_exp()
        p=JointPriorSimpleHybrid(t['weights'],
                                 t['means'],
                                 t['covars'],
                                 g_prior)


    elif type == "great3-real_galaxy-ground-constant-exp-logpars":
        raise RuntimeError("adapt to new system")
        from ngmix.joint_prior import JointPriorSimpleLogPars
        t=files.read_prior(experiment="real_galaxy",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-rgc-deep01",
                           partype="logpars",
                           ext="fits")

        p=JointPriorSimpleLogPars(t['weights'],
                                  t['means'],
                                  t['covars'])
    elif type == "great3-real_galaxy-ground-constant-exp-linpars":
        raise RuntimeError("adapt to new system")
        from ngmix.joint_prior import JointPriorSimpleLinPars
        t=files.read_prior(experiment="real_galaxy",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-rgc-deep01",
                           partype="linpars",
                           ext="fits")

        T_bounds=[0.01,1.5]
        Flux_bounds=[0.01,10.0]
        #T_bounds=[0.15,0.6]
        #Flux_bounds=[0.75,4.0]
        print("T bounds:",T_bounds)
        print("Flux bounds:",Flux_bounds)
        p=JointPriorSimpleLinPars(t['weights'],
                                  t['means'],
                                  t['covars'],
                                  T_bounds,
                                  Flux_bounds)

    else:
        raise ValueError("bad type: '%s'" % type)
    '''

    return p

def get_g_prior(g_prior_type, g_prior_pars=None):
    print("loading g prior:",g_prior_type)

    if g_prior_type =='ba':

        if g_prior_pars is None:
            g_prior_pars=0.3

        print("g prior pars:",g_prior_pars)
        g_prior = ngmix.priors.GPriorBA(g_prior_pars)

    elif g_prior_type=='great-des':

        if g_prior_pars is None:
            g_prior_pars = [1.0, 6680.0, 0.0509, 0.733]

        print("g prior pars:",g_prior_pars)
        g_prior = ngmix.priors.GPriorGreatDES(pars=g_prior_pars, gmax=1.0)

    elif g_prior_type=='m-erf':

        assert g_prior_pars != None,"send pars for merf"
        print("g prior pars:",g_prior_pars)
        g_prior = ngmix.priors.GPriorMErf(pars=g_prior_pars)

    else:
        raise ValueError("bad g_prior_type: '%s'" % (g_prior_type))

    return g_prior


def make_joint_prior_sersic(type="great3-cgc-sersic-hybrid-deep01"):
    raise RuntimeError("adapt to new system")
    if type=="great3-cgc-sersic-hybrid-deep01":

        # pretending we can separate out the shape prior
        from ngmix.joint_prior import JointPriorSersicHybrid
        t=files.read_prior(experiment="control",
                           obs_type="ground",
                           shear_type="constant",
                           run="nfit-cgc-deep01",
                           partype="hybrid",
                           ext="fits")

        g_prior=ngmix.priors.make_gprior_great3_sersic_cgc()

        p=JointPriorSersicHybrid(t['weights'],
                                 t['means'],
                                 t['covars'],
                                 g_prior)


    else:
        raise ValueError("bad type: '%s'" % type)

    return p


class Tester(object):
    def __init__(self, jp):
        self.jp=jp
    def get_lnprob(self, pars):
        return self.jp.get_lnprob_scalar1d(pars, throw=False)

def test_random(nrand=40000, type="great3-real_galaxy-ground-constant-exp-logpars"):
    """
    Make sure our lnprob function properly matches the samples
    drawn from the mixture
    """
    import great3
    import emcee

    jp=make_joint_prior_simple(type=type)
    tester=Tester(jp)

    print("getting random samples")
    rsamp=jp.sample1d(nrand)

    print("getting mcmc samples")
    nwalkers=80
    npars=rsamp.shape[1]
    burnin=800
    nstep=10000
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    npars, 
                                    tester.get_lnprob)

    guess=rsamp[0:nwalkers,:]
    print("burnin per walker",burnin)
    pos, prob, state = sampler.run_mcmc(guess, burnin)
    sampler.reset()
    print("steps per walker",nstep)
    pos, prob, state = sampler.run_mcmc(pos, nstep)


    mcmc_rsamp = sampler.flatchain

    if 'log' in type:
        dolog=True
    else:
        dolog=False
    great3.fit_prior.plot_fits(rsamp, mcmc_rsamp, dolog=dolog, show=True)
