
# we can do better once we train
_SN2={'exp':None,'dev':None}
def get_shape_noise(type):
    """
    get the shape noise per component
    """
    if _SN2[type] is None:

        import ngmix
        
        if type=='exp':
            g_prior=ngmix.priors.make_gprior_cosmos_exp()
        elif type=='dev':
            g_prior=ngmix.priors.make_gprior_cosmos_dev()
        else:
            raise ValueError("bad shape noise type: '%s'" % type)

        n=100000
        g1,g2 = g_prior.sample2d(n)
        e1=g1.copy()
        e2=g2.copy()
        for i in xrange(n):
            e1[i], e2[i] = ngmix.shape.g1g2_to_e1e2(g1[i],g2[i])
        _SN2[type] = e1.var()

        print("SN2['%s']: %s" % (type,_SN2[type]))

    return _SN2[type]


