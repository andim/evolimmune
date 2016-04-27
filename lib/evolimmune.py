import numbers
import numpy as np
import misc

varname_to_tex = misc.DefaultIdentityDict(pi=r'$\pi$', aenv=r'$a_{\rm env}$',
        tauenv=r'$\tau_{\rm env}$', pienv=r'$\pi_{\rm env}$', epsilon=r'$\epsilon$',
        pup=r'$p_{\rm uptake}$', cconstitutive=r'c_{\rm constitutive}',
        cdefense=r'$c_{\rm defense}$', cinfection=r'$c_{\rm infection}$',
        cup=r'$c_{\rm uptake}$')

## calculate long term growth rates
try:
    from cevolimmune import zrecursion, sumlog
    print 'use czrecursion, cztogrowthrate'

    def zstogrowthrate(zs, nburnin=1):
        nburnin = int(nburnin)
        data = zs[nburnin:]
        return sumlog(data)/data.shape[0]

except:
    def zrecursion(lambda_, mu, p, q, alpha, beta, niter=1e3, prng=np.random):
        niter = int(niter)
        r = 0.5
        # set x to 1 with probability pi
        x = (alpha/(alpha+beta)) < prng.rand()
        elambda = np.exp(-lambda_)
        emu = np.exp(-mu)
        prob = prng.rand(niter)
        zs = np.zeros(niter)
        for i in range(niter):
            if x:
                x = prob[i] > beta
            else:
                x = prob[i] < alpha
            elambdax = elambda if x else 1
            z = (1.0-r)*elambdax + r*emu
            r = ((1.0-r)*p*elambdax + r*(1.0-q)*emu)/z
            zs[i] = z
        return zs

    def zstogrowthrate(zs, nburnin=1):
        nburnin = int(nburnin)
        return np.mean(np.log(zs[nburnin:]))

def Lambda_pq(x, lambda_, mus, cup, aenv, pienv, niter, nburnin, seed=None):
    """Calculates the long-term population growth rate numerically

    Parameters
    ----------
    mus: called as mus(epsilon) -> mu1, mu2
        constitutive and defense cost as a parametric function
        epsilon = 0 means no regulation, epsilon = 1 maximal regulation
        if a string then will be converted using `mus_from_str`
    cup: called as cup(pup) -> cup
        cost of active acquisition
        if a string then will be converted using `cup_from_str`
    niter: number of iterations of the recursion equation
    nburnin: number of initial iterations to ignore for calculation of mean
    seed: seed to use for random number generator
    """
    if isinstance(mus, basestring):
        mus = mus_from_str(mus)
    if isinstance(cup, basestring):
        cup = cup_from_str(cup)
    niter = int(niter)
    p, q, epsilon, pup = x
    alpha, beta = from_api(aenv, pienv)
    mu1, mu2 = mus(epsilon)
    return zstogrowthrate(zrecursion(lambda_, mu1, p, q, alpha, beta, pup=pup, niter=niter, mu2=mu2, seed=seed), nburnin=nburnin) - cup(pup)

def piopt_aenv0(pienv, lambda_, mu):
    return np.clip(((np.exp(lambda_)-1.0)*pienv - (np.exp(mu)-1.0)) / ((np.exp(lambda_-mu)-1.0)*(np.exp(mu)-1.0)), 0, 1)

## parameter conversions

def mus_from_str(musstr):
    funcstr='''\
def mus(epsilon):
    return {e}
    '''.format(e=musstr)
    exec(funcstr)
    return mus

def cup_from_str(cupstr):
    funcstr='''\
def cup(pup):
    return {e}
    '''.format(e=cupstr)
    exec(funcstr)
    return cup

def from_api(a, pi):
    "p, q of two state Markov chain from a, pi"
    return (1-a)*pi, (1-a)*(1-pi)

def to_api(p, q):
    "a, pi of two state Markov chain from p, q"
    # if q is not defined then p is 0 and pi 1 (and vice versa)
    return 1-p-q, np.where(p, np.where(q, p/(p+q), 1), 0)

def to_tau(a):
    "characteristic time tau from second eigenvalue a"
    try:
        return 1.0 / np.log(1.0/a)
    except ZeroDivisionError:
        return 0.0

def from_tau(tau):
    "a from tau"
    tau = np.asarray(tau)
    return np.exp(-1.0/tau)

def derived_quantities(df):
    if 'epsilonse' in df.columns:
        derived_quantities_errprop(df)
        return
    if 'pi' in df.columns:
        df['p'], df['q'] = from_api(df.a, df.pi)
    elif 'p' in df.columns:
        df['a'], df['pi'] = to_api(df.p, df.q)
    if 'epsilon' in df.columns:
        df['cconstitutive'] = df.apply(lambda row: mus_from_str(row['mus'])(row['epsilon'])[0],
                                       axis=1)
        df['cdefense'] = df.apply(lambda row: mus_from_str(row['mus'])(row['epsilon'])[1],
                                  axis=1)
    if 'pup' in df.columns:
        df['a1'] = 1.0-df.p-df.q-df.pup
        df['tau1'] = to_tau(df['a1'])
    if 'a' in df.columns:
        df['tau'] = to_tau(df['a'])
    df['tauenv'] = to_tau(df['aenv'])

## agent-based simulations
try:
    from cevolimmune import stepmarkov, stepmarkov2d, stepcrispr
    usecstepmarkov = True
    print 'use cstepmarkov'
except:
    def stepmarkov(x, a, b, rand):
        return (x & (rand < 1-b)) | (~x & (rand < a))
    stepmarkov2d = stepmarkov
    usecstepmarkov = False

def _arrayify(x, shape):
    # Duck typing test for whether x is iterable
    try:
        iter(x)
        return np.asarray(x)
    except:
        return x * np.ones(shape)

def agentbasedsim(L, a, pi, aenv, pienv, xi,
        adev=1.0, pidev=0.5,
        nind=10, ngeneration=100, nburnin=10,
        prng=None,
        callback=None):
    """ Agent based simulation of a population of individuals with different immune systems.
    
        adev, pidev: parameter of expression stochasticity (development)
                     if adev = 1.0, there is no stochasticity and the equations are simplified 
        callback(gen, env): function called at every iteration
    """

    p, q = from_api(a, pi)
    alpha, beta = from_api(aenv, pienv)
    if not adev == 1.0:
        delta, epsilon = from_api(adev, pidev)

    # all parameters need to be in array form if cython acceleration is used
    if usecstepmarkov:
        alpha = _arrayify(alpha, L)
        beta = _arrayify(beta, L)
        p = _arrayify(p, (nind, L))
        q = _arrayify(q, (nind, L))
        if not adev == 1.0:
            delta = _arrayify(delta, (nind, L))
            epsilon = _arrayify(epsilon, (nind, L))
        
    env = np.zeros(L, dtype = bool)
    gen = np.zeros((nind, L), dtype = bool)
    
    totoffsprings = np.zeros(ngeneration)
    prng = prng if prng else np.random
    
    for generation in range(ngeneration):
        # time step environment
        rand = prng.rand(L)
        env = stepmarkov(env, alpha, beta, rand)
        if callback and generation >= nburnin:
            callback(gen, env)
        if not adev == 1.0:
            rand = prng.rand(nind, L)
            phen = stepmarkov2d(gen, delta, epsilon, rand)
        else:
            phen = gen
        # calculate growth rate
        noffspring = xi(phen, env)
        totoffspring = noffspring.sum()
        totoffsprings[generation] = totoffspring
        # time step population
        rand = prng.rand(nind, L)
        parent = gen[np.arange(nind).repeat(prng.multinomial(nind, noffspring/totoffspring))]
        gen = stepmarkov2d(parent, p, q, rand)
   
    # calculate Lambda = mean growth rate
    return np.mean(np.log(totoffsprings[nburnin:]/nind))

def agentbasedsim_evol(aenv, pienv, xi,
        L=1, nind=10, ngens=[100],
        mutrate=1e-4, mutsize=0.025,
        ainit=None, piinit=None, pupinit=None, epsiloninit=None,
        evolvep=True,
        evolveq=True,
        evolvepup=False,
        evolveepsilon=False,
        prng=None):
    """ Agent based simulation of a population of individuals with different immune systems and evolving parameters.
    xi(ind, env): function to calculate growth rate of individuals ind in environment env
                  called as xi(ind, env, nacquired) if pupinit != 0.0
    pupinit: initial uptake probability for crispr (if 0.0 then it will not evolve)
    mutrate: can also supply function mutrate(generation)
    mutsize: can also supply function mutsize(generation)
    """

    prng = prng if prng else np.random

    mutrateisnumber = isinstance(mutrate, numbers.Number) 
    mutsizeisnumber = isinstance(mutsize, numbers.Number) 
    
    alpha, beta = from_api(aenv, pienv)

    # all parameters need to be in array form if cython acceleration is used
    if usecstepmarkov:
        alpha = _arrayify(alpha, L)
        beta = _arrayify(beta, L)

    # p,q can now change dynamically.
    if (ainit is not None) and (piinit is not None) and (pupinit is not None) and (epsiloninit is not None):
        # Initialize all to the same value
        pinit, qinit = from_api(ainit, piinit)
        p = np.clip(np.ones((nind, L)) * pinit, 0, 1)
        q = np.clip(np.ones((nind, L)) * qinit, 0, 1)
        pup = np.clip(np.ones((nind, L)) * pupinit, 0, 1)
        epsilon = np.clip(np.ones(nind) * epsiloninit, 0, 1)
    else:
        # Initialize uniformly
        p = prng.random((nind, L))
        q = prng.random((nind, L))
        pup = prng.random((nind, L))
        epsilon = prng.random(nind)
        
    # Initialize all to the same value, then add randomness
    #p = np.clip(np.ones((nind, L)) * pinit + np.random.normal(scale = mutsizeinit, size = (nind, L)), 0, 1)
    #q = np.clip(np.ones((nind, L)) * qinit + np.random.normal(scale = mutsizeinit, size = (nind, L)), 0, 1)
    # Initialize uniformly in a, pi
    #p, q = from_api(prng.random((nind, L)), prng.random((nind, L)))

    env = np.zeros(L, dtype = bool)
    ind = np.zeros((nind, L), dtype = bool)

    totoffsprings = np.zeros(max(ngens))
    as_ = np.zeros((len(ngens), L))
    pis = np.zeros((len(ngens), L))
    pups = np.zeros((len(ngens), L))
    epsilons = np.zeros(len(ngens))

    # number of generations at which to output
    ngencounter = 0
    # for performance: avoid array look up at each step
    ngennext = ngens[0]-1

    # for performance: precomputations
    # (checked if actually saving significant amounts of time)
    nind_arange = np.arange(nind)
    if mutrateisnumber:
        totmutrate = mutrate * L * nind
    # if mutsize is constant then assign it for all generations
    if mutsizeisnumber:
        mutsize_gen = mutsize

    for generation in range(max(ngens)):
        # time step environment
        rand = prng.rand(L)
        env = stepmarkov(env, alpha, beta, rand)
        # acquire via crispr
        rand = prng.rand(nind, L)
        ind, nacquired = stepcrispr(env, ind, pup, rand)
        # calculate growth rate
        noffspring = xi(ind, env, epsilon, pup)

        # for performance: use np.add.reduce directly to avoid np.sum overhead
        #totoffspring = noffspring.sum()
        totoffspring = np.add.reduce(noffspring)
        # time step population
        indoffspring = nind_arange.repeat(prng.multinomial(nind, noffspring/totoffspring))
        parent = ind.take(indoffspring, axis=0)
        rand = prng.rand(nind, L)
        ind = stepmarkov2d(parent, p, q, rand)
        # inherit strategies
        p = p.take(indoffspring, axis=0)
        q = q.take(indoffspring, axis=0)
        pup = pup.take(indoffspring, axis=0)
        epsilon = epsilon.take(indoffspring, axis=0)
        # mutate strategies
        if not mutrateisnumber:
            totmutrate = mutrate(generation) * L * nind
        if not mutsizeisnumber:
            mutsize_gen = mutsize(generation)
        for c in range(prng.poisson(totmutrate)):
            # pick random site 
            i, j = prng.randint(nind), prng.randint(L)
            if evolvep:
                ptmp = p[i, j] + prng.normal(scale=mutsize_gen)
                # ensure that there is at least some switching
                # needed for sensible definition of pi
                while (ptmp <= 0.0) or (ptmp > 1):
                    ptmp = p[i, j] + prng.normal(scale=mutsize_gen)
                p[i, j] = ptmp
            if evolveq:
                qtmp = q[i, j] + prng.normal(scale=mutsize_gen)
                while (qtmp <= 0.0) or (qtmp > 1):
                    qtmp = q[i, j] + prng.normal(scale=mutsize_gen)
                q[i, j] = qtmp
            if evolvepup:
                puptmp = pup[i, j] + prng.normal(scale=mutsize_gen)
                while (puptmp <= 0.0) or (puptmp > 1):
                    puptmp = pup[i, j] + prng.normal(scale=mutsize_gen)
                pup[i, j] = puptmp 
            if evolveepsilon:
                epsilontmp = epsilon[i] + prng.normal(scale=mutsize_gen)
                while (epsilontmp <= 0.0) or (epsilontmp > 1):
                    epsilontmp = epsilon[i] + prng.normal(scale=mutsize_gen)
                epsilon[i] = epsilontmp 

                    
        # store data
        totoffsprings[generation] = totoffspring
        if generation == ngennext:
            a, pi = to_api(p, q)
            as_[ngencounter] = a.mean(axis=0)
            pis[ngencounter] = pi.mean(axis=0)
            if evolvepup:
                pups[ngencounter] = pup.mean(axis=0)
            if evolveepsilon:
                epsilons[ngencounter] = epsilon.mean(axis=0)
            ngencounter += 1
            if ngencounter >= len(ngens):
                break
            ngennext = ngens[ngencounter]-1

    return totoffsprings / nind, pis, as_, pups, epsilons

## calculate immune phases
try:
    import analysis
    import shapely.geometry.polygon

    def polygons_from_boundaries(df, yconv=None, boundfactor=1.0):
        """Make shapely polygons of left side of boundary.
        Parameters
        ----------
        yconv : transformation to be applied to y coordinates 
        boundfactor : factor by which to enlarge bound tolerance
        """
        if yconv is None:
            yconv = lambda x: x
        aenvmin = df.aenv.min()
        aenvmax = df.aenv.max()
        complete = shapely.geometry.Polygon([(0.0, yconv(aenvmin)),
                                             (0.0, yconv(aenvmax)),
                                             (1.0, yconv(aenvmax)),
                                             (1.0, yconv(aenvmin))])
        polygons = dict(complete=complete)

        # for every boundary make a polygon extrapolating as necessary
        xtol = df['xtolbound'].mean() * boundfactor
        for boundary, dfg in df.groupby('boundary'):
            pienvbnd, aenv = np.asarray(dfg.pienvbnd), np.asarray(dfg.aenv)
            vertices = analysis.polygon_from_boundary(pienvbnd, aenv, ymin=aenvmin, ymax=aenvmax, xtol=xtol)
            if yconv is not None:
                vertices[:, 1] = yconv(vertices[:, 1])

            # make polygon from vertices and store in dict
            p = shapely.geometry.Polygon(vertices)
            polygons[boundary] = p
        return polygons

    def phases_from_polygons(polygons):
        # use individual error handling so that if there is an issue with some
        # phases the others still get calculated
        errmsg = 'problem generating phase {}'
        try:
            c = polygons['cm']-polygons['ac']-polygons['pc']
        except:
            c = None
            print(errmsg.format('c'))
        try:
            a = polygons['ac'].intersection(polygons['ap'])
        except:
            a = None
            print(errmsg.format('a'))
        try:
            p = analysis.cascaded_intersection([polygons['pi'], polygons['pm'],
                                                polygons['po'], polygons['pc']])-polygons['ap']
        except:
            p = None
            print(errmsg.format('p'))
        try:
            o = polygons['complete']-polygons['io']-polygons['po']
        except:
            o = None
            print(errmsg.format('o'))
        try:
            i = polygons['io']-polygons['mi']-polygons['pi']
        except:
            i = None
            print(errmsg.format('i'))
        try:
            m = polygons['mi']-polygons['pm']-polygons['cm']
        except:
            m = None
            print(errmsg.format('m'))
        return dict(a=a, p=p, o=o, i=i, m=m, c=c)
except ImportError:
    pass

