from scipy.spatial.transform import Rotation as R
from scipy import stats
import matplotlib.pyplot as plt
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import Moments
import numpy as np


class TrajectoryDist(dists.ProbDist):
    """Trajectory probability distribution.
    The particles library requires that we provide a method that returns P(observations|particle).
    I'm not sure how to do that, so instead, this returns P(particle|observations) and I'm hoping that
    this is roughly proportional in some sense (i.e. we p(Y) and p(X) are constant!). This class' constructor
    takes two parameters:

       loc = the location of the particle
       scale = the standard deviation of the Normal distribution around the vector of the spotted bee

    The logpdf method takes an observation, which consists of 6 numbers (the position and velocity vectors).
    It computes the distance to the particle specified in loc and uses the distance as the input to a
    zero-centred normal distribution, with standard deviation specified by the scale parameter.
    """
    dim = 6  # dimension of variates [x,y,z,vx,vy,vz]
    dtype = 'float64'

    def getdist(self, origin, vect, points):
        return np.linalg.norm(np.cross(vect, points - origin), axis=1) / np.linalg.norm(vect)

    def logpdf(self, x):
        """
        the log-density for a camera/spot 6-dim vector (first three=camera location, last three=vector to spot)
        """
        distance = self.getdist(x[:3], x[3:6], self.loc)
        pnorm = stats.norm.logpdf(distance, loc=0, scale=self.scale)
        pflat = stats.uniform.logpdf(distance, loc=-15, scale=30)
        return np.logaddexp(pnorm, pflat) - np.log(2)

        # return stats.norm.logpdf(self.getdist(x[:3],x[3:6],self.loc), loc=0, scale=self.scale)

    def rvs(self, size=None):
        """
        generates *size* variates from distribution
        """
        raise NotImplementedError

    def ppf(self, u):
        """
        the inverse CDF function at point u
        """
        raise NotImplementedError

    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = scale


class BeeTrajectoryNoVelocity(ssm.StateSpaceModel):
    """
    based on https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/notebooks/basic_tutorial.html

    The particles in this model have a location ONLY.
    The update adds normally distributed noise to the location.
    The PY method uses the TrajectoryDist distribution to get
    a distribution over possible vectors and cameras for each particle.
    """

    def PX0(self):
        """Distribution of X_0 (initial particle locations)"""
        cov = np.zeros([3, 3])
        cov[:3, :3] = np.eye(3) * 0.5  # location uncertainty
        if hasattr(self, 'startpoint'):
            start = self.startpoint
        else:
            start = np.zeros(3)
        return dists.MvNormal(loc=start, cov=cov)

    def PX(self, t, xp):
        """Distribution of X_t given X_{t-1}=xp (p=past)"""
        cov = np.zeros([3, 3])
        cov[:3, :3] = np.eye(3) * 2.0  # location uncertainty
        timestep = self.times[t] - self.times[t - 1] + 0.00001
        return dists.MvNormal(loc=xp, cov=cov, scale=timestep)

    def PY(self, t, xp, x):
        """Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)"""
        return TrajectoryDist(loc=x[:, :3], scale=0.3)


def draw_3d_path(strajs, draw_uncertainty=False, extent=[[5, 20], [-2, -2 + 15], [-4, -4 + 15]]):
    import ipyvolume as ipv
    ipv.clear()
    meanpath = np.mean(strajs, 1)
    ipv.plot(meanpath[:, 0], meanpath[:, 1], meanpath[:, 2])

    if draw_uncertainty:
        volumeresolution = 50

        drawvol = np.zeros([volumeresolution, volumeresolution, volumeresolution])
        for p in strajs.transpose([1, 0, 2]):
            for qstart, qend in zip(p[:-1], p[1:]):
                for posx, posy, posz in zip(
                        *[np.linspace(qstart[i], qend[i], int(10 * np.linalg.norm(qend - qstart))) for i in range(3)]):
                    volcoord = ((np.array([posx, posy, posz]) - np.array(extent)[:, 0]) * (
                            volumeresolution / (extent[0][1] - extent[0][0]))).astype(int)
                    if np.any(volcoord < 0): continue
                    if np.any(volcoord >= volumeresolution): continue
                    drawvol[volcoord[0], volcoord[1], volcoord[2]] += 1

        ipv.volshow(drawvol.transpose([2, 1, 0]), extent=extent, data_max=1)
    ipv.xlim(extent[0][0], extent[0][1])
    ipv.ylim(extent[1][0], extent[1][1])
    ipv.zlim(extent[2][0], extent[2][1])
    ipv.show()


def run_particle_smoothing(obstimes, startpoint, observations, N=10000, Nback=250):
    """
    Given the observation times (obstimes), a start point (3 dimensional location of e.g. the nest
    entrance) (startpoint) and the Nx7 observations array, run a particle smoother to compute a
    set of paths.

    Returns:
      pf = particle filter object, smoothed-trajectories (PxNx3) array (P=number of particles, N=number of obs)
    """
    model = BeeTrajectoryNoVelocity(times=obstimes, startpoint=startpoint)
    np.set_printoptions(precision=1, suppress=True)
    fk_model = ssm.AuxiliaryBootstrap(ssm=model, data=observations)
    pf = particles.SMC(fk=fk_model, N=10000,
                       collect=[Moments()], store_history=True)  # , ESSrmin=1.0,resampling='stratified')
    pf.run()  # actual computation

    smooth_trajectories = pf.hist.backward_sampling(250)
    strajs = np.array(smooth_trajectories)
    return pf, strajs
