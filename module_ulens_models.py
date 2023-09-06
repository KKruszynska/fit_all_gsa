import numpy as np
from scipy.optimize import least_squares

import parsubs

def ulens_nobl(time, t0, u0, te, I0):
    '''
    Returns magnitude of the microlensing event at time t
    for given parameters of the Paczynski curve without blending

    Args:
    :param time: float  - time for which magnitude is calculated
    :param t0: float - time of maximum
    :param u0: float - impact parameter
    :param te: float - Einstein time
    :param I0: float - baseline magnitude

    Returns:
    :return: I: float - magnitude of the Paczynski curve at given moment
    '''

    tau = (time - t0) / te
    u = np.sqrt(tau**2 + u0**2)
    ampl = (u**2 + 2.) / (u * np.sqrt(u**2 + 4.))
    I = I0 - 2.5 * np.log10(ampl)
    return I


def ulens_bl(time, t0, u0, te, I0, fs):
    '''
    Returns magnitude of the microlensing event at time t
    for given parameters of the Paczynski curve with blending

    Args:
    :param t: float  - time for which magnitude is calculated
    :param t0: float - time of maximum
    :param u0: float - impact parameter
    :param te: float - Einstein time
    :param I0: float - baseline magnitude
    :param fs: float - blending parameter, fraction of the total flux emitted by the source,
                       between 0 and 1

    Return:
    :return: I: float - magnitude of the Paczynski curve at given moment
    '''
    tau = (time - t0) / te

    u = np.sqrt(tau ** 2 + u0 ** 2)
    ampl = (u ** 2 + 2.) / (u * np.sqrt(u ** 2 + 4.))
    F = ampl * fs + (1. - fs)
    I = I0 - 2.5 * np.log10(F)
    return I

def ulens_par_nobl(time, alpha, delta, t0par, t0, u0, te, piEN, piEE, I0):
    '''
    Returns magnitude of the microlensing event at time t
    for given parameters of the Paczynski curve without blending
    including microlensing parallax effect

    Args:
    :param time: float  - time for which magnitude is calculated
    :param alpha: float  - right ascention of the event
    :param delta: float  - declination of the event
    :param t0par: float  - time 
    :param t0: float - time of maximum
    :param u0: float - impact parameter
    :param te: float - Einstein time
    :param piEN: float  - north component of microlensing parallax vector
    :param piEE: float  - east component of microlensing parallax vector
    :param I0: float - baseline magnitude

    Return:
    :return: I: float - magnitude of the Paczynski curve at given moment
    '''
    tau = (time - t0) / te
    qearr = []
    qnarr = []
    for hjd in time:
        qn = 0
        qe = 0
        qn, qe = parsubs.geta(hjd, alpha, delta, t0par)
        qnarr = np.append(qnarr, qn)
        qearr = np.append(qearr, qe)

    qnp = 0.
    qep = 0.

    qnarr = qnarr + qnp
    qearr = qearr + qep
    dtau = piEN * qnarr + piEE * qearr
    dbeta = -piEN * qearr + piEE * qnarr
    dbeta = -dbeta
    taup = tau + dtau
    betap = u0 + dbeta

    u = np.sqrt(taup ** 2 + betap ** 2)
    ampl = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    F = ampl
    I = I0 - 2.5 * np.log10(F)
    return I


def fit_ulensfixedbl_single(epoch, avmag, err):
    t0 = epoch[np.argmin(avmag)]
    te = 100.
    u0 = 0.5
    I0 = np.amax(avmag)
    x = epoch
    y = avmag

    ulensparam = [t0, te, u0, I0]
    print(ulensparam)
    fp = lambda v, x: ulens_nobl(x, v[0], v[1], v[2], v[3])
    e = lambda v, x, y, err: ((fp(v, x) - y) / err)
    v, success = least_squares(e, ulensparam, args=(x, y, err), max_nfev=1000000)
    #    print v, success
    chi2 = sum(e(v, x, y, err) ** 2)
    chi2dof = chi2 / (len(x) - len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof


def fit_ulensparallax_fixedbl_single(epoch, avmag, err, alpha, delta, v1):
    t0 = v1[0]
    t0par = t0
    te = v1[1]
    u0 = v1[2]
    I0 = v1[3]
    piEE = 0.0
    piEN = 0.0
    x = epoch
    y = avmag

    # ulensparallax(t, alpha, delta, t0par, t0, te, u0, piEN, piEE, I0, fs)kać 
    ulensparam = [t0, te, u0, piEN, piEE, I0]
    fp = lambda v, x: ulens_par_nobl(x, alpha, delta, t0par, v[0], v[1], v[2], v[3], v[4], v[5])
    e = lambda v, x, y, err: ((fp(v, x) - y) / err)
    v, success = least_squares(e, ulensparam, args=(x, y, err), max_nfev=1000000)
    #    print v, success
    chi2 = sum(e(v, x, y, err) ** 2)
    chi2dof = chi2 / (len(x) - len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof