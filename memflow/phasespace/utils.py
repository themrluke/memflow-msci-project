import torch
import math
import numpy as np


def set_square_t(inputt, square, negative=False):
    """Change the time component of this LorentzVector
    in such a way that self.square() = square.
    If negative is True, set the time component to be negative,
    else assume it is positive.
    """

    ret = torch.zeros_like(inputt)

    ret[:, 0] = (rho2_t(inputt) + square) ** 0.5

    if negative:
        ret[:, 0] *= -1
    ret[:, 1:] = inputt[:, 1:]

    return ret


def rho2_t(inputt):
    """Compute the radius squared. Vectorized."""
    output = inputt.clone()
    return torch.sum(output[:, 1:] * output[:, 1:], -1)


def rho2_tt(inputt):
    """Compute the radius squared. Vectorized and for batches."""

    return torch.sum(inputt[:, :, 1:] ** 2, -1)


def boostVector_t(inputt):

    if torch.min(inputt[:, 0]) <= 0.0 or torch.min(square_t(inputt)) < 0.0:
        print("Invalid boost")

    output = inputt.clone()
    return output[:, 1:] / output[:, 0].unsqueeze(1)


def square_t(inputt):
    output = inputt.clone()
    if inputt.shape[1] == 4 or inputt.shape[0] == 4:
        return dot_t(output, output)
    else:

        return torch.sum(output * output, -1)


def mass_t(inputt):
    return torch.sqrt(square_t(inputt))


def dot_t(inputa, inputb):
    """Dot product for four vectors"""
    #outputa = inputa.clone()
    #outputb = inputb.clone()
    return (
        inputa[:, 0] * inputb[:, 0]
        - inputa[:, 1] * inputb[:, 1]
        - inputa[:, 2] * inputb[:, 2]
        - inputa[:, 3] * inputb[:, 3]
    )


def dot_fb(inputa, inputb):
    """Dot product for four vectors in batches vectors"""
    return (
        inputa[:, :, 0] * inputb[:, :, 0]
        + inputa[:, :, 1] * inputb[:, :, 1]
        + inputa[:, :, 2] * inputb[:, :, 2]
    )


def dot_s(inputa, inputb):
    """Dot product for space vectors"""
    return (
        inputa[:, 0] * inputb[:, 0]
        + inputa[:, 1] * inputb[:, 1]
        + inputa[:, 2] * inputb[:, 2]
    )


def boost_t(inputt, boost_vector, gamma=-1.0):
    """Transport inputt into the rest frame of the boost_vector in argument.
    This means that the following command, for any vector p=(E, px, py, pz)
        boost_t(p,-boostVector(p))
    returns to (M,0,0,0).
    Version for a single phase pace point
    """

    b2 = square_t(boost_vector)

    if gamma < 0.0:
        gamma = 1.0 / torch.sqrt(1.0 - b2)
    inputt_space = inputt[:, 1:].clone()

    output = inputt.clone()
    E_inputt = inputt[:, 0].clone()

    bp = torch.sum(inputt_space * boost_vector, -1)

    #gamma2 = torch.where(b2 > 1e-7, (gamma - 1.0) / b2, torch.zeros_like(b2))
    gamma2 = torch.zeros_like(b2)
    mask = b2 > 0
    gamma2[mask] = (gamma[mask] - 1.0) / b2[mask]

    factor = gamma2 * bp + gamma * E_inputt

    output[:,1:] += factor.unsqueeze(1) * boost_vector

    output[:, 0] = gamma * (E_inputt + bp)

    return output


def boost_tt(inputt, boost_vector, gamma=-1.0):
    """Transport inputt into the rest frame of the boost_vector in argument.
    This means that the following command, for any vector p=(E, px, py, pz)
        boost_t(p,-boostVector(p))
    returns to (M,0,0,0).
    Version for a batch
    """
    b2 = square_t(boost_vector)
    if gamma < 0.0:
        gamma = 1.0 / torch.sqrt(1.0 - b2)
    inputt_space = inputt[:, :, 1:]
    
    output = inputt.clone()

    bp = torch.sum(inputt_space * boost_vector, -1)

    
    gamma2 = torch.where(b2 > 0, (gamma - 1.0) / b2, torch.zeros_like(b2))

    factor = gamma2 * bp + gamma * inputt[:, :, 0]

    output[:, :, 1:] += factor.unsqueeze(-1) * boost_vector

    output[:, :, 0] = gamma * (inputt[:, :, 0] + bp)

    return output


def cosTheta_t(inputt):

    ptot = torch.sqrt(torch.dot(inputt[1:], inputt[1:]))
    assert ptot > 0.0
    return inputt[3] / ptot


def phi_t(inputt):

    return torch.atan2(inputt[2], inputt[1])


def uniform_distr(r, minv, maxv):
    """distributes r uniformly within (min, max), with jacobian dvariable"""
    minv = torch.ones_like(r) * minv
    maxv = torch.ones_like(r) * maxv
    dvariable = maxv - minv
    variable = minv + dvariable * r

    # print(dvariable)
    return variable, dvariable


def uniform_distr_t(r, minv, maxv):
    """distributes r uniformly within (min, max), with jacobian dvariable"""
    dvariable = maxv - minv
    variable = minv + dvariable * r
    # print(dvariable)
    return variable, dvariable


def boost_to_lab_frame(momenta, xb_1, xb_2):
    """Boost a phase-space point from the COM-frame to the lab frame, given Bjorken x's.
    The first two elements are the initial particles"""

    def boost_lf(momenta, xb_1, xb_2):
        ref_lab = momenta[:, 0, :] * xb_1.unsqueeze(-1) + momenta[
            :, 1, :
        ] * xb_2.unsqueeze(-1)

        if not ((rho2_t(ref_lab) == 0).any()):
            lab_boost = boostVector_t(ref_lab)

            return boost_tt(momenta, lab_boost.unsqueeze(1))
        else:
            return momenta

    return torch.where(
        ((xb_1 != torch.ones_like(xb_1)) | (xb_2 != torch.ones_like(xb_2)))
        .unsqueeze(-1)
        .unsqueeze(-1),
        boost_lf(momenta, xb_1, xb_2),
        momenta,
    )


def pseudoRap(inputt, eps=np.finfo(float).eps ** 0.5, huge=np.finfo(float).max):
    """Compute pseudorapidity. Single PS point"""

    pt = torch.sqrt(torch.sum(inputt[:, 1:3] ** 2, axis=-1))

    th = torch.atan2(pt, inputt[:, 3])
    return torch.where(
        (pt < eps) & (torch.abs(inputt[:, 3]) < eps),
        huge * torch.ones_like(inputt[:, 3]),
        -torch.log(torch.tan(th / 2.0)),
    )


def pseudoRap_t(inputt, eps=np.finfo(float).eps ** 0.5, huge=np.finfo(float).max):
    """Compute pseudorapidity. Batch"""

    pt = torch.sqrt(torch.sum(inputt[:, :, 1:3] ** 2, axis=-1))

    th = torch.atan2(pt, inputt[:, :, 3])
    return torch.where(
        (pt < eps) & (torch.abs(inputt[:, :, 3]) < eps),
        huge * torch.ones_like(inputt[:, :, 3]),
        -torch.log(torch.tan(th / 2.0)),
    )


def getdelphi(
    inputt1, inputt2, eps=np.finfo(float).eps ** 0.5, huge=np.finfo(float).max
):
    """Compute the phi-angle separation with inputt2."""

    pt1 = torch.sqrt(torch.sum(inputt1[:, 1:3] ** 2, axis=-1))
    pt2 = torch.sqrt(torch.sum(inputt2[:, 1:3] ** 2, axis=-1))

    tmp = inputt1[:, 1] * inputt2[:, 1] + inputt1[:, 2] * inputt2[:, 2]
    tmp /= pt1 * pt2
    returner = torch.where(
        torch.abs(tmp) > torch.ones_like(tmp),
        torch.acos(tmp / torch.abs(tmp)),
        torch.acos(tmp),
    )
    returner = torch.where(
        (pt1 == 0.0) | (pt2 == 0.0), huge * torch.ones_like(returner), returner
    )
    return returner


def deltaR(inputt1, inputt2):
    """Compute the deltaR separation with momentum p2."""

    delta_eta = pseudoRap(inputt1) - pseudoRap(inputt2)
    delta_phi = getdelphi(inputt1, inputt2)
    return torch.sqrt(delta_eta**2 + delta_phi**2)


def get_x1x2_from_uniform(unif, final_state_mass, E_cm):
    ''' The function transform [0,1] uniform space
    to x1 and x2 fraction in the lab frame respecting the
    E_cm constraint.
    It returns also a weight, that corresponds to 1/probabilty.
    = 1/ |detJacInverse(x1,x2)|'''
    logtau = torch.log((final_state_mass / E_cm)**2)
    m = -1
    q = - logtau
    A = (m/2)*q**2 + q**2

    def inverse_cdf(u, m, q): 
        return ( -q/A + torch.sqrt((-q/A)**2 +2*m*u/A))/(m/A)

    minus_logx1 = inverse_cdf(unif[:,0], m, q)
    # then sample -log(x2)
    minus_logx2,_ = uniform_distr( unif[:,1], 0, m*minus_logx1 + q)

    x1 = torch.exp(-minus_logx1)
    x2 = torch.exp(-minus_logx2)

    detjacinv = 1/(A*x1*x2)  # det jac of inverse function to get p(x1)
    #determinant of the uniform->x1x2 transformation == 1/det(x1x2->uniform)
    weight = 1 / detjacinv 
    return x1, x2, weight


def get_uniform_from_x1x2(x1, x2, final_state_mass, E_cm):
    '''The function transform the x1, x2 lab frame fractions
    to the uniform [0,1] space.
    It returns the detJacInverse function that correspond to the probability
    of x1, x2'''

    logtau = torch.log((final_state_mass / E_cm)**2)
    m = -1
    q = -logtau
    A = (m/2)*q**2 + q**2  # normalization of the curve
    def cdf(x, m, q):
        ycum = ((m/2)*x**2 + q*x)/A
        return ycum

  
    mlogx1 = -torch.log(x1)
    mlogx2 = -torch.log(x2)
    
    u_1 = cdf(mlogx1, m, q)
    u_2 = mlogx2 / (m*mlogx1 + q)

    jac = 1/(A*x1*x2)
    return torch.cat((u_1.unsqueeze(1), u_2.unsqueeze(1)), axis=1), jac
