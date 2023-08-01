import os
import torch
# import tensorflow as tf

import logging
import math
import copy
import datetime
import sys
from .utils import *

M_HIGGS = 125.25
M_TOP = 172.5
M_GLUON = 0.0


class PhaseSpaceGeneratorError(Exception):
    pass


# =========================================================================================
# Phase space generation
#
# Original source code from https://github.com/NGoetz/TorchPS
#
# =========================================================================================


class VirtualPhaseSpaceGenerator(object):
    def __init__(
        self,
        initial_masses,
        final_masses,
        collider_energy,
        pdf=None,
        pdf_active=False,
        uniform_x1x2=True,
        lhapdf_dir=None,
        dev = None
    ):
        '''
        - pdf_active == True:  the energy of the center of mass is modelled by x1 and x2 fraction.
        It is not always constant.

        - pdf=None:  if the pdf object is none the pdfWeight is not computed

        - uniform_x1x2: if True, two uniform numbers are given to model the x1 and x2 fraction.
        They need to be converted to x1 and x2 from the uniform space
        '''
        if dev == None:
            dev = (
                torch.device("cuda:" + str(0))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.initial_masses = initial_masses
        self.masses_t = final_masses.clone().to(dev)
        self.tot_final_state_masses = torch.sum(self.masses_t)
        self.n_initial = len(initial_masses)
        self.n_final = len(final_masses)
        self.collider_energy = collider_energy

        self.pdf = pdf
        self.pdf_active = pdf_active
        self.uniform_x1x2 = uniform_x1x2
        # if self.pdf_active:
        #     if lhapdf_dir not in sys.path:
        #         sys.path.append(lhapdf_dir)
        #     import pdfflow

    def generateKinematics(self, E_cm, random_variables):
        """Generate a phase-space point with fixed center of mass energy."""

        raise NotImplementedError

    @property
    def nDimPhaseSpace(self):
        """Return the number of random numbers required to produce
        a given multiplicity final state."""

        if self.n_final == 1:
            return 0
        return 3 * self.n_final - 4


class FlatInvertiblePhasespace(VirtualPhaseSpaceGenerator):
    """Implementation following S. Platzer, arxiv:1308.2922"""

    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    epsilon_border = 1e-10

    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymore
    absolute_Ecm_min = 1.0

    def __init__(self, *args, **opts):

        super(FlatInvertiblePhasespace, self).__init__(*args, **opts)
        if self.n_initial == 1:
            raise PhaseSpaceGeneratorError(
                "This basic generator does not support decay topologies."
            )
        if self.n_initial > 2:
            raise PhaseSpaceGeneratorError(
                "This basic generator does not support more than 2 incoming particles."
            )

    @staticmethod
    def get_flatWeights(E_cm, n):
        """Return the phase-space volume for a n massless final states.
        Vol(E_cm, n) = (2*pi)^(4-3n)*(pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
        """
        # includes full phase space factor
        if n == 1:
            # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
            # must typically be accounted for in the MC integration framework since we
            # don't have access to that here, so we just return 1.
            return 1.0
        if not torch.is_tensor(E_cm):
            return (
                math.pow(2 * math.pi, 4 - 3 * n)
                * math.pow((math.pi / 2.0), n - 1)
                * (
                    math.pow((E_cm**2), n - 2)
                    / (math.factorial(n - 1) * math.factorial(n - 2))
                )
            )
        else:
            E_cm = E_cm.to(torch.float)
            return (
                math.pow(2 * math.pi, 4 - 3 * n)
                * math.pow((math.pi / 2.0), n - 1)
                * (
                    torch.pow((E_cm**2), n - 2)
                    / (math.factorial(n - 1) * math.factorial(n - 2))
                )
            )

    def massless_map(self, x, exp):
        return (x ** (exp)) * ((exp + 1) - (exp) * x)

    @staticmethod
    def rho(M, N, m):
        """Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))"""
        Msqr = M**2
        return ((Msqr - (N + m) ** 2) * (Msqr - (N - m) ** 2)) ** 0.5 / (8.0 * Msqr)

    # def get_pdfQ2(self, pdf, pdg, x, scale2):
    #     """Call the PDF and return the corresponding density.
    #     The pdf object needs to be a pdfflow instance."""
    #     if pdf is None:
    #         return torch.ones_like(x)

    #     if pdg not in [21] and abs(pdg) not in range(1, 7):
    #         return torch.ones_like(x)

    #     # Call to lhapdf API
    #     f = pdf.xfxQ2(
    #         [pdg],
    #         tf.convert_to_tensor(x, dtype=tf.float64),
    #         tf.convert_to_tensor(scale2, dtype=tf.float64),
    #     )
    #     return torch.tensor(f.numpy(), dtype=torch.double, device=x.device)

    def generateKinematics_batch(
        self,
        random_variables_full,
        pT_mincut=-1,
        delR_mincut=-1,
        rap_maxcut=-1,
        pdgs=[0, 0],
    ):
        """Generate a self.n_initial -> self.n_final phase-space point
        using the random variables passed in argument, including phase space cuts and PDFs.

        The function returns the four momenta of incoming and outcoming particle
        in the CM, a weights, and the x1 and x2 fractions in the lab.

        The weight is the determinant of the Rambo function from uniform
        space to the parton space.
        The probability density of the partons is then 1/weight as the weight is
        |detRambo(uniform)|. 

        """
        self.masses_t = self.masses_t.to(random_variables_full.device)
        # Make sure that none of the random_variables is NaN.
        if (
            torch.is_tensor(random_variables_full)
            and torch.isnan(random_variables_full).any()
        ):
            raise PhaseSpaceGeneratorError(
                "Some of the random variables passed "
                + "to the phase-space generator are NaN: %s"
                % str(random_variables.data.tolist())
            )

        wgt_jac = torch.ones(
            random_variables_full.shape[0],
            dtype=torch.double,
            device=random_variables_full.device,
        )
        xb_1 = torch.ones(
            random_variables_full.shape[0],
            dtype=torch.double,
            device=random_variables_full.device,
        )
        xb_2 = torch.ones(
            random_variables_full.shape[0],
            dtype=torch.double,
            device=random_variables_full.device,
        )
        if not self.pdf_active:
            random_variables = random_variables_full
        else:
            random_variables = random_variables_full[:, :-2]

            if self.uniform_x1x2:
                # Compute x1 and x2 from the uniform input random numbers
                xb_1, xb_2, wgt_jac_x1x2 = get_x1x2_from_uniform(random_variables_full[:, -2:],
                                                                 self.tot_final_state_masses, self.collider_energy)
                # Computing the actual E_cm in the incoming particle restframe
                E_cm = torch.sqrt(xb_1 * xb_2) * self.collider_energy
                # the get_x1x2_from_uniform return the weight, 1/detjac_inv
                wgt_jac *= wgt_jac_x1x2
            else:
                # Just consider the last two numbers the x1 and x2 fractions
                xb_1 = random_variables_full[:, -2]
                xb_2 = random_variables_full[:, -1]
                # Computing the actual E_cm in the incoming particle restframe
                E_cm = torch.sqrt(xb_1 * xb_2) * self.collider_energy

            # The original code was using the Z scale as Q^2 of the PDF
            p_energy = (torch.ones_like(xb_1) * (91.188) ** 2).to(
                random_variables.device
            )

            # x_cut = torch.where(
            #     xb_1 < 1e-4, torch.zeros_like(xb_1), torch.ones_like(xb_1)
            # )
            # x_cut = torch.where(xb_2 < 1e-4, torch.zeros_like(x_cut), x_cut)
            # wgt_jac *= (
            #     self.get_pdfQ2(self.pdf, pdgs[0], xb_1, p_energy)
            #     * self.get_pdfQ2(self.pdf, pdgs[1], xb_2, p_energy)
            #     * x_cut
            # )

        assert random_variables.shape[1] == self.nDimPhaseSpace

        # The distribution weight of the generate PS point
        weight = torch.ones(
            random_variables.shape[0],
            dtype=torch.double,
            device=random_variables.device,
        )
        weight *= wgt_jac
        output_momenta_t = []
        self.masses_t = self.masses_t.to(random_variables.device)
        mass = self.masses_t[0]

        if not self.pdf_active:
            M = [0.0] * (self.n_final - 1)
            M[0] = E_cm
            M = torch.tensor(
                M,
                requires_grad=False,
                dtype=torch.double,
                device=random_variables.device,
            )
            M = torch.unsqueeze(M, 0).repeat(random_variables.shape[0], 1)
        else:
            M = [[0.0] * (self.n_final - 1)] * random_variables.shape[0]
            M = torch.tensor(
                M,
                requires_grad=False,
                dtype=torch.double,
                device=random_variables.device,
            )
            M[:, 0] = E_cm
            M.to(random_variables.device)
            E_cm.to(random_variables.device)
            self.masses_t.to(random_variables.device)

        # generate the intermediate masses and get the additional weight for massive particles
        weight *= self.generateIntermediatesMassive_batch(M, E_cm, random_variables)
        Q_t = torch.tensor(
            [0.0, 0.0, 0.0, 0.0],
            requires_grad=False,
            dtype=torch.double,
            device=random_variables.device,
        )
        Q_t = Q_t.unsqueeze(0).repeat(random_variables.shape[0], 1)
        Q_t[:, 0] = M[:, 0]
        M = torch.cat(
            (
                M,
                self.masses_t.unsqueeze(0).repeat(random_variables.shape[0], 1)[:, -1:],
            ),
            -1,
        )
        q_t = (
            4.0
            * M[:, :-1]
            * self.rho(M[:, :-1], M[:, 1:], self.masses_t[:-1].to(M.device))
        )
        rnd = random_variables[:, self.n_final - 2 : 3 * self.n_final - 4]
        cos_theta_t = 2.0 * rnd[:, 0::2] - 1.0
        theta_t = torch.acos(cos_theta_t)
        sin_theta_t = torch.sqrt(1.0 - cos_theta_t**2)
        phia_t = 2 * math.pi * rnd[:, 1::2]
        cos_phi_t = torch.cos(phia_t)
        sqrt = torch.sqrt(1.0 - cos_phi_t**2)
        sin_phi_t = torch.where(phia_t > math.pi, -sqrt, sqrt)
        a = torch.unsqueeze((q_t * sin_theta_t * cos_phi_t), 0)
        b = torch.unsqueeze((q_t * sin_theta_t * sin_phi_t), 0)
        c = torch.unsqueeze((q_t * cos_theta_t), 0)
        lv = torch.cat((torch.zeros_like(a), a, b, c), 0)
        output_returner = torch.zeros(
            (random_variables.shape[0], self.n_initial + self.n_final, 4),
            dtype=torch.double,
            device=random_variables.device,
        )
        for i in range(self.n_initial + self.n_final - 1):
            if i < self.n_initial:
                output_returner[:, i, :] = 0
                continue
            p2 = lv[:, :, i - self.n_initial].t()
            p2 = set_square_t(p2, self.masses_t[i - self.n_initial] ** 2)
            p2 = boost_t(p2, boostVector_t(Q_t))
            p2 = set_square_t(p2, self.masses_t[i - self.n_initial] ** 2)
            output_returner[:, i, :] = p2
            nextQ_t = Q_t - p2
            nextQ_t = set_square_t(nextQ_t, M[:, i - self.n_initial + 1] ** 2)
            Q_t = nextQ_t

        output_returner[:, -1, :] = Q_t
        # Create the partons in the CM
        self.setInitialStateMomenta_batch(output_returner, E_cm)

        output_returner_save = output_returner.clone()
        output_returner = boost_to_lab_frame(output_returner, xb_1, xb_2)

        q_theta2 = torch.min(
            torch.abs(
                torch.sqrt(
                    output_returner[:, 2:, 1] ** 2 + output_returner[:, 2:, 2] ** 2
                )
            ),
            axis=1,
        ).values

        factor2 = torch.where(
            q_theta2 < torch.ones_like(q_theta2) * pT_mincut,
            torch.zeros_like(weight),
            torch.ones_like(weight),
        )

        for i in range(output_returner[:, 2:, :].shape[1]):
            for j in range(output_returner[:, 2:, :].shape[1]):
                if i > j:

                    factor2 *= torch.where(
                        torch.abs(
                            deltaR(
                                output_returner[:, i + 2, :],
                                output_returner[:, j + 2, :],
                            )
                        )
                        < torch.ones_like(weight) * delR_mincut,
                        torch.zeros_like(weight),
                        torch.ones_like(weight),
                    )

        if rap_maxcut > 0:
            factor2 *= torch.where(
                rap_maxcut
                < torch.abs(
                    torch.max(pseudoRap_t(output_returner[:, 2:, :]), axis=1).values
                ),
                torch.zeros_like(weight),
                torch.ones_like(weight),
            )

        weight = weight * factor2
        # Add the additional weight factor 1/2s
        # We don't add it here because it is not part of the rambo
        # transformation
        # shat = xb_1 * xb_2 * self.collider_energy**2
        return output_returner_save, weight, xb_1, xb_2

    def bisect_vec_batch(self, v_t, target=1.0e-16, maxLevel=600):
        """Solve v = (n+2) * u^(n+1) - (n+1) * u^(n+2) for u. Vectorized, batched"""
        if v_t.size(1) == 0:
            return

        exp = torch.arange(
            self.n_final - 2, 0, step=-1, device=v_t.device, dtype=torch.double
        )

        exp = exp.unsqueeze(0).repeat(v_t.shape[0], 1)
        level = 0
        left = torch.zeros_like(v_t)
        right = torch.ones_like(v_t)

        checkV = torch.ones_like(v_t) * -1
        u = torch.ones_like(v_t) * -1
        error = torch.ones_like(v_t)
        maxLevel = maxLevel / 10
        ml = maxLevel
        oldError = 100
        while torch.max(error) > target and ml < 10 * maxLevel:

            while level < ml:
                u = (left + right) * (0.5 ** (level + 1))

                checkV = self.massless_map(u, exp)

                left *= 2.0
                right *= 2.0
                con = torch.ones_like(left) * 0.5
                adder = torch.where(v_t <= checkV, con * -1.0, con)

                left = left + (adder + 0.5)
                right = right + (adder - 0.5)

                level += 1

            error = torch.abs(1.0 - checkV / v_t)

            ml = ml + maxLevel
            newError = torch.max(error)
            if newError >= oldError:
                break
            else:
                oldError = newError

        return u

    def generateIntermediatesMassless_batch(self, M_t, E_cm, random_variables):
        """Generate intermediate masses for a massless final state. Batch version"""

        u = self.bisect_vec_batch(random_variables[:, : self.n_final - 2])

        for i in range(2, self.n_final):
            M_t[:, i - 1] = torch.sqrt(u[:, i - 2] * (M_t[:, i - 2] ** 2))
        if not torch.is_tensor(E_cm) or len(E_cm.size()) == 0:
            return torch.tensor(
                [self.get_flatWeights(E_cm, self.n_final)] * random_variables.shape[0],
                dtype=torch.double,
                device=random_variables.device,
            )
        else:
            return self.get_flatWeights(E_cm, self.n_final)

    def generateIntermediatesMassive_batch(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massive final state. Batch version"""

        M[:, 0] -= torch.sum(self.masses_t)

        weight = self.generateIntermediatesMassless_batch(M, E_cm, random_variables)
        K_t = M.clone()
        masses_sum = torch.flip(
            torch.cumsum(torch.flip(self.masses_t, (-1,)), -1), (-1,)
        )
        M += masses_sum[:-1].to(M.device)

        weight[:] *= 8.0 * self.rho(
            M[:, self.n_final - 2],
            self.masses_t[self.n_final - 1],
            self.masses_t[self.n_final - 2],
        )
        weight[:] *= torch.prod(
            (
                self.rho(
                    M[:, : self.n_final - 2],
                    M[:, 1:],
                    self.masses_t[: self.n_final - 2].to(M.device),
                )
                / self.rho(K_t[:, : self.n_final - 2], K_t[:, 1:], 0.0)
            )
            * (M[:, 1 : self.n_final - 1] / K_t[:, 1 : self.n_final - 1]),
            -1,
        )

        weight[:] *= torch.pow(K_t[:, 0] / M[:, 0], 2 * self.n_final - 4)

        return weight

    def setInitialStateMomenta_batch(self, output_momenta, E_cm):
        """Generate the initial state momenta. Batch version"""
        if self.n_initial not in [2]:
            raise PhaseSpaceGeneratorError(
                "This PS generator only supports 2 initial states"
            )

        if self.n_initial == 2:
            if self.initial_masses[0] == 0.0 or self.initial_masses[1] == 0.0:
                if not torch.is_tensor(E_cm):

                    output_momenta[:, 0, :] = torch.tensor(
                        [E_cm / 2.0, 0.0, 0.0, +E_cm / 2.0],
                        dtype=torch.double,
                        device=output_momenta[0].device,
                    )
                    output_momenta[:, 1, :] = torch.tensor(
                        [E_cm / 2.0, 0.0, 0.0, -E_cm / 2.0],
                        dtype=torch.double,
                        device=output_momenta[0].device,
                    )
                else:
                    E_cm_p = E_cm.unsqueeze(-1).float()
                    output_momenta[:, 0, :] = torch.cat(
                        (
                            E_cm_p / 2,
                            torch.zeros_like(E_cm_p),
                            torch.zeros_like(E_cm_p),
                            1 * E_cm_p / 2,
                        ),
                        -1,
                    )
                    output_momenta[:, 1, :] = torch.cat(
                        (
                            E_cm_p / 2,
                            torch.zeros_like(E_cm_p),
                            torch.zeros_like(E_cm_p),
                            -1 * E_cm_p / 2,
                        ),
                        -1,
                    )
            else:
                M1sq = self.initial_masses[0] ** 2
                M2sq = self.initial_masses[1] ** 2
                E1 = (E_cm**2 + M1sq - M2sq) / E_cm
                E2 = (E_cm**2 - M1sq + M2sq) / E_cm

                if not torch.is_tensor(E_cm):
                    Z = (
                        math.sqrt(
                            E_cm**4
                            - 2 * E_cm**2 * M1sq
                            - 2 * E_cm**2 * M2sq
                            + M1sq**2
                            - 2 * M1sq * M2sq
                            + M2sq**2
                        )
                        / E_cm
                    )
                    output_momenta[:, 0, :] = torch.tensor(
                        [E1 / 2.0, 0.0, 0.0, +Z / 2.0],
                        dtype=torch.double,
                        device=output_momenta[0].device,
                    )
                    output_momenta[:, 1, :] = torch.tensor(
                        [E2 / 2.0, 0.0, 0.0, -Z / 2.0],
                        dtype=torch.double,
                        device=output_momenta[0].device,
                    )
                else:
                    Z = (
                        torch.sqrt(
                            E_cm**4
                            - 2 * E_cm**2 * M1sq
                            - 2 * E_cm**2 * M2sq
                            + M1sq**2
                            - 2 * M1sq * M2sq
                            + M2sq**2
                        )
                        / E_cm
                    )
                    E1_p = E1.unsqueeze(-1)
                    E2_p = E2.unsqueeze(-1)
                    Z_p = Z.unsqueeze(-1)
                    output_momenta[:, 0, :] = torch.cat(
                        (
                            E1_p / 2,
                            torch.zeros_like(E1_p),
                            torch.zeros_like(E1_p),
                            1 * Z_p / 2,
                        ),
                        -1,
                    )
                    output_momenta[:, 1, :] = torch.cat(
                        (
                            E2_p / 2,
                            torch.zeros_like(E1_p),
                            torch.zeros_like(E1_p),
                            -1 * Z_p / 2,
                        ),
                        -1,
                    )
        return

    def getPSpoint_batch(self, momenta_batch, x1, x2, order=[0,1,2,3], target_mass=torch.Tensor([[M_HIGGS, M_TOP, M_TOP, M_GLUON]]), ensure_CM=True, ensure_onShell=True):
        """Generate a self.n_final -> self.n_initial phase-space point
        using the four momenta given in input.

        Only the final particle momenta are given.
        The Final state is assumed to be in the CM
        """

        # do the permutation of H t t ISR
        perm = torch.LongTensor(order)
        momenta_batch = momenta_batch[:,perm,:]

        n = self.n_final
        P = momenta_batch.clone()  # copy the final state particles
        
        if ensure_CM:
            # Check if we are in the CM
            ref_lab = torch.sum(P, axis=1)
            if not ((rho2_t(ref_lab) < 1e-3).any()):
                raise Exception("Momenta batch not in the CM, failing to convert back to PS point")
        
        # We start getting M and then K
        M = torch.tensor(
            [0.0] * n, requires_grad=False, dtype=torch.double, device=P.device
        )
        M = torch.unsqueeze(M, 0).repeat(P.shape[0], 1)
        K_t = M.clone()
        Q = torch.zeros_like(P).to(P.device)
        Q[:, -1] = P[:, -1]  # Qn = pn

        if ensure_onShell:
            masses_t = self.masses_t[perm].unsqueeze(dim=0)
        else:
            masses_t = target_mass
        #print(masses_t.shape)

        # intermediate mass
        for i in range(n, 0, -1):
            j = i - 1
            M[:, j] = torch.sqrt(square_t(torch.sum(P[:, j:n], axis=1)))
            # Remove the final masses to convert back to K
            K_t[:, j] = M[:,j] - torch.sum(masses_t[:,j:])
        
        # output [0,1] distributed numbers
        r = torch.zeros(P.shape[0], self.nDimPhaseSpace, device=P.device)

        for i in range(n, 1, -1):
            j = i - 1  # index for 0-based tensors
            # in the direct algo the u are squared.

            u = (K_t[:, j]/K_t[:, j-1]) ** 2

            r[:, j - 1] = (n + 1 - i) * (torch.pow(u, (n - i))) - (n - i) * (
                torch.pow(u, (n + 1 - i))
            )

            Q[:, j - 1] = Q[:, j] + P[:, j - 1]

            P[:, j - 1] = boost_t(P[:, j - 1], -1*boostVector_t(Q[:, j - 1]))

            P_copy = P.clone().to(P.device)
            r[:, n - 5 + 2 * i - 1] = (
                (P_copy[:, j - 1, 3] / torch.sqrt(rho2_t(P_copy[:, j - 1]))) + 1
            ) / 2
            # phi= tan^-1(Py/Px)
            phi = torch.atan(P_copy[:, j - 1, 2] / P_copy[:, j - 1, 1])
            # Fixing phi depending on X and y sign
            # 4th quandrant  (px > 0, py < 0)
            deltaphi = torch.where(
                (P_copy[:, j - 1, 2] < 0) & (P_copy[:, j - 1, 1] > 0), 2 * torch.pi, 0.0
            )
            # 2th and 3th quadratant  (px < 0, py whatever)
            deltaphi += torch.where((P_copy[:, j - 1, 1] < 0), torch.pi, 0.0)
            phi += deltaphi
            r[:, n - 4 + 2 * i - 1] = phi / (2 * torch.pi)

        # Get uniform from x1x2 space
        r1r2, jacinv_x1x2 = get_uniform_from_x1x2(x1, x2, self.tot_final_state_masses, self.collider_energy)
        Ecm_byev =  torch.sqrt(x1 * x2) * self.collider_energy
        
        # Get the Rambo weight
        # massless weight
        # This is needed to have the same formula as the direct
        M = M[:, :-1]
        K_t = K_t[:,:-1]
        rambo_jac = self.get_flatWeights(Ecm_byev, self.n_final)
        # Now for the mass case
        rambo_jac[:] *= 8.0 * self.rho(
            M[:, self.n_final - 2],
            masses_t[:,self.n_final - 1],
            masses_t[:,self.n_final - 2],
        )
        rambo_jac[:] *= torch.prod(
            (
                self.rho(
                    M[:, : self.n_final - 2],
                    M[:, 1:],
                    masses_t[:,: self.n_final - 2].to(M.device),
                )
                / self.rho(K_t[:, : self.n_final - 2], K_t[:, 1:], 0.0)
            )
            * (M[:, 1 : self.n_final - 1] / K_t[:, 1 : self.n_final - 1]),
            -1,
        )

        rambo_jac[:] *= torch.pow(K_t[:, 0] / M[:, 0], 2 * self.n_final - 4)

        # The probability if |detJac Rambo^-1| = 1/ |detJac Rambo| = 1 / weight
        # When we want to evaluate the probability of a ps point
        # we need the determinant of the Rambo^-1 converion --> 1/detRambo
        prob = 1/rambo_jac
        # Adding the jac of the inverse x1x2 transformation
        prob *= jacinv_x1x2
        
        return torch.cat((r, r1r2), axis=1), prob
