import torch
import vector
import numpy as np
import awkward as ak
from memflow.phasespace.phasespace import PhaseSpace
import memflow.phasespace.utils as utils

M_HIGGS = 125.25
M_TOP = 172.5
M_GLUON = 1e-5



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_flat_tensor(X, fields, axis=1, allow_missing=False):
    return torch.tensor(np.stack([ak.to_numpy(getattr(X,f), allow_missing=allow_missing) for f in fields], axis=axis))

class Compute_ParticlesTensor:

    M_MIN_TOT = M_HIGGS + 2*M_TOP + M_GLUON

    def get_HttISR(cond_X, log_mean, log_std, device):

        higgs = cond_X[0].unsqueeze(dim=1)
        thad = cond_X[1].unsqueeze(dim=1)
        tlep = cond_X[2].unsqueeze(dim=1)
        Htt = torch.cat((higgs, thad, tlep), dim=1)

        unscaledlog = Htt*log_std + log_mean
        data_regressed = unscaledlog
        data_regressed[:,:,0] = torch.sign(unscaledlog[:,:,0])*(torch.exp(torch.abs(unscaledlog[:,:,0])) - 1)

        higgs = data_regressed[:,0]
        thad = data_regressed[:,1]
        tlep = data_regressed[:,2]

        higgs = vector.array(
            {
                "pt": higgs[:,0].detach().cpu().numpy(),
                "eta": higgs[:,1].detach().cpu().numpy(),
                "phi": higgs[:,2].detach().cpu().numpy(),
                "mass": M_HIGGS*np.ones(higgs.shape[0])
            }
        )

        thad = vector.array(
            {
                "pt": thad[:,0].detach().cpu().numpy(),
                "eta": thad[:,1].detach().cpu().numpy(),
                "phi": thad[:,2].detach().cpu().numpy(),
                "mass": M_TOP*np.ones(thad.shape[0])
            }
        )

        tlep = vector.array(
            {
                "pt": tlep[:,0].detach().cpu().numpy(),
                "eta": tlep[:,1].detach().cpu().numpy(),
                "phi": tlep[:,2].detach().cpu().numpy(),
                "mass": M_TOP*np.ones(tlep.shape[0])
            }
        )

        higgs = ak.with_name(higgs, name="Momentum4D")
        thad = ak.with_name(thad, name="Momentum4D")
        tlep = ak.with_name(tlep, name="Momentum4D")

        gluon_px = -(higgs.px + thad.px + tlep.px)
        gluon_py = -(higgs.py + thad.py + tlep.py)
        gluon_pz = -(higgs.pz + thad.pz + tlep.pz)
        E_gluon = np.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2)

        gluon_px = np.expand_dims(gluon_px, axis=1)
        gluon_py = np.expand_dims(gluon_py, axis=1)
        gluon_pz = np.expand_dims(gluon_pz, axis=1)
        E_gluon = np.expand_dims(E_gluon, axis=1)

        gluon = np.concatenate((E_gluon, gluon_px, gluon_py, gluon_pz), axis=1)

        glISR = vector.array(
            {
                "E": gluon[:,0],
                "px": gluon[:,1],
                "py": gluon[:,2],
                "pz": gluon[:,3],
            }
        )

        glISR = ak.with_name(glISR, name="Momentum4D")

        higgs_tensor = to_flat_tensor(higgs, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        thad_tensor = to_flat_tensor(thad, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        tlep_tensor = to_flat_tensor(tlep, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        gluon_tensor = torch.Tensor(gluon).unsqueeze(dim=1)

        data_regressed = torch.cat((higgs_tensor, thad_tensor, tlep_tensor, gluon_tensor), dim=1)

        boost_regressed = higgs + thad + tlep + glISR
        boost_regressed = to_flat_tensor(boost_regressed, ["t", "x", "y", "z"], axis=1, allow_missing=False)

        if (device == torch.device('cuda')):
            data_regressed = data_regressed.cuda()
            boost_regressed = boost_regressed.cuda()
        
        return data_regressed, boost_regressed

    def get_cartesian_comp(particle, mass):
        # px
        Px = particle[:,0]*torch.cos(particle[:,2])
        # py
        Py = particle[:,0]*torch.sin(particle[:,2])
        # pz
        Pz = particle[:,0]*torch.sinh(particle[:,1])
         # E
        E = torch.sqrt(Px**2 + Py**2 + Pz**2 + mass**2)
        return torch.stack((E, Px, Py, Pz), dim=1)

    def get_ptetaphi_comp(particle):
        particle_pt = (particle[:,1]**2 + particle[:,2]**2)**0.5
        particle_eta = -torch.log(torch.tan(torch.atan2(particle_pt, particle[:,3])/2))
        particle_phi = torch.atan2(particle[:,2], particle[:,1])
        return torch.stack((particle_pt, particle_eta, particle_phi), dim=1)

    def get_HttISR_numpy(cond_X, log_mean, log_std, device, cartesian=True, eps=0.0):

        higgs = cond_X[0].unsqueeze(dim=1)
        thad = cond_X[1].unsqueeze(dim=1)
        tlep = cond_X[2].unsqueeze(dim=1)
        Htt = torch.cat((higgs, thad, tlep), dim=1)

        unscaledlog = Htt*log_std + log_mean
        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.sign(unscaledlog[:,:,0])*(torch.exp(torch.abs(unscaledlog[:,:,0])) - 1)

        higgs = data_regressed[:,0]
        thad = data_regressed[:,1]
        tlep = data_regressed[:,2]

        higgs_cartesian = Compute_ParticlesTensor.get_cartesian_comp(higgs, M_HIGGS).unsqueeze(dim=1)
        thad_cartesian = Compute_ParticlesTensor.get_cartesian_comp(thad, M_TOP).unsqueeze(dim=1)
        tlep_cartesian = Compute_ParticlesTensor.get_cartesian_comp(tlep, M_TOP).unsqueeze(dim=1)

        gluon_px = -(higgs_cartesian[:,0,1] + thad_cartesian[:,0,1] + tlep_cartesian[:,0,1]).unsqueeze(dim=1)
        gluon_py = -(higgs_cartesian[:,0,2] + thad_cartesian[:,0,2] + tlep_cartesian[:,0,2]).unsqueeze(dim=1)
        gluon_pz = -(higgs_cartesian[:,0,3] + thad_cartesian[:,0,3] + tlep_cartesian[:,0,3]).unsqueeze(dim=1)
        E_gluon = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps # add epsilon to have positive masses of the gluon
        gluon_cartesian = torch.cat((E_gluon, gluon_px, gluon_py, gluon_pz), dim=1).unsqueeze(dim=1).to(device)

        if cartesian:
            data_regressed = torch.cat((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
            # order: by default higgs/thad/tlep/glISR
            # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
            # I changed to do this permutation of the particles inside the get_PS
        else:
            gluon_pt = (gluon_px**2 + gluon_py**2)**0.5
            gluon_eta = -torch.log(torch.tan(torch.atan2(gluon_pt, gluon_pz)/2))
            gluon_phi = torch.atan2(gluon_py, gluon_px)
            gluon = torch.cat((gluon_pt, gluon_eta, gluon_phi), dim=1).to(device)
            data_regressed = torch.cat((higgs, thad, tlep, gluon), dim=1)
            
        boost_regressed = gluon_cartesian + higgs_cartesian + thad_cartesian + tlep_cartesian
        boost_regressed = boost_regressed.squeeze(dim=1)

        # if (device == torch.device('cuda')):
        #     data_regressed = data_regressed.cuda()
        #     boost_regressed = boost_regressed.cuda()
        
        return data_regressed, boost_regressed

    def get_HttISR_fromlab(cond_X, log_mean_parton,
                                  log_std_parton,
                                  log_mean_boost, log_std_boost,
                                  device, cartesian=True, eps=1e-4,
                                  return_both=False):

        boost = cond_X[3] # there was a bug here using -1, now fixed
        Htt = torch.stack((cond_X[0], cond_X[1],cond_X[2]), dim=1)

        unscaledlog = Htt*log_std_parton + log_mean_parton
        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1

        boost_regressed = boost*log_std_boost[1] + log_mean_boost[1]
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        #boost_regressed = torch.sign(boost_regressed)*torch.exp(torch.abs(boost_regressed)-1)        

        higgs = data_regressed[:,0]
        thad = data_regressed[:,1]
        tlep = data_regressed[:,2]
        boost_pz = boost_regressed

        higgs_cartesian = Compute_ParticlesTensor.get_cartesian_comp(higgs, M_HIGGS)
        thad_cartesian = Compute_ParticlesTensor.get_cartesian_comp(thad, M_TOP)
        tlep_cartesian = Compute_ParticlesTensor.get_cartesian_comp(tlep, M_TOP)

        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_py = -(higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        gluon_pz = boost_pz[:,0] -(higgs_cartesian[:,3] + thad_cartesian[:,3] + tlep_cartesian[:,3])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps

        # we need it for boost that is always cartesian
        gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
        boost_regressed_new = gluon_cartesian + higgs_cartesian + thad_cartesian + tlep_cartesian

        if not cartesian or return_both:
            gluon_pt = (gluon_px**2 + gluon_py**2)**0.5
            gluon_eta = -torch.log(torch.tan(torch.atan2(gluon_pt, gluon_pz)/2))
            gluon_phi = torch.atan2(gluon_py, gluon_px)
            gluon_ptetaphi = torch.stack((gluon_pt, gluon_eta, gluon_phi), dim=1)

        # order: by default higgs/thad/tlep/glISR
        # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
        # I changed to do this permutation of the particles inside the get_PS
            
        if not return_both:
            if cartesian:
                data_regressed_new = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
                return data_regressed_new, boost_regressed_new
            else:
                data_regressed_new = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
                return data_regressed_new, boost_regressed_new
        else:
            data_regressed_new_cart = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
            data_regressed_new_ptetaphi = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
            return data_regressed_new_cart, data_regressed_new_ptetaphi, boost_regressed_new
        

    def get_PS(data_HttISR, boost_reco, order=[0,1,2,3],
               target_mass=torch.Tensor([[M_HIGGS, M_TOP, M_TOP, M_GLUON]]),
               E_CM = 13000):

        # do the permutation of H t t ISR
        perm = torch.LongTensor(order)
        data_HttISR = data_HttISR[:,perm,:]

        # check order of components of boost_reco
        x1 = (boost_reco[:, 0] + boost_reco[:, 3]) / E_CM
        x2 = (boost_reco[:, 0] - boost_reco[:, 3]) / E_CM
        # the input validation is done here!! Check your vectors!!!
        mask_problems = (x1>1.)| (x1<0.) | (x2<0.) | (x1>1.)
        x1 = torch.clamp(x1, min=0., max=1.)
        x2 = torch.clamp(x2, min=0., max=1.)
        
        n = 4
        nDimPhaseSpace = 8
        masses_t = target_mass[:,perm]

        P = data_HttISR.clone()  # copy the final state particless
        
        # Check if we are in the CM
        ref_lab = torch.sum(P, axis=1)
        mask_not_cm = (utils.rho2_t(ref_lab) > 1e-5)
        mask_problems |= mask_not_cm
        
        # We start getting M and then K
        M = torch.tensor(
            [0.0] * n, requires_grad=False, dtype=torch.double, device=P.device
        )
        M = torch.unsqueeze(M, 0).repeat(P.shape[0], 1)
        K_t = M.clone()
        Q = torch.zeros_like(P).to(P.device)
        Q[:, -1] = P[:, -1]  # Qn = pn

        # intermediate mass
        for i in range(n, 0, -1):
            j = i - 1
            square_t_P = utils.square_t(torch.sum(P[:, j:n], axis=1))
            M[:, j] = torch.sqrt(square_t_P)

            # new version
            #M[:, j] = torch.nan_to_num(M[:, j], nan=0.0)

            # Remove the final masses to convert back to K
            K_t[:, j] = M[:,j] - torch.sum(masses_t[:,j:])

        # output [0,1] distributed numbers        
        r = torch.zeros(P.shape[0], nDimPhaseSpace, device=P.device)

        for i in range(n, 1, -1):
            j = i - 1  # index for 0-based tensors
            # in the direct algo the u are squared.

            # u = (K_t[:, j]/K_t[:, j-1]) ** 2
            # NB: Removed the square factor from the implementation!
            u = (K_t[:, j]/K_t[:, j-1])

            r[:, j - 1] = (n + 1 - i) * (torch.pow(u, (n - i))) - (n - i) * (
                torch.pow(u, (n + 1 - i))
            )

            Q[:, j - 1] = Q[:, j] + P[:, j - 1]

            P[:, j - 1] = utils.boost_t(P[:, j - 1], -1*utils.boostVector_t(Q[:, j - 1]))

            P_copy = P.clone().to(P.device)
            r[:, n - 5 + 2 * i - 1] = (
                (P_copy[:, j - 1, 3] / torch.sqrt(utils.rho2_t(P_copy[:, j - 1]))) + 1
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

        detjinv_regressed = 0

        ## Improve error handling 
        mask_problems  |= (r<0).any(1) | (r>1).any(1)
        # if (maskr_0.any() or maskr_1.any()):
        #     print("ERROR: r lower than 0")
        #     print(r[maskr_0])
        #     print(r[maskr_1])
        #     exit(0)

        # get x1 x2 in the uniform space
        r_x1x2, jacx1x2 = utils.get_uniform_from_x1x2(x1, x2, target_mass.sum(), E_CM )

        return torch.cat((r, r_x1x2), axis=1), detjinv_regressed*jacx1x2, mask_problems
