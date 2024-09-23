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

    def get_cartesian_comp_noEnergy(particle):
        # px
        Px = particle[:,0]*torch.cos(particle[:,2])
        # py
        Py = particle[:,0]*torch.sin(particle[:,2])
        # pz
        Pz = particle[:,0]*torch.sinh(particle[:,1])
         # E
        return torch.stack((Px, Py, Pz), dim=1)

    def get_particle_sumOnlyMomenta(particles, mass, device):

        sum_momenta = torch.zeros((particles[0].shape[0], 4), device=device)

        for particle in particles:
            sum_momenta[:,1:] = torch.add(sum_momenta[:,1:], particle) # sum px/py/pz contrib from all particles

        sum_momenta_copy = sum_momenta.clone()

        sum_momenta[:,0] = torch.sqrt(sum_momenta_copy[:,1]**2 + sum_momenta_copy[:,2]**2 + sum_momenta_copy[:,3]**2 + mass**2) # find Energy
            
        return sum_momenta

    def get_particle_sumMomenta_andComputeMass(particles, device):

        sum_momenta = torch.zeros((particles[0].shape[0], 4), device=device)

        for particle in particles:
            sum_momenta[:,1:] = torch.add(sum_momenta[:,1:], particle) # sum px/py/pz contrib from all particles
            E_square = torch.sum(torch.square(particle), dim=1)
            sum_momenta[:,0] = torch.add(sum_momenta[:,0], torch.sqrt(E_square)) # find Energy

        sum_momenta_mass = sum_momenta.clone()
        sum_momenta_mass[:,0] = torch.sqrt(sum_momenta[:,0]**2 - torch.sum(torch.square(sum_momenta[:,1:]), dim=1))
        return sum_momenta_mass

    # first particle is the mother
    def get_particle_substract_sumMomenta(particles, device):

        sum_momenta = torch.zeros((particles[0].shape[0], 3), device=device)
        
        sum_momenta = torch.add(sum_momenta, particles[0]) # add mother particle

        for particle in particles[1:]:
            sum_momenta = torch.sub(sum_momenta, particle) # sum px/py/pz contrib from all particles

        return sum_momenta

    def get_ptetaphi_comp(particle):
        particle_pt = (particle[:,1]**2 + particle[:,2]**2)**0.5
        particle_eta = -torch.log(torch.tan(torch.atan2(particle_pt, particle[:,3])/2))
        particle_phi = torch.atan2(particle[:,2], particle[:,1])
        return torch.stack((particle[:,0], particle_pt, particle_eta, particle_phi), dim=1) # E/pt/eta/phi

    def get_ptetaphi_comp_batch(particles):
        particle_pt = (particles[...,1]**2 + particles[...,2]**2)**0.5
        particle_eta = -torch.log(torch.tan(torch.atan2(particle_pt, particles[...,3])/2))
        particle_phi = torch.atan2(particles[...,2], particles[...,1])        
        return torch.stack((particles[...,0], particle_pt, particle_eta, particle_phi), dim=2) # E/pt/eta/phi

    def get_ptetaphi_comp_noEnergy(particle):
        particle_pt = (particle[:,0]**2 + particle[:,1]**2)**0.5
        particle_eta = -torch.log(torch.tan(torch.atan2(particle_pt, particle[:,2])/2))
        particle_phi = torch.atan2(particle[:,1], particle[:,0])
        return torch.stack((particle_pt, particle_eta, particle_phi), dim=1) # E/pt/eta/phi

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
                                  return_both=False, unscale_phi=True):

        boost = cond_X[3] # there was a bug here using -1, now fixed
        Htt = torch.stack((cond_X[0], cond_X[1],cond_X[2]), dim=1)

        if unscale_phi:
            unscaledlog = Htt*log_std_parton + log_mean_parton
            data_regressed = unscaledlog.clone()
            data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
    
            boost_regressed = boost*log_std_boost[1] + log_mean_boost[1]
            # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
            #boost_regressed = torch.sign(boost_regressed)*torch.exp(torch.abs(boost_regressed)-1)       
        else:
            unscaledlog = Htt
            unscaledlog[...,:2] = unscaledlog[...,:2] * log_std_parton[:2] + log_mean_parton[:2]
            
            data_regressed = unscaledlog.clone()
            data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
    
            boost_regressed = boost*log_std_boost[1] + log_mean_boost[1]
            # Do not rescale the pz of the boost, we removed the logscale and kept it only for E

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


    # ALWAYS PAY ATTENTION TO THE ORDER OF PARTONS
    def get_HttISR_fromlab_decayPartons(decay_partons, boost,
                                        log_mean_parton, log_std_parton,
                                      log_mean_boost, log_std_boost,
                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                      device, cartesian=True, eps=1e-4,
                                      return_both=False, pt_cut=None, unscale_phi=True):


        if unscale_phi:
            unscaledlog = decay_partons*log_std_parton + log_mean_parton
        else:
            unscaledlog = decay_partons[...,:2]*log_std_parton[:2] + log_mean_parton[:2]
            unscaledlog = torch.cat((unscaledlog, decay_partons[...,2:]), dim=2)

        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]

        unscaled_decayPartons = []
        for i in range(decay_partons.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            unscaled_decayPartons.append(Compute_ParticlesTensor.get_cartesian_comp_noEnergy(data_regressed[:,i]))

        higgs_indices = [2,3]
        thad_indices = [4,5,6]
        tlep_indices =  [0,1,7]

        higgs_cartesian = Compute_ParticlesTensor.get_particle_sumOnlyMomenta([unscaled_decayPartons[i] for i in higgs_indices], M_HIGGS, device)
        thad_cartesian = Compute_ParticlesTensor.get_particle_sumOnlyMomenta([unscaled_decayPartons[i] for i in thad_indices], M_TOP, device)
        tlep_cartesian = Compute_ParticlesTensor.get_particle_sumOnlyMomenta([unscaled_decayPartons[i] for i in tlep_indices], M_TOP, device)

        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_py = -(higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        gluon_pz = boost_pz - (higgs_cartesian[:,3] + thad_cartesian[:,3] + tlep_cartesian[:,3])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps

        # we need it for boost that is always cartesian
        gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)

        if not cartesian or return_both:
            higgs = Compute_ParticlesTensor.get_ptetaphi_comp(higgs_cartesian)
            thad = Compute_ParticlesTensor.get_ptetaphi_comp(thad_cartesian)
            tlep = Compute_ParticlesTensor.get_ptetaphi_comp(tlep_cartesian)
            gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp(gluon_cartesian)

        # order: by default higgs/thad/tlep/glISR
        # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
        # I changed to do this permutation of the particles inside the get_PS
            
        if not return_both:
            if cartesian:
                data_regressed_new = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)

                # scale everything
                data_regressed_new[...,1:] = (data_regressed_new[...,1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
            else:
                data_regressed_new = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
                
                # scale everything
                data_regressed_new[...,1] = torch.log(data_regressed_new[...,1] + 1) # log(pt)
                if unscale_phi:
                    data_regressed_new[...,1:] = (data_regressed_new[...,1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                else:
                    data_regressed_new[...,1:3] = (data_regressed_new[...,1:3] - log_mean_parton_Hthadtlep[:2])/log_std_parton_Hthadtlep[:2]
                    
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
        else:
            data_regressed_new_cart = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
            data_regressed_new_ptetaphi = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
            
            #return data_regressed_new_cart, data_regressed_new_ptetaphi, boost_regressed_new
            return data_regressed_new_cart, data_regressed_new_ptetaphi

    # ALWAYS PAY ATTENTION TO THE ORDER OF PARTONS
    def get_HttISR_fromlab_decayPartons_withMasses(decay_partons, boost,
                                        log_mean_parton, log_std_parton,
                                      log_mean_boost, log_std_boost,
                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                      device, cartesian=True, eps=1e-4,
                                      return_both=False, pt_cut=None, unscale_phi=True):


        if unscale_phi:
            unscaledlog = decay_partons*log_std_parton + log_mean_parton
        else:
            unscaledlog = decay_partons[...,:2]*log_std_parton[:2] + log_mean_parton[:2]
            unscaledlog = torch.cat((unscaledlog, decay_partons[...,2:]), dim=2)
            
        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]

        unscaled_decayPartons = []
        for i in range(decay_partons.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            unscaled_decayPartons.append(Compute_ParticlesTensor.get_cartesian_comp_noEnergy(data_regressed[:,i]))

        higgs_indices = [2,3]
        thad_indices = [4,5,6]
        tlep_indices =  [0,1,7]
        
        higgs_cartesian = Compute_ParticlesTensor.get_particle_sumMomenta_andComputeMass([unscaled_decayPartons[i] for i in higgs_indices], device)
        thad_cartesian = Compute_ParticlesTensor.get_particle_sumMomenta_andComputeMass([unscaled_decayPartons[i] for i in thad_indices], device)
        tlep_cartesian = Compute_ParticlesTensor.get_particle_sumMomenta_andComputeMass([unscaled_decayPartons[i] for i in tlep_indices], device)

        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_py = -(higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        gluon_pz = boost_pz - (higgs_cartesian[:,3] + thad_cartesian[:,3] + tlep_cartesian[:,3])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps

        # we need it for boost that is always cartesian
        gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)

        if not cartesian or return_both:
            higgs = Compute_ParticlesTensor.get_ptetaphi_comp(higgs_cartesian)
            thad = Compute_ParticlesTensor.get_ptetaphi_comp(thad_cartesian)
            tlep = Compute_ParticlesTensor.get_ptetaphi_comp(tlep_cartesian)
            gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp(gluon_cartesian)

        # order: by default higgs/thad/tlep/glISR
        # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
        # I changed to do this permutation of the particles inside the get_PS
            
        if not return_both:
            if cartesian:
                data_regressed_new = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
                
                 # scale everything
                data_regressed_new[...,1:] = (data_regressed_new[...,1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
            else:
                data_regressed_new = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)

                 # scale everything
                data_regressed_new[...,1] = torch.log(data_regressed_new[...,1] + 1) # log(pt)
                if unscale_phi:
                    data_regressed_new[...,1:] = (data_regressed_new[...,1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                else:
                    data_regressed_new[...,1:3] = (data_regressed_new[...,1:3] - log_mean_parton_Hthadtlep[:2])/log_std_parton_Hthadtlep[:2]
                
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
        else:
            data_regressed_new_cart = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
            data_regressed_new_ptetaphi = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
            #return data_regressed_new_cart, data_regressed_new_ptetaphi, boost_regressed_new
            return data_regressed_new_cart, data_regressed_new_ptetaphi

    # ALWAYS PAY ATTENTION TO THE ORDER OF PARTONS
    # partons order: higgs, b1_higgs, tlep, tlep_e, tlep_b, thad, thad_b, thad_q1
    def get_HttISR_fromlab_decayPartons_propagators(decay_partons, boost,
                                        log_mean_parton, log_std_parton,
                                      log_mean_boost, log_std_boost,
                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                      device, cartesian=True, eps=1e-4,
                                      return_both=False, pt_cut=None, unscale_phi=True):


        unscaledlog = decay_partons.clone()
        if unscale_phi:
            unscaledlog[:,[0,2,5]] = decay_partons[:,[0,2,5]]*log_std_parton_Hthadtlep + log_mean_parton_Hthadtlep
            unscaledlog[:,[1,3,4,6,7]] = decay_partons[:,[1,3,4,6,7]]*log_std_parton + log_mean_parton
        else:
            unscaledlog[:,[0,2,5], :2] = decay_partons[:,[0,2,5], :2]*log_std_parton_Hthadtlep[:2] + log_mean_parton_Hthadtlep[:2]
            unscaledlog[:,[1,3,4,6,7], :2] = decay_partons[:,[1,3,4,6,7], :2]*log_std_parton[:2] + log_mean_parton[:2]
            
        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]

        unscaled_decayPartons = []
        for i in range(decay_partons.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            unscaled_decayPartons.append(Compute_ParticlesTensor.get_cartesian_comp_noEnergy(data_regressed[:,i]))

        higgs_cartesian = unscaled_decayPartons[0]
        tlep_cartesian = unscaled_decayPartons[2]
        thad_cartesian = unscaled_decayPartons[5]

        higgs_indices = [0,1]
        tlep_indices =  [2,3,4]
        thad_indices = [5,6,7]
        
        higgs_b2_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in higgs_indices], device)
        thad_q3_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in thad_indices], device)
        tlep_nu_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in tlep_indices], device)

        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,0] + thad_cartesian[:,0] + tlep_cartesian[:,0])
        gluon_py = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_pz = boost_pz - (higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        #gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps

        mask = torch.abs(decay_partons[:,0,0] - 0.6693) < 0.01
        mask_2 = torch.abs(decay_partons[:,0,1] - 0.3717) < 0.01
        mask = torch.logical_and(mask, mask_2)

        if False:
            if (torch.count_nonzero(mask)):
                print()
                print()
                print('DEBUG ================================')
                #print(f'partons: mean - {log_mean_parton} & std - {log_std_parton}')
                #print(f'propagators: mean - {log_mean_parton_Hthadtlep} & std - {log_std_parton_Hthadtlep}')
                #print(f'boost: mean - {log_mean_boost} & std - {log_std_boost}')
                
                print(torch.count_nonzero(mask))
                print(data_regressed[mask])
                print(boost_regressed[mask])
    
                print(f'H_b2   : {higgs_b2_cartesian[mask]}')
                print(f'thad_q3: {thad_q3_cartesian[mask]}')
                print(f'tlep_nu: {tlep_nu_cartesian[mask]}')
                print()
                
                print(f'H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask]}, ISR: {gluon_px[mask]} {gluon_py[mask]} {gluon_pz[mask]}')
                print(decay_partons[mask])
                #print(f'ISR py: H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask][0,1]}, ISR: {gluon_py[mask]}')
                #print(f'ISR pz: H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask][0,2]}, ISR: {gluon_pz[mask]}')
                print()
                print()

        # we need it for boost that is always cartesian
        #gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
        gluon_cartesian = torch.stack((gluon_px, gluon_py, gluon_pz), dim=1)

        if not cartesian or return_both:
            higgs_b2 = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(higgs_b2_cartesian)
            thad_q3 = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(thad_q3_cartesian)
            tlep_nu = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(tlep_nu_cartesian)
            gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(gluon_cartesian)

        # TODO: change this after you solve the problem in general with pz conservation etc for all particles 
        mask_px = torch.abs(gluon_px) < 0.1
        mask_py = torch.abs(gluon_py) < 0.1
        mask_pz = torch.abs(gluon_pz) < 0.1
        
        mask_1 = torch.logical_and(mask_px, mask_py)
        mask_2 = torch.logical_and(mask_1, mask_pz)
        if torch.count_nonzero(mask_2) > 0:
            gluon_padding = torch.tensor([0.0010, 0.0000, 0.0000], device=device)
            gluon_ptetaphi = torch.where(mask_2[:,None], gluon_padding[None,:], gluon_ptetaphi)

        # order: by default higgs/thad/tlep/glISR
        # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
        # I changed to do this permutation of the particles inside the get_PS
        if not return_both:
            if cartesian:
                data_regressed_new = torch.stack((higgs_b2_cartesian, thad_q3_cartesian, tlep_nu_cartesian, gluon_cartesian), dim=1)
                
                 # scale everything
                data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
            else:
                data_regressed_new = torch.stack((higgs_b2, thad_q3, tlep_nu, gluon_ptetaphi), dim=1)

                 # scale everything
                data_regressed_new[...,0] = torch.log(data_regressed_new[...,0] + 1) # log(pt)
                if unscale_phi:
                    data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                else:
                    data_regressed_new[...,:2] = (data_regressed_new[...,:2] - log_mean_parton[:2])/log_std_parton[:2]
                
                #return data_regressed_new, boost_regressed_new
                return data_regressed_new
        else:
            data_regressed_new_cart = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
            data_regressed_new_ptetaphi = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
            #return data_regressed_new_cart, data_regressed_new_ptetaphi, boost_regressed_new
            return data_regressed_new_cart, data_regressed_new_ptetaphi


    # ALWAYS PAY ATTENTION TO THE ORDER OF PARTONS
    # partons order: higgs, b1_higgs, tlep, tlep_e, tlep_b, thad, thad_b, thad_q1
    def get_HttISR_fromlab_decayPartons_propagators_experimental(decay_partons, boost,
                                        log_mean_parton, log_std_parton,
                                      log_mean_boost, log_std_boost,
                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                      device, cartesian=True, eps=1e-4,
                                      return_both=False, pt_cut=None, unscale_phi=True):


        unscaledlog = decay_partons.clone()
        if unscale_phi:
            unscaledlog[:,[0,2,5]] = decay_partons[:,[0,2,5]]*log_std_parton_Hthadtlep + log_mean_parton_Hthadtlep
            unscaledlog[:,[1,3,4,6,7]] = decay_partons[:,[1,3,4,6,7]]*log_std_parton + log_mean_parton
        else:
            unscaledlog[:,[0,2,5], :2] = decay_partons[:,[0,2,5], :2]*log_std_parton_Hthadtlep[:2] + log_mean_parton_Hthadtlep[:2]
            unscaledlog[:,[1,3,4,6,7], :2] = decay_partons[:,[1,3,4,6,7], :2]*log_std_parton[:2] + log_mean_parton[:2]
            
        data_regressed = unscaledlog.clone()
        data_regressed[:,:,0] = torch.exp(torch.abs(unscaledlog[:,:,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]

        unscaled_decayPartons = []
        mass = torch.tensor([125.25, 0., 172.5, 0., 0., 172.5, 0., 0.], device=device)
        for i in range(decay_partons.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            unscaled_decayPartons.append(Compute_ParticlesTensor.get_cartesian_comp(data_regressed[:,i], mass))

        higgs_cartesian = unscaled_decayPartons[0]
        tlep_cartesian = unscaled_decayPartons[2]
        thad_cartesian = unscaled_decayPartons[5]

        # boost b1 with (-px, -py, -pz)
        b1_CM = utils.boost_tt(unscaled_decayPartons[1], -1*higgs_cartesian, gamma=-1.0)

        if False:

            higgs_indices = [0,1]
            tlep_indices =  [2,3,4]
            thad_indices = [5,6,7]
            
            higgs_b2_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in higgs_indices], device)
            thad_q3_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in thad_indices], device)
            tlep_nu_cartesian = Compute_ParticlesTensor.get_particle_substract_sumMomenta([unscaled_decayPartons[i] for i in tlep_indices], device)
    
            # Now gluon = boost - partons
            gluon_px = -(higgs_cartesian[:,0] + thad_cartesian[:,0] + tlep_cartesian[:,0])
            gluon_py = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
            gluon_pz = boost_pz - (higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
            # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
            #gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps
    
            mask = torch.abs(decay_partons[:,0,0] - 0.6693) < 0.01
            mask_2 = torch.abs(decay_partons[:,0,1] - 0.3717) < 0.01
            mask = torch.logical_and(mask, mask_2)
    
            if False:
                if (torch.count_nonzero(mask)):
                    print()
                    print()
                    print('DEBUG ================================')
                    #print(f'partons: mean - {log_mean_parton} & std - {log_std_parton}')
                    #print(f'propagators: mean - {log_mean_parton_Hthadtlep} & std - {log_std_parton_Hthadtlep}')
                    #print(f'boost: mean - {log_mean_boost} & std - {log_std_boost}')
                    
                    print(torch.count_nonzero(mask))
                    print(data_regressed[mask])
                    print(boost_regressed[mask])
        
                    print(f'H_b2   : {higgs_b2_cartesian[mask]}')
                    print(f'thad_q3: {thad_q3_cartesian[mask]}')
                    print(f'tlep_nu: {tlep_nu_cartesian[mask]}')
                    print()
                    
                    print(f'H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask]}, ISR: {gluon_px[mask]} {gluon_py[mask]} {gluon_pz[mask]}')
                    print(decay_partons[mask])
                    #print(f'ISR py: H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask][0,1]}, ISR: {gluon_py[mask]}')
                    #print(f'ISR pz: H: {higgs_cartesian[mask]}, thad: {thad_cartesian[mask]}, tlep: {tlep_cartesian[mask][0,2]}, ISR: {gluon_pz[mask]}')
                    print()
                    print()
    
            # we need it for boost that is always cartesian
            #gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
            gluon_cartesian = torch.stack((gluon_px, gluon_py, gluon_pz), dim=1)
    
            if not cartesian or return_both:
                higgs_b2 = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(higgs_b2_cartesian)
                thad_q3 = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(thad_q3_cartesian)
                tlep_nu = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(tlep_nu_cartesian)
                gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp_noEnergy(gluon_cartesian)
    
            # TODO: change this after you solve the problem in general with pz conservation etc for all particles 
            mask_px = torch.abs(gluon_px) < 0.1
            mask_py = torch.abs(gluon_py) < 0.1
            mask_pz = torch.abs(gluon_pz) < 0.1
            
            mask_1 = torch.logical_and(mask_px, mask_py)
            mask_2 = torch.logical_and(mask_1, mask_pz)
            if torch.count_nonzero(mask_2) > 0:
                gluon_padding = torch.tensor([0.0010, 0.0000, 0.0000], device=device)
                gluon_ptetaphi = torch.where(mask_2[:,None], gluon_padding[None,:], gluon_ptetaphi)
    
            # order: by default higgs/thad/tlep/glISR
            # but for glISR negative masses in Rambo ==> can switch the order of the data_regressed
            # I changed to do this permutation of the particles inside the get_PS
            if not return_both:
                if cartesian:
                    data_regressed_new = torch.stack((higgs_b2_cartesian, thad_q3_cartesian, tlep_nu_cartesian, gluon_cartesian), dim=1)
                    
                     # scale everything
                    data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                    
                    #return data_regressed_new, boost_regressed_new
                    return data_regressed_new
                else:
                    data_regressed_new = torch.stack((higgs_b2, thad_q3, tlep_nu, gluon_ptetaphi), dim=1)
    
                     # scale everything
                    data_regressed_new[...,0] = torch.log(data_regressed_new[...,0] + 1) # log(pt)
                    if unscale_phi:
                        data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                    else:
                        data_regressed_new[...,:2] = (data_regressed_new[...,:2] - log_mean_parton[:2])/log_std_parton[:2]
                    
                    #return data_regressed_new, boost_regressed_new
                    return data_regressed_new
            else:
                data_regressed_new_cart = torch.stack((higgs_cartesian, thad_cartesian, tlep_cartesian, gluon_cartesian), dim=1)
                data_regressed_new_ptetaphi = torch.stack((higgs, thad, tlep, gluon_ptetaphi), dim=1)
                #return data_regressed_new_cart, data_regressed_new_ptetaphi, boost_regressed_new
                return data_regressed_new_cart, data_regressed_new_ptetaphi

    def higgs_decayProducts(higgs_momenta, higgs_angles, higgs_mass=125.25, ptetaphi=True, device=torch.device('cpu')):

    
        E_b1 = torch.tensor((higgs_mass / 2.), device=device).repeat(higgs_angles[...,0:1].shape)
        
        # E = mass / 2 and pt = E/cosh(eta)
        b1 = torch.cat((E_b1, E_b1 / torch.cosh(higgs_angles[..., None, 0]), higgs_angles[...,None, 0], higgs_angles[...,None, 1]), dim=1)
    
        # b2 = "-b1" in CM
        b2 = torch.cat((E_b1, E_b1 / torch.cosh(higgs_angles[..., None, 0]), -higgs_angles[..., None, 0], np.pi+higgs_angles[..., None, 1]), dim=1)
    
        b1 = Compute_ParticlesTensor.get_cartesian_comp(b1[...,1:], 0.0)
        b2 = Compute_ParticlesTensor.get_cartesian_comp(b2[...,1:], 0.0)
    
        # boost in lab frame using higgs kinematics
        b1_lab = utils.boost_t(b1, utils.boostVector_t(higgs_momenta), gamma=-1.0)
        b2_lab = utils.boost_t(b2, utils.boostVector_t(higgs_momenta), gamma=-1.0)
    
        if ptetaphi:
            higgs_momenta = Compute_ParticlesTensor.get_ptetaphi_comp(higgs_momenta)
            b1_lab = Compute_ParticlesTensor.get_ptetaphi_comp(b1_lab)
            b2_lab = Compute_ParticlesTensor.get_ptetaphi_comp(b2_lab)
    
        higgs_b1_b2 = torch.stack((higgs_momenta, b1_lab, b2_lab), dim=1)
    
        return higgs_b1_b2
    
    def top_decayProducts(top_momenta, top_b_angles, top_W_angles, top_mass=172.5, W_mass = 80.4, b_mass = 0.0, ptetaphi=True, device=torch.device('cpu')):
        # this time particles are massive, not massless
        E_b = torch.tensor(((top_mass**2 - W_mass**2 + b_mass**2)/2./top_mass), device=device).repeat(top_b_angles[...,0:1].shape)
        E_W = torch.tensor(((top_mass**2 + W_mass**2 - b_mass**2)/2./top_mass), device=device).repeat(top_b_angles[...,0:1].shape)
    
        # E_b = pt*cosh(eta)... formulas here https://indico.cern.ch/event/391122/contributions/928962/attachments/782786/1073126/twoBodyDecay.pdf
        # W = onshell = 80.4 GeV
        b = torch.cat((E_b, torch.sqrt(E_b**2 - b_mass**2) / torch.cosh(top_b_angles[..., None, 0]), top_b_angles[..., None, 0], top_b_angles[..., None, 1]), dim=1)
    
        # pt for W = pt for b
        mask =  top_b_angles[..., 1] > 0
        W = torch.cat((E_W, torch.sqrt(E_b**2 - b_mass**2) / torch.cosh(top_b_angles[..., None, 0]), -top_b_angles[...,None, 0], np.pi+top_b_angles[..., None, 1]), dim=1)
        W[...,3] = torch.where(mask, W[...,3] - 2*np.pi, W[...,3])
        
        E_W_rest = torch.tensor([W_mass], device=device).repeat(top_W_angles[...,0:1].shape)
    
        # W decay in massless particles
        W_q1 = torch.cat((E_W_rest / 2., E_W_rest / 2. / torch.cosh(top_W_angles[..., None, 0]), top_W_angles[..., None, 0], top_W_angles[..., None, 1]), dim=1)
    
        mask =  top_W_angles[..., 1] > 0
        W_q2 = torch.cat((E_W_rest / 2., E_W_rest / 2. / torch.cosh(top_W_angles[..., 0:1]), -top_W_angles[..., 0:1], np.pi+top_W_angles[..., 1:2]), dim=1)
        W_q2[...,3] = torch.where(mask, W_q2[...,3] - 2*np.pi, W_q2[...,3])
    
        # move to E/px/py/pz
        b = Compute_ParticlesTensor.get_cartesian_comp(b[...,1:], b_mass)
        W_q1 = Compute_ParticlesTensor.get_cartesian_comp(W_q1[...,1:], 0.0)
        W_q2 = Compute_ParticlesTensor.get_cartesian_comp(W_q2[...,1:], 0.0)
        W = Compute_ParticlesTensor.get_cartesian_comp(W[...,1:], W_mass)
    
        # boost in lab frame
        b_lab = utils.boost_t(b, utils.boostVector_t(top_momenta), gamma=-1.0)
        W_lab = utils.boost_t(W, utils.boostVector_t(top_momenta), gamma=-1.0)
        W_q1_lab = utils.boost_t(W_q1, utils.boostVector_t(W_lab), gamma=-1.0)
        W_q2_lab = utils.boost_t(W_q2, utils.boostVector_t(W_lab), gamma=-1.0)
    
        if ptetaphi:
            top_momenta = Compute_ParticlesTensor.get_ptetaphi_comp(top_momenta)
            b_lab = Compute_ParticlesTensor.get_ptetaphi_comp(b_lab)
            W_q1_lab = Compute_ParticlesTensor.get_ptetaphi_comp(W_q1_lab)
            W_q2_lab = Compute_ParticlesTensor.get_ptetaphi_comp(W_q2_lab)
    
        top_b_q1_q2 = torch.stack((top_momenta, b_lab, W_q1_lab, W_q2_lab), dim=1)
    
        return top_b_q1_q2

    # Inputs:
    # 1. propagators_kinematics --> H thad tlep in the lab: needs to be scaled in log(pt + 1) and eta
    # 2. higgs_angles --> in the CM --> UNSCALED
    # 3. thad_b_angles --> same as above
    # 4. boost -- only [E, pz] components: it is scaled in log(E + 1) and pz
    # 5. ptetaphi = True for output in ptetaphi
    # 6. final scaling --> scale or not the full parton event
    # Outputs:
    # full parton level event --> in a specific order --> with 'ptetaphi' components & 'scaled'
    def get_decayPartons_fromlab_propagators_angles(propagators_kinematics, 
                                                      higgs_angles,
                                                      thad_b_angles,
                                                      thad_W_angles,
                                                      tlep_b_angles,
                                                      tlep_W_angles,
                                                      boost,
                                                      log_mean_parton_lab, log_std_parton_lab,
                                                      log_mean_boost, log_std_boost,
                                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                                      device,
                                                      higgs_mass=125.25,
                                                      thad_mass=172.5,
                                                      tlep_mass=172.5,
                                                      W_had_mass=80.4,
                                                      W_lep_mass=80.4,
                                                      b_mass=0.0,
                                                      ptetaphi=True, eps=1e-4,
                                                      pt_cut=None, unscale_phi=False, debug=False,
                                                      final_scaling=True):
    
            
        if unscale_phi:
            propagators_kinematics = propagators_kinematics*log_std_parton_Hthadtlep + log_mean_parton_Hthadtlep   
        else:
            propagators_kinematics[..., :2] = propagators_kinematics[..., :2]*log_std_parton_Hthadtlep[:2] + log_mean_parton_Hthadtlep[:2]
            
        #data_regressed = unscaledlog.clone()
        propagators_kinematics_copy = propagators_kinematics.clone()
        propagators_kinematics_copy[...,0] = torch.exp(torch.abs(propagators_kinematics[...,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]
        #boost: (log(E), pz)
    
        propagators_4momenta = []
        mass = torch.tensor([higgs_mass, tlep_mass, thad_mass], device=device)
        for i in range(propagators_kinematics.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            propagators_4momenta.append(Compute_ParticlesTensor.get_cartesian_comp(propagators_kinematics_copy[:,i], mass[i]))
    
        higgs_cartesian = propagators_4momenta[0]
        tlep_cartesian = propagators_4momenta[1]
        thad_cartesian = propagators_4momenta[2]
    
        higgs_b1_b2 = Compute_ParticlesTensor.higgs_decayProducts(higgs_cartesian, higgs_angles, higgs_mass=higgs_mass, ptetaphi=ptetaphi, device=device)
        topHad_b_q1_q2 = Compute_ParticlesTensor.top_decayProducts(thad_cartesian, thad_b_angles, thad_W_angles, top_mass=thad_mass,
                                                                   W_mass = W_had_mass, b_mass=b_mass, ptetaphi=ptetaphi, device=device)
        topLep_b_e_nu = Compute_ParticlesTensor.top_decayProducts(tlep_cartesian, tlep_b_angles, tlep_W_angles, top_mass=tlep_mass,
                                                                  W_mass = W_lep_mass, b_mass=b_mass, ptetaphi=ptetaphi, device=device)    
    
        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_py = -(higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        gluon_pz = boost_pz - (higgs_cartesian[:,3] + thad_cartesian[:,3] + tlep_cartesian[:,3])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps
    
        # we need it for boost that is always cartesian
        #gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
        gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)

        if ptetaphi:
    
            gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp(gluon_cartesian)
        
           # TODO: change this after you solve the problem in general with pz conservation etc for all particles 
            mask_px = torch.abs(gluon_px) < 0.1
            mask_py = torch.abs(gluon_py) < 0.1
            mask_pz = torch.abs(gluon_pz) < 0.1
            
            mask_1 = torch.logical_and(mask_px, mask_py)
            mask_2 = torch.logical_and(mask_1, mask_pz)
            if torch.count_nonzero(mask_2) > 0:
                gluon_padding = torch.tensor([0.0010, 0.0010, 0.0000, 0.0000], device=device)
                gluon_ptetaphi = torch.where(mask_2[:,None], gluon_padding[None,:], gluon_ptetaphi)

        if final_scaling:
            if not ptetaphi:

                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_cartesian[:,None,:]), dim=1)
                
                 # scale everything
                data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                                
            else:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_ptetaphi[:,None,:]), dim=1)     
                
                 # scale everything
                data_regressed_new[...,1] = torch.log(data_regressed_new[...,1] + 1) # log(pt)
                if unscale_phi:
                    data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:] = (data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:] - log_mean_parton_lab)/log_std_parton_lab
                    data_regressed_new[...,[0,3,7],1:] = (data_regressed_new[...,[0,3,7],1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                else:
                    data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:3] = (data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:3] - log_mean_parton_lab[:2])/log_std_parton_lab[:2]
                    data_regressed_new[...,[0,3,7],1:3] = (data_regressed_new[...,[0,3,7],1:3] - log_mean_parton_Hthadtlep[:2])/log_std_parton_Hthadtlep[:2]

        else:
            if not ptetaphi:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_cartesian[:,None,:]), dim=1)
                
            else:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_ptetaphi[:,None,:]), dim=1)   


        return data_regressed_new

    # Inputs:
    # 1. propagators_kinematics --> H thad tlep in the lab: needs to be scaled in log(pt + 1) and eta
    # 2. higgs_angles --> in the CM --> UNSCALED
    # 3. thad_b_angles --> same as above
    # 4. boost -- only [E, pz] components: it is scaled in log(E + 1) and pz
    # 5. ptetaphi = True for output in ptetaphi
    # 6. final scaling --> scale or not the full parton event
    # Outputs:
    # full parton level event --> in a specific order --> with 'ptetaphi' components & 'scaled'
    def get_decayPartons_fromlab_propagators_angles_masses(propagators_kinematics, 
                                                      higgs_angles,
                                                      thad_b_angles,
                                                      thad_W_angles,
                                                      tlep_b_angles,
                                                      tlep_W_angles,
                                                      boost,
                                                      log_mean_parton_lab, log_std_parton_lab,
                                                      log_mean_boost, log_std_boost,
                                                      log_mean_parton_Hthadtlep, log_std_parton_Hthadtlep,
                                                      device,
                                                      higgs_mass=125.25,
                                                      thad_mass=172.5,
                                                      tlep_mass=172.5,
                                                      W_had_mass=80.4,
                                                      W_lep_mass=80.4,
                                                      b_mass=0.0,
                                                      ptetaphi=True, eps=1e-4,
                                                      pt_cut=None, unscale_phi=False, debug=False,
                                                      final_scaling=True):
    
            
        if unscale_phi:
            propagators_kinematics = propagators_kinematics*log_std_parton_Hthadtlep + log_mean_parton_Hthadtlep   
        else:
            propagators_kinematics[..., :2] = propagators_kinematics[..., :2]*log_std_parton_Hthadtlep[:2] + log_mean_parton_Hthadtlep[:2]
            
        #data_regressed = unscaledlog.clone()
        propagators_kinematics_copy = propagators_kinematics.clone()
        propagators_kinematics_copy[...,0] = torch.exp(torch.abs(propagators_kinematics[...,0])) - 1
        
        boost_regressed = boost*log_std_boost + log_mean_boost # boost and mean contains only E and pz
        
        boost_regr_copy = boost_regressed.clone()
        boost_regressed[:,:,0] = torch.exp(torch.abs(boost_regr_copy[:,:,0])) - 1
        # Do not rescale the pz of the boost, we removed the logscale and kept it only for E
        boost_pz = boost_regressed[:,0,1]
        #boost: (log(E), pz)
    
        propagators_4momenta = []
        mass = torch.tensor([higgs_mass[0], tlep_mass[0], thad_mass[0]], device=device)
        for i in range(propagators_kinematics.shape[1]):
            # find px py pz from decay partons -> then find higgs = px1 + px2 then find higgs energy as higgs sqrt(px^2 + .. + M^2)
            propagators_4momenta.append(Compute_ParticlesTensor.get_cartesian_comp(propagators_kinematics_copy[:,i], mass[i]))
    
        higgs_cartesian = propagators_4momenta[0]
        tlep_cartesian = propagators_4momenta[1]
        thad_cartesian = propagators_4momenta[2]
    
        higgs_b1_b2 = Compute_ParticlesTensor.higgs_decayProducts(higgs_cartesian, higgs_angles, higgs_mass=higgs_mass, ptetaphi=ptetaphi, device=device)
        topHad_b_q1_q2 = Compute_ParticlesTensor.top_decayProducts(thad_cartesian, thad_b_angles, thad_W_angles, top_mass=thad_mass,
                                                                   W_mass = W_had_mass, b_mass=b_mass, ptetaphi=ptetaphi, device=device)
        topLep_b_e_nu = Compute_ParticlesTensor.top_decayProducts(tlep_cartesian, tlep_b_angles, tlep_W_angles, top_mass=tlep_mass,
                                                                  W_mass = W_lep_mass, b_mass=b_mass, ptetaphi=ptetaphi, device=device)    
    
        # Now gluon = boost - partons
        gluon_px = -(higgs_cartesian[:,1] + thad_cartesian[:,1] + tlep_cartesian[:,1])
        gluon_py = -(higgs_cartesian[:,2] + thad_cartesian[:,2] + tlep_cartesian[:,2])
        gluon_pz = boost_pz - (higgs_cartesian[:,3] + thad_cartesian[:,3] + tlep_cartesian[:,3])
        # Now, also energy needs to be correct, but we need to be carefull about the mass of the gluon
        gluon_E = torch.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2) + eps
    
        # we need it for boost that is always cartesian
        #gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
        gluon_cartesian = torch.stack((gluon_E, gluon_px, gluon_py, gluon_pz), dim=1)
    
        gluon_ptetaphi = Compute_ParticlesTensor.get_ptetaphi_comp(gluon_cartesian)
    
       # TODO: change this after you solve the problem in general with pz conservation etc for all particles 
        mask_px = torch.abs(gluon_px) < 0.1
        mask_py = torch.abs(gluon_py) < 0.1
        mask_pz = torch.abs(gluon_pz) < 0.1
        
        mask_1 = torch.logical_and(mask_px, mask_py)
        mask_2 = torch.logical_and(mask_1, mask_pz)
        if torch.count_nonzero(mask_2) > 0:
            gluon_padding = torch.tensor([0.0010, 0.0010, 0.0000, 0.0000], device=device)
            gluon_ptetaphi = torch.where(mask_2[:,None], gluon_padding[None,:], gluon_ptetaphi)

        if final_scaling:
            if not ptetaphi:

                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_cartesian[:,None,:]), dim=1)
                
                 # scale everything
                data_regressed_new = (data_regressed_new - log_mean_parton)/log_std_parton
                                
            else:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_ptetaphi[:,None,:]), dim=1)     
                
                 # scale everything
                data_regressed_new[...,1] = torch.log(data_regressed_new[...,1] + 1) # log(pt)
                if unscale_phi:
                    data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:] = (data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:] - log_mean_parton_lab)/log_std_parton_lab
                    data_regressed_new[...,[0,3,7],1:] = (data_regressed_new[...,[0,3,7],1:] - log_mean_parton_Hthadtlep)/log_std_parton_Hthadtlep
                else:
                    data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:3] = (data_regressed_new[...,[1,2,4,5,6,8,9,10,11],1:3] - log_mean_parton_lab[:2])/log_std_parton_lab[:2]
                    data_regressed_new[...,[0,3,7],1:3] = (data_regressed_new[...,[0,3,7],1:3] - log_mean_parton_Hthadtlep[:2])/log_std_parton_Hthadtlep[:2]

        else:
            if not ptetaphi:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_cartesian[:,None,:]), dim=1)
                
            else:
                data_regressed_new = torch.cat((higgs_b1_b2, topHad_b_q1_q2, topLep_b_e_nu, gluon_ptetaphi[:,None,:]), dim=1)   


        return data_regressed_new


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
