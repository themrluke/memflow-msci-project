from . import rambo_generator
from . import utils
import torch
from particle import Particle


# def get_pdfxQ2(pdf, pdg, x, scale2):
#     """Call the PDF and return the corresponding density."""
#     if pdf is None:
#         return torch.ones_like(x)

#     if pdg not in [21] and abs(pdg) not in range(1, 7):
#         return torch.ones_like(x)

#     # Call to lhapdf API
#     f = pdf.xfxQ2(
#         [pdg],
#         tf.convert_to_tensor(x.cpu(), dtype=tf.float64),
#         tf.convert_to_tensor(scale2.cpu(), dtype=tf.float64),
#     )
#     return torch.tensor(f.numpy(), dtype=torch.double, device=x.device)

# def get_pdf_weight(x1, x2, pdgs, pdf):
#     q2 = torch.ones_like(x1)*91.88**2
#     return get_pdfxQ2(pdf, pdgs[0], x1, q2)*\
#          get_pdfxQ2(pdf, pdgs[1], x2, q2)
        
    


class PhaseSpace:
    def __init__(self, collider_energy, initial_pdgs, final_pdgs, final_masses, pdf=None, dev=None):
        if dev == None:
            self.dev = (
                torch.device("cuda:" + str(0))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.dev = dev
        self.collider_energy = collider_energy
        self.initial_pdgs = initial_pdgs
        self.final_pdgs = final_pdgs
        self.final_masses = final_masses.to(dev)
        self.final_state_mass = torch.sum(self.final_masses)
        self.pdf = pdf
        # init the generatoer
        self.generator = rambo_generator.FlatInvertiblePhasespace(
            [0.0, 0.0],  # initial particle mass
            self.final_masses,
            self.collider_energy,
            pdf=pdf,
            pdf_active=True,
            uniform_x1x2=True,
            dev=self.dev
        )

    def generate_random_phase_space_points(self, N,
                                           pT_mincut=-1,
                                           delR_mincut=-1,
                                           rap_maxcut=-1):
        '''
        Generate N random phase space points from the CM of collider_energy energy,
        representing n final state particles with final_masses mass.

        '''
        ps_rand = torch.rand(N, self.generator.nDimPhaseSpace + 2, device=self.dev)
        # for the output we return the [0,1] uniform x1x2 representation
        momenta, weight, x1, x2 = self.generator.generateKinematics_batch(
            ps_rand, pdgs=self.initial_pdgs,
            pT_mincut=pT_mincut, delR_mincut=delR_mincut,
            rap_maxcut=rap_maxcut
        )
        return ps_rand, momenta, weight, x1, x2

    def get_momenta_from_ps(self,
                            points, #(3n-4) + 2
                            pT_mincut=-1,
                            delR_mincut=-1,
                            rap_maxcut=-1):
        '''
        The returned 4-vector momenta are in order:
        - gluon1, gluon2 in the CM (not in the lab frame!!)
        - final state particles in the requested order

        Both the incoming and outgoing particles are in the CM to be able to
        use the output of this transformation for Matrix Element computation. 
        The x1 and x2 fractions are returned to build the lab frame boost.
        '''
        momenta, weight, x1, x2 = self.generator.generateKinematics_batch(
            points, pdgs=self.initial_pdgs,
            pT_mincut=pT_mincut, delR_mincut=delR_mincut,
            rap_maxcut=rap_maxcut
        )
        return momenta, weight, x1, x2

    def get_ps_from_momenta(self, momenta, x1, x2, ensure_CM=True):
        ''' Momenta contains the two incoming particle 1 and 2 and the
        final state particles in the correct order'''
        ps, detjacinv = self.generator.getPSpoint_batch(momenta, x1, x2)
        return ps, detjacinv
