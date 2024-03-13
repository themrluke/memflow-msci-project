from hist import Hist
import torch
import hist
import awkward as ak
import numpy as np
import os
import mplhep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl

prov = {
  "higgs": 1,
  "thad": 2,
  "thad_lightQ": 5,
  "tlep": 3
}

spanet_columns = {
  "higgs": 2,
  "thad": 0,
  "tlep": 1
}

# delete elements from tensor at specific indices
def tensor_delete(tensor, indices):
    mask = torch.ones(tensor[:,0].numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# This function takes as an inputs: the spanet_tensor, a spanet_column (e.g. Higgs: 2) and no_jets_required
# no_jets_required: 3 for thad, 2 for higgs, 1 for tlep
# the output = 1. the probability of spanet assignment for every event for one of the particles
# 2. the positions of the jets which were assigned by SPANET
def getPosition_spanet(spanet_values, spanet_column, no_jets_required):
    candidate = spanet_values[:,:,spanet_column] > 0.0
    positions = candidate.nonzero()
    positions = torch.reshape(positions, (spanet_values.shape[0], no_jets_required, 2))
    
    spanet_pos = positions[:,:,1]
    spanet_prob = torch.gather(spanet_values[:,:,spanet_column], dim=1, index=spanet_pos)
    
    return spanet_prob, spanet_pos


# SPANET returns the best jets combinations for thad and Higgs
# but sometimes SPANET assign the same jet for thad and Higgs => need a strategy assignment
# here we prioritize the Higgs assignment, than the tlep assignment and finally the thad
# WE DON'T consider if one jet has 90% prob to be assigned for thad and 40% to be assigned to Higgs
#       This jet will be assigned to Higgs
def higgsAssignment_SPANET(spanet_values):
    spanet_prob_thad, spanet_thad_pos = getPosition_spanet(spanet_values, spanet_column=spanet_columns['thad'], no_jets_required=3)
    spanet_prob_tlep, spanet_tlep_pos = getPosition_spanet(spanet_values, spanet_column=spanet_columns['tlep'], no_jets_required=1)
    spanet_prob_higgs, spanet_higgs_pos = getPosition_spanet(spanet_values, spanet_column=spanet_columns['higgs'], no_jets_required=2)
    
    # the unassigned jets will always have value -1 in the assignment tensor
    spanet_assignment_higgs = torch.ones((spanet_values.shape[0], spanet_values.shape[1]), device=spanet_values.device) * -1
    
    for jet in range(16):
        # check if the jet was assigned by SPANET to one of the particles
        jet_thad = spanet_thad_pos == jet
        jet_tlep = spanet_tlep_pos == jet
        jet_higgs = spanet_higgs_pos == jet

        # get events where the jet was assignned for one of the particle
        is_thad = torch.any(jet_thad, dim=1)
        is_tlep = torch.any(jet_tlep, dim=1)
        is_higgs = torch.any(jet_higgs, dim=1)

        # first priority: assign to Higgs, 2nd priority tlep, last: thad
        spanet_assignment_higgs[is_thad, jet] = 2
        spanet_assignment_higgs[is_tlep, jet] = 3
        spanet_assignment_higgs[is_higgs, jet] = 1
        
    # Using this strategy: Higgs always has 2 jets assigned, but this is not the case for thad/tlep
    # => Add padding in order to have always 3 "jets" assigned for thad
    # example: our dataset has max 16 jets. if I have an event with 2 thad assigned
    #          then the 3rd assigned jet will have the position 17
    #          by adding these padding positions -> we keep the same dimension for the tensor
    # the padding tensor has 6 columns: 2 higgs + 3 thad + 1 tlep
    padding_tensor = torch.ones((spanet_assignment_higgs.shape[0],6), device=spanet_values.device) * -1
    spanet_assignment_higgs = torch.cat((spanet_assignment_higgs, padding_tensor), dim=1)

    # THAD
    no_spanet_thad = torch.count_nonzero(spanet_assignment_higgs == 2, dim=1)
    
    # check events with 0 jets assigned for thad
    mask_thad_0 = no_spanet_thad == 0
    spanet_assignment_higgs[mask_thad_0,16:19] = 2

    # check events with 1 jet assigned for thad
    mask_thad_1 = no_spanet_thad == 1
    spanet_assignment_higgs[mask_thad_1,16:18] = 2
    
    # check events with 2 jets assigned for thad
    mask_thad_2 = no_spanet_thad == 2
    spanet_assignment_higgs[mask_thad_2,16:17] = 2

    
    # TLEP
    no_spanet_tlep = torch.count_nonzero(spanet_assignment_higgs == 3, dim=1)
    
    # check events with 0 jets assigned for tlep
    mask_tlep_0 = no_spanet_tlep == 0
    spanet_assignment_higgs[mask_tlep_0,19:20] = 3

    
    # HIGGS
    no_spanet_higgs = torch.count_nonzero(spanet_assignment_higgs == 1, dim=1)
    
    # check events with 0 jets assigned for higgs (none of them)
    mask_higgs_0 = no_spanet_higgs == 0
    spanet_assignment_higgs[mask_higgs_0,20:22] = 1

    # check events with 1 jet assigned for higgs (none of them)
    mask_higgs_1 = no_spanet_higgs == 1
    spanet_assignment_higgs[mask_higgs_1,20:21] = 1
    
    return spanet_assignment_higgs

    
# Inputs: spanet_assignment -> provenience tensor obtained by using the SPANET assignment
#         ev_indices -> choose a specific subset of indices
#         particle -> use prov['higgs']/prov['thad']/prov['tlep]
#         no_jets_required -> as before: 2 for higgs, 3 for thad etc
def get_JetPositions_forParticle(spanet_assignment, ev_indices, particle, no_jets_required):
    no_ev = torch.count_nonzero(ev_indices)
    
    jetsPositions = spanet_assignment[ev_indices] == particle
    # jetsPositions will have structure like this: [[event, positionJetIn_event]]
    jetsPositions = jetsPositions.nonzero()
    
    # reshape to have the following structure: [all_positions_FirstEvent, all_positions_SecondEvent ...]
    jetsPositions_reshaped = torch.reshape(jetsPositions, (no_ev, no_jets_required, 2))
    jetsPositions_reshaped = jetsPositions_reshaped[:,:,1]
    
    return jetsPositions_reshaped, jetsPositions

# Important: the higgs jets are ordered by pt
#            Same for thad jets (the b jet is NOT always the first one!!!)
def get_JetsPositions_ForEachParticle(spanet_assignment):
    # use all the events
    all_events = torch.ones(spanet_assignment.shape[0], dtype=torch.bool, device=spanet_assignment.device)
    
    # get jets positions for each of the particles
    # important: some tensors contain positions > 16 -> these are the ghosts positions added for padding
    #            not real positions
    positions_higgs, _ = get_JetPositions_forParticle(spanet_assignment, all_events, prov['higgs'], no_jets_required=2)
    positions_thad, _ = get_JetPositions_forParticle(spanet_assignment, all_events, prov['thad'], no_jets_required=3)
    positions_tlep, _ = get_JetPositions_forParticle(spanet_assignment, all_events, prov['tlep'], no_jets_required=1)

    return positions_higgs, positions_thad, positions_tlep


# By default order: higgs, thad, tlep
def sortJets_bySpanet(spanet_assignment, order=[0, 1, 2]):
    
    positions_higgs, positions_thad, positions_tlep = get_JetsPositions_ForEachParticle(spanet_assignment)
    
    list_positions = [positions_higgs, positions_thad, positions_tlep]
    
    jetsPositions_SortedbySpanet = torch.cat((list_positions[order[0]],
                                     list_positions[order[1]],
                                     list_positions[order[2]]), dim=1)
    
    return jetsPositions_SortedbySpanet

# all jets possible from events = maskJets
# jets already assigned: jetsPositions_SortedbySpanet
# jetsPositions_SortedbySpanet has in addition position > 16 (padding)
# jetsPositions_SortedbySpanet -> mask_jetsPositions_SortedbySpanet -> same tensor but written in maskJets style
# do a xor between maskJets and mask_jetsPositions_SortedbySpanet[:,:16] => find the unassigned jets
def find_unassignedJets(objects_sortedbySpanet, maskJets, jetsPositions_SortedbySpanet):
    
    # 22 values as spanet_assignment_higgs (check in higgsAssignment_SPANET function)
    mask_jetsPositions_SortedbySpanet = torch.zeros((jetsPositions_SortedbySpanet.shape[0], 22), device=objects_sortedbySpanet.device)
    mask_ones = torch.ones((jetsPositions_SortedbySpanet.shape[0], 22), device=objects_sortedbySpanet.device)

    # if I have the position tensors [[1, 2], [0,2]] -> build the mask tensor as [[0,1,1],[1,0,1]]
    mask_jetsPositions_SortedbySpanet = mask_jetsPositions_SortedbySpanet.scatter_(dim=1,
                                                            index=jetsPositions_SortedbySpanet,
                                                            src=mask_ones)
    
    # apply xor function to find the unattached jets jets
    # the result is a tensor which values 0 and 1 with 1 meaning not attached
    mask_jetsPositions_NotAttached = torch.logical_xor(mask_jetsPositions_SortedbySpanet[:,:16], maskJets)
    
    # get the positions of jets not attached 
    jetsPositions_NotAttached = mask_jetsPositions_NotAttached.nonzero()
    
    return jetsPositions_NotAttached, mask_jetsPositions_NotAttached

# By default order: higgs, thad, tlep
def sortObjects_bySpanet(spanet_assignment, scaledLogReco, maskJets, order=[0, 1, 2]):
    
    # 21 objects because before we had 18 objects before
    # but due to our new way of sorting, there could be some missing jets due to the spanet overlapping
    objects_sortedbySpanet = torch.ones((spanet_assignment.shape[0], 21, 8), dtype=scaledLogReco.dtype, device=spanet_assignment.device) * -100
    
    jetsPositions_SortedbySpanet = sortJets_bySpanet(spanet_assignment, order=[0, 1, 2])
    
    no_jets = 6 # example: H1, H2, thad1, thad2, thad3, tlep1
    for i in range(no_jets):
        good_events = jetsPositions_SortedbySpanet[:,i] < 16 # if position >= 16 => this is a padding (not real position)
        objects_sortedbySpanet[good_events, i] = scaledLogReco[good_events, jetsPositions_SortedbySpanet[good_events,i]]

    # now attach the lepton and MET
    objects_sortedbySpanet[:,6] = scaledLogReco[:,16]
    objects_sortedbySpanet[:,7] = scaledLogReco[:,17]
    
    jetsPositions_NotAttached, mask_jetsPositions_NotAttached = find_unassignedJets(objects_sortedbySpanet, maskJets, jetsPositions_SortedbySpanet)
    max_NoUnassignedJets = torch.max(torch.count_nonzero(mask_jetsPositions_NotAttached, dim=1))
    
    # strategy: 1. attach one jet for each event with unassigned jets
    # 2. remove the attached jets -> do these 2 steps until there are no events with unassigned jets
    for i in range(max_NoUnassignedJets):

        # find the events which still have unassigned jets
        # here the result will be [0,2,4,18...] for i=0
        unassignedEvents, counts = torch.unique(jetsPositions_NotAttached[:,0], return_counts=True)

        # I must compute the 'indexOf_firstAppearanceEvent' which represent the first index of each event from 'jetsPositions_NotAttached'
        # in my case: event 0 has 2 unassigned jets, event 2 has 2 unassigned jets, event 4 has 5 unassigned jets
        # => indexOf_firstAppearanceEvent = [0,2,4,9,...]
        indexOf_firstAppearanceEvent = torch.cumsum(counts, dim=0)
        firstElem = torch.zeros(1, dtype=torch.int64, device=spanet_assignment.device)
        indexOf_firstAppearanceEvent = torch.cat((firstElem, indexOf_firstAppearanceEvent[:-1]), dim=0) 

        # for the unassigned_events -> attach the jets from jetsPositions_NotAttached[indexOf_firstAppearanceEvent,1]
        objects_sortedbySpanet[unassignedEvents,8+i] = scaledLogReco[unassignedEvents, jetsPositions_NotAttached[indexOf_firstAppearanceEvent,1]]

        # remove the assigned jets at this step -> keep only the unassigned jets
        # repeat until the tensor is empty
        jetsPositions_NotAttached = tensor_delete(jetsPositions_NotAttached, indexOf_firstAppearanceEvent)
    
    
    return objects_sortedbySpanet

# build padding Tensor for the proveniance
# I do this because sometimes the events are not fully matched
# so: when I look for the positions of Higgs: if there is only one matched Higgs -> I add another prov=Higgs at position 17
# position 17 doesn't exist in data -> it's just a padding to keep the dimension fixed for pytorch
# Input: prov_assignment -> tensor with matched assignments: [Ev, prov]
#        padding_tensor -> tensor which is updated
#        max_matched_jets -> no_jets for the corresponding particle (e.g. higgs=2, tlep=1 etc.)
#        particle -> we read the prov value of the jet from the 'prov' dictionary (see above)
#        first_elem -> index of the element where the assignment starts
def build_paddingTensor(prov_assignment, padding_tensor, max_matched_jets=2, first_elem=0, particle='higgs'):
    particle_pos = prov_assignment == prov[particle]
    no_jet_matched = torch.count_nonzero(particle_pos, dim=1)

    # for over all posible cases: events with 0/1/2/... matched jets
    for NoMatched_jets in range(max_matched_jets):
        mask = no_jet_matched == NoMatched_jets
        
        padding_tensor[mask,first_elem:first_elem+max_matched_jets-NoMatched_jets] = prov[particle]

    return padding_tensor

# Important: the higgs jets are ordered by pt
#            Same for the other jets
# IMPORTANT 2: Here the first jet from the thad decay is the b-jet
def get_JetsPositions_ForEachParticle_prov(prov_tensor):
    # use all the events
    all_events = torch.ones(prov_tensor.shape[0], dtype=torch.bool, device=prov_tensor.device)
    
    # get jets positions for each of the particles
    # important: some tensors contain positions > 16 -> these are the ghosts positions added for padding
    #            not real positions
    positions_higgs, _ = get_JetPositions_forParticle(prov_tensor, all_events, prov['higgs'], no_jets_required=2)
    positions_thad_lightQ, _ = get_JetPositions_forParticle(prov_tensor, all_events, prov['thad_lightQ'], no_jets_required=2)
    positions_thad, _ = get_JetPositions_forParticle(prov_tensor, all_events, prov['thad'], no_jets_required=1)
    positions_tlep, _ = get_JetPositions_forParticle(prov_tensor, all_events, prov['tlep'], no_jets_required=1)

    return positions_higgs, positions_thad, positions_thad_lightQ, positions_tlep


# By default order: higgs, thad_b, thad_lightQ, tlep
def sortJets_byProv(prov_tensor, order=[0, 1, 2, 3]):
    
    positions_higgs, positions_thad, positions_thad_lightQ, positions_tlep = get_JetsPositions_ForEachParticle_prov(prov_tensor)
    
    list_positions = [positions_higgs, positions_thad, positions_thad_lightQ, positions_tlep]
    
    jetsPositions_SortedbyProv = torch.cat((list_positions[order[0]],
                                     list_positions[order[1]],
                                     list_positions[order[2]],
                                     list_positions[order[3]]), dim=1)
    
    return jetsPositions_SortedbyProv

def find_unassignedJets_prov(objects_sortedbySpanet, maskJets, jetsPositions_SortedbySpanet):
    
    # 24 values because 16 jets + 2 lepton/MET + 6 padding
    mask_jetsPositions_SortedbySpanet = torch.zeros((jetsPositions_SortedbySpanet.shape[0], 24), device=objects_sortedbySpanet.device)
    mask_ones = torch.ones((jetsPositions_SortedbySpanet.shape[0], 24), device=objects_sortedbySpanet.device)

    # if I have the position tensors [[1, 2], [0,2]] -> build the mask tensor as [[0,1,1],[1,0,1]]
    mask_jetsPositions_SortedbySpanet = mask_jetsPositions_SortedbySpanet.scatter_(dim=1,
                                                            index=jetsPositions_SortedbySpanet,
                                                            src=mask_ones)
    
    # apply xor function to find the unattached jets jets
    # the result is a tensor which values 0 and 1 with 1 meaning not attached
    mask_jetsPositions_NotAttached = torch.logical_xor(mask_jetsPositions_SortedbySpanet[:,:16], maskJets)
    
    # get the positions of jets not attached 
    jetsPositions_NotAttached = mask_jetsPositions_NotAttached.nonzero()
    
    return jetsPositions_NotAttached, mask_jetsPositions_NotAttached

# By default order: higgs, thad, tlep
def sortObjects_byProv(scaledLogReco, maskJets, order=[0, 1, 2]):
    
    # 21 objects because before we had 18 objects before
    # but due to our new way of sorting, there could be some missing jets due to the spanet overlapping
    objects_sortedbyProv = torch.ones((scaledLogReco.shape[0], 24, 8), dtype=scaledLogReco.dtype, device=scaledLogReco.device) * -100

    prov_assignment = scaledLogReco[:,:,-1]

    padding_tensor = torch.ones((scaledLogReco.shape[0],6), device=scaledLogReco.device) * -1

    padding_tensor = build_paddingTensor(prov_assignment=prov_assignment, padding_tensor=padding_tensor,
                                         max_matched_jets=1, first_elem=0, particle='thad')
    padding_tensor = build_paddingTensor(prov_assignment=prov_assignment, padding_tensor=padding_tensor,
                                         max_matched_jets=2, first_elem=1, particle='thad_lightQ')
    padding_tensor = build_paddingTensor(prov_assignment=prov_assignment, padding_tensor=padding_tensor,
                                         max_matched_jets=1, first_elem=3, particle='tlep')
    padding_tensor = build_paddingTensor(prov_assignment=prov_assignment, padding_tensor=padding_tensor,
                                         max_matched_jets=2, first_elem=4, particle='higgs')

    # now this prov_withPadding contains events with 2 higgs assigned, 1 tlep assigned etc (fully matched)
    # the positions > 16 represent the padding
    prov_withPadding = torch.cat((prov_assignment, padding_tensor), dim=1)

    # order: higgs, thad_b, thad_lightQ, tlep
    jetsPositions_SortedbyProv = sortJets_byProv(prov_withPadding, order=[0, 1, 2, 3])
    
    no_jets = 6 # example: H1, H2, thad1, thad2, thad3, tlep1
    for i in range(no_jets):
        good_events = jetsPositions_SortedbyProv[:,i] < 16 # if position >= 16 => this is a padding (not real position)
        objects_sortedbyProv[good_events, i] = scaledLogReco[good_events, jetsPositions_SortedbyProv[good_events,i]]

    # now attach the lepton and MET
    objects_sortedbyProv[:,6] = scaledLogReco[:,16]
    objects_sortedbyProv[:,7] = scaledLogReco[:,17]

    # TODO from here
    
    jetsPositions_NotAttached, mask_jetsPositions_NotAttached = find_unassignedJets_prov(objects_sortedbyProv, maskJets, jetsPositions_SortedbyProv)
    max_NoUnassignedJets = torch.max(torch.count_nonzero(mask_jetsPositions_NotAttached, dim=1))
    
    # strategy: 1. attach one jet for each event with unassigned jets
    # 2. remove the attached jets -> do these 2 steps until there are no events with unassigned jets
    for i in range(max_NoUnassignedJets):

        # find the events which still have unassigned jets
        # here the result will be [0,2,4,18...] for i=0
        unassignedEvents, counts = torch.unique(jetsPositions_NotAttached[:,0], return_counts=True)

        # I must compute the 'indexOf_firstAppearanceEvent' which represent the first index of each event from 'jetsPositions_NotAttached'
        # in my case: event 0 has 2 unassigned jets, event 2 has 2 unassigned jets, event 4 has 5 unassigned jets
        # => indexOf_firstAppearanceEvent = [0,2,4,9,...]
        indexOf_firstAppearanceEvent = torch.cumsum(counts, dim=0)
        firstElem = torch.zeros(1, dtype=torch.int64, device=scaledLogReco.device)
        indexOf_firstAppearanceEvent = torch.cat((firstElem, indexOf_firstAppearanceEvent[:-1]), dim=0) 

        # for the unassigned_events -> attach the jets from jetsPositions_NotAttached[indexOf_firstAppearanceEvent,1]
        objects_sortedbyProv[unassignedEvents,8+i] = scaledLogReco[unassignedEvents, jetsPositions_NotAttached[indexOf_firstAppearanceEvent,1]]

        # remove the assigned jets at this step -> keep only the unassigned jets
        # repeat until the tensor is empty
        jetsPositions_NotAttached = tensor_delete(jetsPositions_NotAttached, indexOf_firstAppearanceEvent)
    
    
    return objects_sortedbyProv
