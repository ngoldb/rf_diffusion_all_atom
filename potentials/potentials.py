import torch
import rf2aa
import os 

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, seq, xyz):
        '''
            Given the current sequence and structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                seq (torch.tensor, size: [L,22]:    The current sequence of the sample.
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')


class ligand_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, weight=1, r_0=8, d_0=4):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()

        # Extract ligand Ca residues
        Ca_l = xyz[is_atom,1] # [Ll,3]

        # Extract binder Ca residues
        Ca_b = xyz[~is_atom,1] # [Lb,3]


        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_l[None,...].contiguous(), p=2) # [1,Ll,Lb]
        ligand_ncontacts = -1 * contact_energy(dgram, self.r_0, self.d_0)
        #Potential is the sum of values in the tensor
        ligand_ncontacts = ligand_ncontacts.sum()

        return self.weight * ligand_ncontacts


class monomer_contacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        Author: PV - adapted by NG from RFDiffusion to RFDiffusion all atom

        NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, seq, xyz):
        
        # Extract protein Ca
        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()
        Ca = xyz[~is_atom,1] # [Lb,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))

        return self.weight * ncontacts.sum()
    

class constraint_to_input(Potential):
    """
        Potential to constraint relative positions of a selection to the input structure

        Author: NG
    """

    def __init__(self, target_xyz, weight=1, residues="1-10"):
        self.weight = weight

        # convert to 0 indexing
        residues = [int(res) - 1 for res in residues.split('-')]

        # Extract protein Ca from input structure
        # NB: probably not the best idea, but I think it works
        Ca = target_xyz[: ,[1], :].squeeze() # [Lb,3]

        # cdist needs a batch dimension - NRB
        self.dgram_init = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        self.mask = torch.zeros(self.dgram_init.shape)
        self.mask[0][
            residues[0]:residues[1] + 1,
            residues[0]:residues[1] + 1
        ] = 1
        self.dgram_init = self.dgram_init * self.mask

    def compute(self, seq, xyz):

        # Extract protein Ca
        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()
        Ca = xyz[~is_atom,1] # [Lb,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        dgram = dgram * self.mask
        dgram_diff = dgram - self.dgram_init

        return self.weight * dgram_diff.abs().sum() * -1


class move_from_input(Potential):
    """
        Potential to move a region of the input away from its initial relative potential

        Author: NG
    """

    def __init__(self, target_xyz, weight=1, residues="1-10"):
        self.weight = weight

        # convert to 0 indexing
        residues = [int(res) - 1 for res in residues.split('-')]

        # Extract protein Ca from input structure
        # NB: probably not the best idea, but I think it works
        Ca = target_xyz[: ,[1], :].squeeze() # [Lb,3]

        # cdist needs a batch dimension - NRB
        self.dgram_init = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        self.mask = torch.ones(self.dgram_init.shape)
        self.mask[0][
            residues[0]:residues[1] + 1,
            residues[0]:residues[1] + 1
        ] = 0
        self.mask[0][
            residues[1] + 1:,
            residues[1] + 1:
        ] = 0
        self.dgram_init = self.dgram_init * self.mask

    def compute(self, seq, xyz):

        # Extract protein Ca
        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()
        Ca = xyz[~is_atom,1] # [Lb,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        dgram = dgram * self.mask
        dgram_diff = dgram - self.dgram_init

        return self.weight * dgram_diff.abs().sum()
    

class repulsion(Potential):
    '''
        Potential to minimze contacts between two sets of residues
        
        residues1 and residues2 are two sets of 1-indexed residues between which the potential will be applied

        Author: NG - adapted from PV's potentials
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, residues1="1-10", residues2="20-15"):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

        residues1 = [int(x) for x in residues1.split('-')]
        residues2 = [int(x) for x in residues2.split('-')]

        # change to 0-indexing
        self.residues1 = [i-1 for i in range(residues1[0], residues1[1] + 1)]
        self.residues2 = [i-1 for i in range(residues2[0], residues2[1] + 1)]
    
    def compute(self, seq, xyz):
        
        # Extract protein Ca
        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()
        Ca = xyz[~is_atom,1] # [Lb,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        # ncontacts.shape = [batch, prot_length, prot_length]
        ncontacts = (1 - numerator) / ((1 - denominator))

        # create a weight matrix (would be nicer to do in init to reduce computation)
        weight_m = torch.zeros(ncontacts.shape)
        print('ncontacts:', ncontacts.shape)
        for i in self.residues1:
            for j in self.residues2:
                # not sure what to do with first dimension
                weight_m[0][i,j] = -1
                weight_m[0][j,i] = -1

        repulsion_m = weight_m * ncontacts
        return self.weight * repulsion_m.sum()


def contact_energy(dgram, d_0, r_0):
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = torch.pow(divide_by_r_0,6)
    denominator = torch.pow(divide_by_r_0,12)
    
    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return - ncontacts


# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = {
    'ligand_ncontacts':     ligand_ncontacts,
    'monomer_contacts':     monomer_contacts,
    'repulsion':            repulsion,
    'constraint_to_input':  constraint_to_input,
    'move_from_input':      move_from_input
}

