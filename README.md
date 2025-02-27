Code for RFDiffusion AA
--------------------
<p align="center">
  <img src="./img/RFDiffusionAA.png" alt="alt text" width="600px"/>
</p>

## Setup/Installation
1. Clone the package
```
git clone https://github.com/baker-laboratory/rf_diffusion_all_atom.git
cd rf_diffusion_all_atom
```
2. Download the container used to run RFAA.
```
wget http://files.ipd.uw.edu/pub/RF-All-Atom/containers/rf_se3_diffusion.sif
```
3. Download the model weights.
```
wget http://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFDiffusionAA_paper_weights.pt
```

4. Initialize git submodules
```
git submodule init
git submodule update
```
5. Install Apptainer

Install apptainer if you do not already have it on your system.  This will allow you to run our code without installing any python packages using a prepackaged sif: https://apptainer.org/docs/admin/main/installation.html


## Inference
### Small molecule binder design

To generate a binder to the ligand OQO from PDB 7v11, run the following:


Example (ligand binder):
```
/usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.deterministic=True diffuser.T=100 inference.output_prefix=output/ligand_only/sample inference.input_pdb=input/7v11.pdb contigmap.contigs=[\'150-150\'] inference.ligand=OQO inference.num_designs=1 inference.design_startnum=0
```

Note: The --nv flag must be omitted if not using a GPU.

Explanation of arguments:
- `inference.deterministic=True` seeds the random number generators used so that results are reproducible.  i.e. running with inference.design_startnum=X will produce the same reusults.  Note that torch does not guarantee reproducibility across CPU/GPU architectures: https://pytorch.org/docs/stable/notes/randomness.html
- `inference.num_designs=1` specifies that 1 design will be generated
- `contigmap.contigs=[\'150-150\']` specifies that the length of the generated protein should be 150
- `diffuser.T=100` specifies the number of denoising steps taken.

Expected outputs:
- `output/ligand_only/sample_0.pdb` The design PDB
- `output/ligand_only/sample_0_Xt-1_traj.pdb` The partially denoised intermediate structures
- `output/ligand_only/sample_0_X0-1_traj.pdb` The predictions of the ground truth made by the network at each step

Note that the sequences associated with these structure have no meaning, apart from the given motif.  LigandMPNN or similar must be used to generate sequences for the backbones if they are to be used for structure prediction / expression.

To include protein residues A430-435 in the motif, use the argument contigmap.contigs.  e.g. `contigmap.contigs=[\'10-120,A84-87,10-120\']` tells the model to design a protein containing the 4 residue motif A84-87 with 10-120 residues on either side.

### Small molecule binder design with protein motif
Example (ligand binder with protein motif):
```
/usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.deterministic=True diffuser.T=200 inference.output_prefix=output/ligand_protein_motif/sample inference.input_pdb=input/1haz.pdb contigmap.contigs=[\'10-120,A84-87,10-120\'] contigmap.length="150-150" inference.ligand=CYC inference.num_designs=1 inference.design_startnum=0
```

An end-to-end design pipeline illustrating the design of heme-binding proteins using RFdiffusionAA, proteinMPNN, AlphaFold2, LigandMPNN and PyRosetta is available at: https://github.com/ikalvet/heme_binder_diffusion

#### Note
To run with access to scratch or work you need to run:
```
apptainer run --bind /scratch:/scratch --bind /work:/work --nv rf_se3_diffusion.sif -u run_inference.py ...
```

### Custom Potentials
I added some potentials:
- monomer contacts (basically copied from RFDiffusion)
- repulsion
- constraint_to_input
- move_from_input

#### Repulsion potential
This potentials minimizes the number of contacts between two selctions. Selections are called `residues1` and `residues2` and need to be a range of residues. Note that these assume to be 1-indexed (PDB numbering starting at 1 - even if your PDB is numbered differently). Example usage:
```
... potentials.guiding_potentials=[\'type:repulsion,weight:5,residues1:1-26,residues2:114-148\'] potentials.guide_scale=1 potentials.guide_decay="quadratic" diffuser.partial_T=80
```

#### Constraint to input potential
This potential only works with partial diffusion. It will take the input structure and constraint the relative positions of a given region to the relative input positions. Example usage:
```
... potentials.guiding_potentials=[\'type:constraint_to_input,weight:1,residues:1-23\'] potentials.guide_scale=1 potentials.guide_decay="quadratic" diffuser.partial_T=80
```

If you want to break up the relative positions of the region, simply use a negative weight. 

#### Move from input potential
This potential only works with partial diffusion. This potential does kind of the opposite of the `constraint_to_input`potential: It maximizes a difference in relative positions of the specified region to the rest of the structure - basically moving the region away from where it started. Example usage:
```
... potentials.guiding_potentials=[\'type:move_from_input,weight:0.03,residues:1-23\'] potentials.guide_scale=1 potentials.guide_decay="quadratic" diffuser.partial_T=80
```
The difference of this to the `repulsion` potential is that the repulsion potential minimizes the number of contacts between two selections while the `move_from_input` potential simply tries to maximizes the difference in the distogram from current to input between the selection and the rest of the structure. Not sure if this is helpful for anything.