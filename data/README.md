# The 168

Connectomes for 42 CN/EMCI/LMCI/AD participants in the ADNI study.

Generated circa 2016 by Neil Oxtoby with Sara Garbarino using Neil's [pipeline](https://github.com/noxtoby/NetMON/tree/master/connectome_pipeline) based on MRtrix's Anatomically Constrained Tractography pipeline (which was new at the time).

Neil Oxtoby, UCL POND and CMIC, University College London, 2024.

## Files

- `ADNI_SC_The168.xlsx`: list of individuals, plus corresponding baseline rows from `ADNIMERGE`
- `connectomes_freesurfer`: folder containing the raw connectomes. `...assignment_all_voxels` indicates use of the corresponding option in mrtrix.
- `fs_default.txt`: connectome lookup table (used in `labelconvert` step)
