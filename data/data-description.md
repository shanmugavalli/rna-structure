# Dataset Description

In this competition you will predict five 3D structures for each RNA sequence.

## Files

### `[train/validation/test]_sequences.csv`

The target sequences of the RNA molecules.

**Columns:**

- `target_id` _(string)_ - An arbitrary identifier. In `train_sequences.csv`, this is the ID of the entry in the Protein Data Bank.
- `sequence` _(string)_ - The RNA sequence of all chains in the target, concatenated together according to stoichiometry.
- `temporal_cutoff` _(string)_ - The date in `yyyy-mm-dd` format that the sequence was or will be published.
- `description` _(string)_ - Details of the origins of the sequence. For PDB entries, this is the entry title.
- `stoichiometry` _(string)_ - The chains used for the target. These take the form of `{chain:number}`, where chain corresponds to the author-defined chain in `all_sequences`, joined with a semicolon delimiter (`;`).
- `all_sequences` _(string)_ - FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. May include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules, just the ones specified in `stoichiometry` which are concatenated in `sequence`. Can be parsed into a dictionary with `extra/parse_fasta_py.py`.
- `ligand_ids` _(string)_ - Three-letter names in PDB chemical component dictionary of any small molecule ligands solved in the experimental structure, joined with a semicolon delimiter (`;`). You don't need to make predictions for these molecules.
- `ligand_SMILES` _(string)_ - SMILES strings giving chemical structures of any small molecule ligands solved in the experimental structure, joined with a semicolon delimiter (`;`).

### `[train/validation]_labels.csv`

Experimental structures.

**Columns:**

- `ID` _(string)_ - Identifies the `target_id` and residue number, separated by `_`. **Note:** residue numbers use one-based indexing.
- `resname` _(character)_ - The RNA nucleotide (`A`, `C`, `G`, or `U`) for the residue.
- `resid` _(integer)_ - Residue number.
- `x_1, y_1, z_1, x_2, y_2, z_2, …` _(float)_ - Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and `train_labels.csv` curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions.
- `chain` _(string)_ - Residue's chain ID. For the target there is one chain assigned to each unique sequence, potentially derived from author-assigned chain in PDB entry. **Note:** Multiple chains of the molecule can share the same chain if they have the same sequence.
- `copy` _(integer)_ - Which chain copy (1, 2, …) the residue is in. Greater than 1 if there are multiple copies of the same sequence in the structure.

### `sample_submission.csv`

**Format:** Same format as `train_labels.csv` but with five sets of coordinates for each of your five predicted structures: `x_1, y_1, z_1, x_2, y_2, z_2, … x_5, y_5, z_5`.

**Requirements:**

- You must submit five sets of coordinates.
- Note that `x, y, z` are clipped between `-999.999` and `9999.999` before scoring, due to use of a legacy PDB format that has maximal 8 characters for coordinates.
- `chain` and `copy` do not have to be provided.

### `MSA/`

Contains multiple sequence alignments in FASTA format for each target in `train_sequences.csv` and `validation_sequences.csv`.

**Details:**

- Files are named `{target_id}.MSA.fasta`.
- During evaluation with hidden test sequences, your notebook will have access to these MSA files for the hidden `test_sequences.csv`.
- For multi-chain targets, each homolog found for a given chain sequence is presented in a separate row with placeholders for other chain sequences provided as gaps (`-`).
- The header for each homolog encodes the source of the sequence. A tag `chain={chain}` is appended. If multiple copies of the chain are present in the target, a tag `copies={copy}` is also appended. Tags are separated by `|` delimiter.

### `PDB_RNA/`

Contains 3D structural information available in the Protein Data Bank:

- `{PDB_id}.cif` - Files for each RNA-containing entry.
- `pdb_seqres_NA.fasta` - Sequences of all nucleic acid chains in the PDB in FASTA format.
- `pdb_release_dates_NA.csv` - Entry ID and release dates of the RNA-containing PDB entries in CSV format.

### `extra/`

Additional helper files:

- `parse_fasta_py.py` - Helper script with function `parse_fasta()`, which can take the `all_sequences` field of `{train/test/validation}_sequences.csv` and produce a dictionary of `chain:sequence`.
- `rna_metadata.csv` - Data extracted from all RNA and RNA/DNA hybrid structures up to December 17, 2025.
- `README.md` - Description of the data in `rna_metadata.csv`.

## Additional Notes

### Train/Validation Split

The datasets were partitioned using a cluster-based temporal split to minimize homology between training and evaluation data:

- All chain sequences were clustered using **MMseqs2** at a **30% identity level**, and each cluster was assigned a `temporal_cutoff` based on its oldest member.
- Any cluster containing at least one entry released **before May 29, 2025**, was assigned to the **training set**.
- Only clusters where all members were released **after May 29, 2025** (the final submission date of the last Stanford RNA 3D Folding competition and up to December 17, 2025) were included in the **validation set**.
- This ensures that no structure in the validation set shares more than 30% sequence identity with any structure released prior to the training cutoff.

### Validation Set Filtering

The sequences in `validation_sequences.csv` (which is the same as `test_sequences.csv` publicly provided here) were further filtered to have:

- Composition of at least **40% RNA**
- Unique sequences (up to **sequence identity 90%**)

### Training Set Notes

- `train_sequences.csv` has **not been filtered for sequence redundancy** but contains only PDB entries that pass selection criteria.
- `train_sequences.csv` should have **no overlap** with `validation_sequences.csv`.
- Note that `train_sequences.csv` has been filtered, so it does not have all RNA targets in the PDB to date or all the CIF files in `PDB_RNA/`.

### Filtering Criteria

The additional file `extra/rna_metadata.csv` contains data extracted from all RNA and RNA/DNA hybrid structures up to December 17, 2025. The metadata description is included in `extra/README.md`.

These metadata were used to filter the structures that were included in the `{train, test, validation}_sequences.csv` using the following criteria:

- Canonical `ACGU` residues or modified residues that can be mapped to canonical using either PDB chemical component dictionary or NAKB mapping.
- No undefined (`N`) residues and no `T` (for hybrid NA).
- No more than **25%** of residues that were modified / non-canonical.
- At least **50%** residues reported in the sequence were modeled/observed.
- Total adjusted structuredness (see `extra/README.md`) of all RNA chains is at least **20%**.

If any RNA chain in the PDB file didn't meet those criteria, the whole entry has been removed.

Additionally, the remaining entries were filtered to have:

- At least **10 nt** in all chains combined.
- Entries without resolved C1' atoms (for example P traces) were rejected.

Targets in `{test, validation}_sequences.csv` were further filtered to contain only entries with at least **40% RNA composition** and the redundancy was removed by selecting single entry based on **MMseqs2 clustering at 90% identity level**.

### Processing Pipeline

The processing pipeline is available here: https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline