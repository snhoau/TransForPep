# TransForPep

This program uses a trained Transformer model[arXiv:1706.03762] to generate atomic coordinates, types, and charges for protein interaction surfaces on any given protein surface region. The training data for this model is extracted from the PDB database. Based on this data, the program can use the Point Drift algorithm alignment [DOI:10.1109/TPAMI.2010.46] method to search for peptides in the database (which should have a stable secondary structure and calculated atomic charges, and can be stable peptides or cyclic peptides) to obtain backbone proteins. Finally, software like Rosetta is used for side-chain optimization of the interaction site to obtain interacting peptides at specified protein residues.

The training dataset can be constructed by the user or can be obtained from a pre-built database (https://u.pcloud.link/publink/show?code=XZFQeaVZlRnVZny3nygSc7h4j5llMY1NW6SUFoVXy).

1_transformer_redo_ignh_dis350_cbem.py: Program for training the model.

2_transformer_evel_dis350_cbem.py: Program that calls the trained model for outputting the interaction interfaces.

3_ICPdatabase_cbem.py: Program that uses the alignment method to search the interaction interfaces in the database.

4_score.py: Program that calls the results of the alignment and scores the peptides in the database.

supplymentary/cbem.pkl: Contains a combination dictionary of atomic types and charges.

supplymentary/transpep_model_dis350_cbem_final.pt: Trained model file.

# Usage

# 1. Training
If you are planning to use my pre-trained model, you can skip this step. However, you can also modify the model parameters and train a new model using the real-world protein-protein interaction interface data (from PDB). We provide a dataset for calculating charges using a force field on the interaction interface. See above for the download method.

Operating Environment
We provide the training environment for reference. It should not give any errors. Adjust the batch size according to the available GPU memory. Install the missing components using pip. CUDA and PyTorch versions mentioned below are required.

CUDA 11.8
PyTorch 2.0.1
Pickle
GPUtil
Altair
Pandas
NumPy

Areas to be modified
1) Modify the map.txt file in the dataset to your own path, or use a relative address.
2) Various modifiable sections in the program (1):
   - line 599: location of the training database file
   - line 660: training parameters
   - line 561: dimensions
   - line 563: layers
   - line 307: other model parameters

# 2. Generating an Ideal Interaction Interface
Generate an interaction surface directly for the target protein's target region.

Prepare the atomic information for the target protein's region, including coordinates, charges, and atom types. You can refer to our provided file gsdmdinput.pkl for formatting. This file is saved using pickle, and you can view it by using the pickle.load() command as follows:

```python
with open('gsdmdinput.pkl', 'rb') as f:
    datainfo = pickle.load(f)
print(datainfo)
```

lines to be modified
- line 406: path to save the results of the ideal interaction interface
- line 411: path to save the ideal interaction interface as an XYZ file

Note: Delete the atomic information in the XYZ file that exceeds the predicted region of the model (where the coordinate points are mostly zero). Also, make sure to modify the number of atoms at the top. You can use software (such as DS View) to view the file.

# 3. Database Search
We use the publicly available scorffed database from David Baker's lab, which contains short peptides, along with our own custom peptide database. Due to originality concerns, we cannot provide the download link for the full database (we only provide a small amount of data as an example, link: https://u.pcloud.link/publink/show?code=XZIKoaVZFsXBdLSXO2Vkw5sR1BNhEFxNlmt7). Please download the complete database from David Baker's lab website and cite it appropriately. You can also use your own custom peptide database (we recommend using structurally stable peptides for building the database). After calculating the charges for each PDB file in the database (using Amber), save the data in the format specified in the input file (gsdmdinput.pkl) or the database file (pickle). If you have any questions, feel free to contact us.

lines to be modified
- line 42: path to the generated ideal interaction interface from the previous step
- line 48: list of paths to the files in the database to be searched (similar to the structure of the training database)
- lines 103, 106, 109: collection of output score files for comparing atomic distances, types, and charges

# 4. Scoring the Search Results
Score and analyze the comparison results from the search.

lines to be modified
- lines 45, 47, 49: paths to the intermediate files outputted in the previous step
- line 53: list of paths to the files in the database to be searched
- line 91: path to the output result file (higher scores indicate better results)

This program only provides the basic framework for interacting peptides. For better results, we recommend using interaction optimization software such as Rosetta to further optimize the amino acids based on the interaction conformation.


