# DeepFMPO
Adopting the technique used in the paper: [Deep Reinforcement Learning for Multiparameter Optimization in
de novo Drug Design](https://doi.org/10.26434/chemrxiv.7990910.v1) for optimising the pIC50 value of the molecules

Outputs of some of the experiments are in the folder "past outputs"
## Instructions

To run the main program on the same data as used in the best outputs (in folder "past outputs/7July/clean_good_manual/"):
```sh
python Main.py
```
It is also possible to run the program on a custom set of lead molecules and/or fragments. 
```sh
python Main.py fragment_molecules.smi lead_file.smi
```
Molecules that are generated during the process can be viewed by running:
```sh
python viewing_outputs.py -epoch epoch
```
where `epoch` is the epoch that should be viewed. 

New molecules can also be generated from a saved generation model. For this run:
```sh
python viewing_outputs.py -gen 1
```
Please note that I have NOT SAVED the best generation model. And also the this output IS NOT STOCHASTIC. 

In either of the ways, the output is as follows:
1. Displays two columns of molecules as PNG file. The first column contains the original lead molecule, while the second column contains modified molecules.
2. Displays a histogram containing the pIC50 distributions in the lead molecules and the final output.
3. Saves two csv files- one containing a table of all the changed molecules and one containing a table of all the molecules which have been made from inactive to active. These files are saved in the folder _past outputs_

Any global parameters can be changed by changing them in the file "Modules/global_parameters.py"

## Short description of all code files
1. Main.py: The main file. This has to be run for training.
2. viewing_outputs.py: File to view outputs as described above.
3. Show_Epoch.py: Reads and decodes generated molecules, used by viewing_outputs.py
4. FMPO-Visualising the outputs.ipynb: Jupyter notebook used for testing parts of the code, as well as viewing outputs
5. Files inside "Modules":
	1. build_encoding.py: Contains functions involved in building and saving encodings
	2. file_reader.py: Contains functions involved in reading .smi and .csv input files
	3. global_parameters.py: All global parameters can be set here
	4. models.py: Architecture of Actor and Critic are present
	5. mol_utils.py: Utility functions for handling molecules(like breaking fragments)
	6. rewards.py: The predictive model is deployed here. Contains all funcions pertaining to generating the rewards.
	7. similarity.py: Contains functions that can be used to calculate similarity coefficients- Tanimoto and Levenshtein/Edit Distance
	8. training.py: Calculates the initial distribution and trains the actor and critic networks.
	9. tree.py: Implements the tree class along with "btl": Build tree from list function

##Short Description of non-code files:
1. Padel.txt: Contains the outputs of Padel file
2. descriptors.csv: used to store the initial descriptors in this file
3. uneval_desc.csv: in case, descriptors.csv contains NaN values, such rows are re-evaluated in uneval_desc.csv


## Requirements

The following Python libraries are required to run it:
- rdkit
- numpy
- sklearn
- keras
- pandas
- bisect
- Levenshtein
- A backend to keras, such as theano, tensorflow or CNTK
- xgboost
