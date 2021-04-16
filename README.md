# MMD-Assignment #1: Local Sensitive Hashing (LSH) for Item Search and Genre Classification for FMA dataset.
##Team: Emin Guliev, Justus Rass & Christian Wiskott.

###Requirements
- Python 3
- Numpy
- Matplotlib
- Jupyter

###Cloning

Github repo can be cloned from:
```bash
https://github.com/ezorrio/genre-classification.git
```
If you have data files fetched from original source, clone using following command:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ezorrio/genre-classification
```

###Description
The 4 Python files should be placed in a directory with the .csv files in a folder called 'metadata'.

- FMA.py
- LSH.py
- main.py
- MusicSearch.py
- metadata/
    - features.csv
    - tracks.csv
- experiments/
- paper/ - contains LaTeX source files for paper

To run it with the hyperparameters used for the final results, simply execute the 'experiments_manual.ipynb' file
located in the experiments folder.

The experiments folder contains the ipython notebooks used to carry out the tests for validation.
The experiments/results folder contains the raw results for the test suite used for k=3, 5, 7 nearest
neighbours, as well as a condensed list of the best achieved results and the manual tests carried out
with those parameters.