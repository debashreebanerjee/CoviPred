# CoviPred

Code repository collecting all the python code related to [this paper on prediction of COVID-19 via hematological data across populations](https://doi.org/10.1101/2023.03.07.23286949) (under review), developed at the BioInformatics Lab (P.I.- Dr. Debashree Bandyopadhyay) in the Department of Biological Sciences at BITS-Pilani, Hyderabad Campus.

Code dependencies are: SciPy, NumPy, Pandas, XGBoost(0.90), SciKit-Learn, Seaborn, MatPlotLib


# Files in this Repo

* Zipped Directories:
    * rawdata.zip - Raw input datasets.
    * datasets.zip - All processed datasets used for model training.

* Python Files:
    * datasetCreation.py - Generates the processed datasets from raw datasets (not attached in this repo).
    * stat.py - Computes correlation via the point-biserial test, & dataset similarity via SciPy's Kolmogorov-Smirnow test.
    * xgboost.py - Fits Extreme Gradient Boosting classifiers & computes all performance metrics for internal validation for dataset.
    * otheralgos.py - Fits seven classifiers, computes some performance metrics for internal validation, & plots Receiver Operator Curves, for each dataset.
    * extval.py - For select datasets, this code computes all performance metrics for external validation of Extreme Gradient Boosting classifiers at different classification thresholds.
