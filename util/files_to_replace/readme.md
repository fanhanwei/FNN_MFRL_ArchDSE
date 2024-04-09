>[!NOTE] We directly modify the sklearn and botorch files to implement the baseline algorithms. We provide the modified files in this directory, please replace the corresponding files.

replace .conda-env/lib/python3.10/site-packages/botorch/models/gp_regression.py with ./gp_regression.py

replace .conda-env/lib/python3.10/site-packages/sklearn/ensemble/__init__.py with ./__init__.py
replace .conda-env/lib/python3.10/site-packages/sklearn/ensemble/_bagging.py with ./_bagging.py
replace .conda-env/lib/python3.10/site-packages/sklearn/ensemble/_weight_boosting.py with ./_weight_boosting.py

replace .conda-env/lib/python3.10/site-packages/skopt/optimizer/optimizer.py with ./optimizer.py
replace .conda-env/lib/python3.10/site-packages/skopt/learning/forest.py with ./forest.py