# CS589 final project

k-NN, random forest and neural network (from our homeworks) on Digits, Parkinsons, Rice and Credit Approval,
10-fold stratified CV. Extra credit: CMC dataset (EC2), ensemble (EC3), extra trees (EC4).

- `code/` - all code. `python run_experiments.py` runs everything (or e.g. `python run_experiments.py rice credit`)
- `report/` - FinalProject.tex / FinalProject.pdf, figures and results json

The course csv files (parkinsons, rice, credit_approval) are not in the repo, get them from Canvas and put them in `code/data/`.
Digits loads through scikit-learn. `pip install -r code/requirements.txt`
