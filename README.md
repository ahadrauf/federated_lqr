# Personalized Federated Learning for Inverse LQR
Written as part of a PhD research rotation in Winter 2021 as a 
part of the Autonomous Systems Lab under Professor Marco Pavone

Paper Link: https://drive.google.com/file/d/1GvfAKxxFkInUe_8MVJKWhXGdy0RAgIo9/view?usp=sharing

## Repo Structure
1. `fedadmm_*.py`: Code to run a personalized federated learning
2. `ADMM_*.py`: Helper repos that run various versions of ADMM
3. `utils.py`: Helper scripts (mostly `latexify()` for nice figure generation)
4. `tests\` repo
   * `lqr_*.py`: Test code to see how an LQR controller behaves given different configurations. Replaced by the function `simulate()` in `fedadmm_*.py`
   * `test_*.py`: Various test scripts I wrote over the course of the quarter to test different Python modules or configurations
   * `_*.m`: More test scripts, except in MATLAB
5. `generate_figures` repo: The scripts used to generate the paper's figures
6. `data` repo: Where all data from `fedadmm_*.py` scripts are stored
7. `figures` repo: Where all figures from `fedadmm_*.py` scripts are stored
8. `figures_final` repo: Where the paper's figures are stored

## Installation Instructions
Unfortunately, installation instructions for this repo are kinda a pain. Sorry! The main reason is that CVXPY's SCS 
solver implementation doesn't work for matrices larger than 2x2 (as of 3/22/2021) unless you have it installed with `blas+lapack`,
but you can only install _that_ if you have `numpy` with the `mkl` extension installed.

I used Python 3.7 in an Anaconda virtual environment, if that helps.

Install the following packages (in roughly the following order) from this link (or any alternative link that
kindly provides binaries of these libraries for your given Python version: https://www.lfd.uci.edu/~gohlke/pythonlibs/)
* numpy+mkl
* scipy
* scs
* cvxopt
* ecos
* osqp
* cvxpy
* matplotlib
* slycot (you don't need it for this library, but it's super common for other controls packages and this is the only way I've gotten it to work on Windows. Life hacks!)

## Instructions to Run Code
The default script I used for the paper is `fedadmm.py`, and the default ADMM script that I used was `ADMM_slack.py`

The installation was the hard part. Now you should be able to just run `python fedadmm.py` with your Conda environment active.