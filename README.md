# man-opt
 588 Project 1: Optimization on smooth manifolds for low-rank matrix completion

## Installation
Please run
```
pip install -r requirements.txt
```
before using the code.

## Tests
Tests can be run with the command
```
python -m pytest
```


## Usage
Refer to `experiment.py` for usage of this package - the API is similar to
`pymanopt`. Experiments can be run using the command
```
python experiment.py -m 200 -n 500 -r 5 --maxiter 500 --mu 1 --scale 1e-2
```
