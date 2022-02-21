## This README file is a copy from Nick Kunz's package SMOGN and should be modified by the end of Winter 2022 semester!

<div align="center">
  <img src="https://github.com/paobranco/smogn/blob/master/media/images/smogn_banner.png">
</div>

## Imbalanced Learn Regression
[![PyPI version](https://badge.fury.io/py/smogn.svg)](https://badge.fury.io/py/smogn)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/nickkunz/smogn.svg?branch=master)](https://travis-ci.com/nickkunz/smogn)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1bfe5a201f3b4a9787c6cf4b365736ed)](https://www.codacy.com/manual/nickkunz/smogn?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nickkunz/smogn&amp;utm_campaign=Badge_Grade)
![GitHub last commit](https://img.shields.io/github/last-commit/paobranco/ImbalancedLearnRegression)

## Description
A Python implementation of sampling techniques for Regression. Conducts different sampling techniques for Regression. Useful for prediction problems where regression is applicable, but the values in the interest of predicting are rare or uncommon. This can also serve as a useful alternative to log transforming a skewed response variable, especially if generating synthetic data is also of interest.
<br>

## Features
1. An open-source Python supported version of sampling techniques for Regression, a variation of Nick Kunz's package SMOGN.

2. Supports Pandas DataFrame inputs containing mixed data types, auto distance metric selection by data type, and optional auto removal of missing values.

3. Flexible inputs available to control the areas of interest within a continuous response variable and friendly parameters for over-sampling synthetic data.

4. Purely Pythonic, developed for consistency, maintainability, and future improvement, no foreign function calls to C or Fortran, as contained in original R implementation.

## Requirements
1. Python 3
2. NumPy
3. Pandas

## Installation
```python
## install pypi release
pip install ImbalancedLearnRegression

## install developer version
pip install git+https://github.com/paobranco/ImbalancedLearnRegression.git
```

## Usage
```python
## load libraries
import ImbalancedLearnRegression
import pandas

## load data
housing = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/housing.csv"
)

## conduct ro
housing_ro = ro(
    
    data = housing, 
    y = "SalePrice"
    
)

## conduct gn
housing_gn = gn(
    
    data = housing, 
    y = "SalePrice"
    
)
```

## Examples
1. [Beginner](https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_1_beg.ipynb) <br>
2. [Intermediate](https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_2_int.ipynb) <br>
3. [Advanced](https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb) <br>

## License

© Paula Branco, 2022. Licensed under the General Public License v3.0 (GPLv3).

## Contributions

ImbalancedLearnRegression is open for improvements and maintenance. Your help is valued to make the package better for everyone.

## Reference

Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.

Branco, P., Torgo, L., & Ribeiro, R. P. (2019). Pre-processing approaches for imbalanced distributions in regression. Neurocomputing, 343, 76-99. https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638

Kunz, N., (2019). SMOGN. https://github.com/nickkunz/smogn

Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33



