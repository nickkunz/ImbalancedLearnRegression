## load libraries
from gn import gn
from ro import ro
import pandas

## load data
housing = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/housing.csv"
)

## conduct smogn
housing_smogn = ro(
    
    data = housing, 
    y = "SalePrice",
    #replace = True,
    #under_samp=False
    
    
)
