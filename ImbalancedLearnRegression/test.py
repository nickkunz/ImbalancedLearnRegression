## load libraries
from gn import gn
from ro import ro
import pandas

## load data
housing = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/housing.csv"
)

## conduct ro
housing_ro = ro(
    
    data = housing, 
    y = "SalePrice",
    #replace = True,
    #under_samp=False
    
)

## load data
college = pandas.read_csv(
    
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/College.csv"
)

## conduct gn
college_gn = gn(
    
    data = college, 
    y = "Grad.Rate",
    #replace = True,
    #under_samp=False
     
)

# ## load data
# diabetic = pandas.read_csv(
    
#     ## http://jse.amstat.org/v19n3/decock.pdf
#     "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/diabetic_data.csv"
# )

# ## conduct smogn
# diabetic_gn = gn(
    
#     data = diabetic, 
#     y = "num_lab_procedures",
#     #replace = True,
#     #under_samp=False
     
# )
