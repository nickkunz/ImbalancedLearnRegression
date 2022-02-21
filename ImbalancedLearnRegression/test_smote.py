## load libraries
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import ssl

from smoter import smoter

ssl._create_default_https_context = ssl._create_unverified_context

college = pd.read_csv(
        ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearnRegression/master/data/College.csv"
)

college_smote = smoter(

    ## main arguments
    data=college,  ## pandas dataframe
    y='Grad.Rate',  ## string ('header name')
)
college_smote.reset_index(inplace=True)

seaborn.kdeplot(college['Grad.Rate'], label="Original")
seaborn.kdeplot(college_smote['Grad.Rate'], label="Modified")
plt.show()