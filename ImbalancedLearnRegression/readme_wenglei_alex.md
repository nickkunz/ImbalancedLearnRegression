Introduction
In the machine learning training process, there is a problem of unbalanced training data categories, and the main purpose of these two methods is to obtain a balanced sample distribution by changing the
The main purpose of these two methods is to obtain a balanced sample distribution by changing the original unbalanced sample set, and then learn a suitable model.
The difference lies mainly in the different approaches to balancing the data classes.

Description: This is a python package of Over-sampling technique, there are two parts of the package: RO and GN. 

Random Over-sampling (RO)
What is it for?
Random oversampling: Randomly replicating and repeating a small number of classes, so that the number of classes is the same as the number of classes in the majority, resulting in a new balanced data set. Here, for the regression task, there is no explicit class division, but rather a continuum of values (the corresponding y columns in the function below), so the data are divided into two classes (rel_thres) based on the correlation coefficients by checking the data for correlation. The data imbalance is solved by randomly oversampling the data in the smaller category (randomly selected numbers are copied and the oversampling ratio perc_o is set according to the function).

 
def ro(
    ## main arguments / inputs
    data,
    y,
    pert = 0.02,
## training set (pandas dataframe)
## response variable y by name (string)
## perturbation / noise percentage (pos
real)
    samp_method = "balance",  ## over / under sampling ("balance" or extreme")
    drop_na_col = True,
    drop_na_row = True,
    replace = False,
    manual_perc = False,
## auto drop columns with nan's (bool)
## auto drop rows with nan's (bool)
## sampling replacement (bool)
## user defines percentage of under-

sampling and over-sampling  # added
    perc_o = -1,              ## percentage of over-sampling  # added
    ## phi relevance function arguments / inputs
    rel_thres = 0.5,          ## relevance threshold considered rare(pos real)
    
    rel_method = "auto","manual")
    rel_xtrm_type = "both", "both")
    rel_coef = 1.5,
    rel_ctrl_pts_rg = Nonearray)
)

Parameters:
data (pd.Dataframe):The data to be balanced
y (string):the column name of the tag class in data pert ( between 0 and 1):noise ratio samp_method ("balance" or "extreme"):over/undersampled method/bucket drop_na_col (bool):if nan is present in the data, remove/ignore the columns of nan drop_na_row (bool):if nan is present in the data, remove/ignore the rows of nan replace (bool):If True, then copy a variable (no changes to data) manual_perc (bool):User-defined oversampling/undersampling ratio
perc_o (float):proportion of oversampling (greater than 0) phi correlation coefficient correlation:
rel_thresre (0-1):correlation coefficient threshold. Divides data into two categories, greater than threshold and less than threshold rel_method ("auto" or "manual"): method of correlation coefficient function, auto uses quartiles for calculation, manual uses user-defined correlation coefficients (manual is recommended for experts in the comments)
rel_xtrm_type ("high", "low", "both"): whether to keep the distribution below or above the quantile or both
rel_coef (float >0):box plot coefficients
rel_ctrl_pts_rg (2d array): coefficients corresponding to the manual above





Gaussian Noise
What is it for?
Gaussian noise: Noise that obeys the probability density function of normal distribution. Here, for the regression task, there is no clear class division, but a continuous value (the corresponding y column in the function below), so
so that the data are divided into two categories (rel_thres) by performing correlation detection on the data, based on the correlation coefficients. The class with a large amount of data and the class with a small amount of data
The two types of data are balanced by the method of Gaussian noise oversampling (randomly selected data with Gaussian noise, and the oversampling ratio perc_o is set according to the function). For the class with a large amount of data, random undersampling is performed by the method of random undersampling (in proportion to perc_u ).

Parameters:
data (pd.Dataframe):The data to be balanced
y (string):the column name of the tag class in data
pert ( between 0 and 1):noise ratio
samp_method ("balance" or "extreme"):over/undersampled method/bucket under_samp (bool):whether to undersample
drop_na_col (bool):Delete/ignore nan's columns if nan is present in the data drop_na_row (bool):Delete/ignore nan's rows if nan is present in the data
replace (bool):If True, then copy a variable (no changes to data) manual_perc (bool):User-defined oversampling/undersampling ratio
perc_u (float): the ratio of undersampling (greater than 0)
perc_o (float): the ratio of oversampling (greater than 0)
phi correlation coefficient correlation:
rel_thresre (0-1):correlation coefficient threshold. Divides data into two categories, greater than threshold and less than threshold rel_method ("auto" or "manual"): method of correlation coefficient function, auto uses quartiles for calculation, manual uses user-defined correlation coefficients (manual is recommended for experts in the comments)
rel_xtrm_type ("high", "low", "both"): whether to keep the distribution below or above the quantile or both
rel_coef (float >0):box plot coefficients
rel_ctrl_pts_rg (2d array): coefficients corresponding to the manual above



Requirements





Installation
TBD after uloaded.



Usage


Reference
Branco, P., Torgo, L., Ribeiro, R. (2017).
SMOGN: A Pre-Processing Approach for Imbalanced Regression.
Proceedings of Machine Learning Research, 74:36-50.
http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
