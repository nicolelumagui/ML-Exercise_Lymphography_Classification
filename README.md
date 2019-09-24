
#### MACHINE LEARNING EXERCISE: CLASSIFICATION
# LYMPHOGRAPHY

#### Models
* K-Nearest Neighbors
* Decision Tree Classifier
* Random Forest Classifier

#### About
* This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature. (See also breast-cancer and primary-tumor.)

#### Target Variable
* class: normal find, metastases, malign lymph, fibrosis

#### Features
1. lymphatics: normal, arched, deformed, displaced
1. block of affere: no, yes
1. bl. of lymph. c: no, yes
1. bl. of lymph. s: no, yes
1. by pass: no, yes
1. extravasates: no, yes
1. regeneration of: no, yes
1. early uptake in: no, yes
1. lym.nodes dimin: 0-3
1. lym.nodes enlar: 1-4
1. changes in lym.: bean, oval, round
1. defect in node: no, lacunar, lac. marginal, lac. central
1. changes in node: no, lacunar, lac. margin, lac. central
1. changes in stru: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
1. special forms: no, chalices, vesicles
1. dislocation of: no, yes
1. exclusion of no: no, yes
1. no. of nodes in: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70

#### Source
* https://archive.ics.uci.edu/ml/datasets/Lymphography

## Import Libraries


```python
##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster")

%matplotlib inline
```


```python
##### Other Libraries #####

## Classification Algorithms ##
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

## For building models ##
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

## For measuring performance ##
from sklearn import metrics
from sklearn.model_selection import cross_val_score

## To visualize decision tree ##
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

## Ignore warnings ##
import warnings
warnings.filterwarnings('ignore')
```


    

## Load the Dataset

When opened on a text editor, it can be seen that the data file "lymphography.data" does not have column headers. To add headers, let's list the column names then specify this list when loading the data.


```python
### List columns names based on the description
col_names = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 
 'extravasates', 'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 
'changes in lym.', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 
'dislocation of', 'exclusion of no', 'no. of nodes in']
```


```python
### Load the data
df = pd.read_csv("lymphography.data", names=col_names)
print("Size of dataset:", df.shape)
df.head()
```

    Size of dataset: (148, 19)
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>lymphatics</th>
      <th>block of affere</th>
      <th>bl. of lymph. c</th>
      <th>bl. of lymph. s</th>
      <th>by pass</th>
      <th>extravasates</th>
      <th>regeneration of</th>
      <th>early uptake in</th>
      <th>lym.nodes dimin</th>
      <th>lym.nodes enlar</th>
      <th>changes in lym.</th>
      <th>defect in node</th>
      <th>changes in node</th>
      <th>changes in stru</th>
      <th>special forms</th>
      <th>dislocation of</th>
      <th>exclusion of no</th>
      <th>no. of nodes in</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>8</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>




## Explore the Dataset


```python
df.describe()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>lymphatics</th>
      <th>block of affere</th>
      <th>bl. of lymph. c</th>
      <th>bl. of lymph. s</th>
      <th>by pass</th>
      <th>extravasates</th>
      <th>regeneration of</th>
      <th>early uptake in</th>
      <th>lym.nodes dimin</th>
      <th>lym.nodes enlar</th>
      <th>changes in lym.</th>
      <th>defect in node</th>
      <th>changes in node</th>
      <th>changes in stru</th>
      <th>special forms</th>
      <th>dislocation of</th>
      <th>exclusion of no</th>
      <th>no. of nodes in</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
      <td>148.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.452703</td>
      <td>2.743243</td>
      <td>1.554054</td>
      <td>1.175676</td>
      <td>1.047297</td>
      <td>1.243243</td>
      <td>1.506757</td>
      <td>1.067568</td>
      <td>1.702703</td>
      <td>1.060811</td>
      <td>2.472973</td>
      <td>2.398649</td>
      <td>2.966216</td>
      <td>2.804054</td>
      <td>5.216216</td>
      <td>2.331081</td>
      <td>1.662162</td>
      <td>1.790541</td>
      <td>2.601351</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.575396</td>
      <td>0.817509</td>
      <td>0.498757</td>
      <td>0.381836</td>
      <td>0.212995</td>
      <td>0.430498</td>
      <td>0.501652</td>
      <td>0.251855</td>
      <td>0.458621</td>
      <td>0.313557</td>
      <td>0.836627</td>
      <td>0.568323</td>
      <td>0.868305</td>
      <td>0.761834</td>
      <td>2.171368</td>
      <td>0.777126</td>
      <td>0.474579</td>
      <td>0.408305</td>
      <td>1.905023</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>




By observing the count in the describe table above, luckily, there is no missing data. 

Also, the min and max values  each of the column corresponds to the dataset description posted so we can assume assume that we properly added the column names.

Now, let's look how balanced or imbalanced the classes (target variable) in the dataset.


```python
df["class"].value_counts()
```




    2    81
    3    61
    4     4
    1     2
    Name: class, dtype: int64



Majority of the classes 2 (metastases) and 3 (malign lymph) dominate the dataset. While the classes 4 (fibrosis) and 1 (normal find) are really under-represented.

Because of this, we will create and compare models trained using the upsampled data and the regular (not upsampled) data.

## Prepare the Data for Modelling

### Train-Test Split

As usual, we should separate the target variable from the predictors then split the data into train and test sets.


```python
### Split the features and the target column.
x = df.drop('class', axis=1)
y = df['class']

print("Size of x (predictors): {}\nSize of y (target): {}".format(x.shape, y.shape))
```

    Size of x (predictors): (148, 18)
    Size of y (target): (148,)
    


```python
### Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

### Check shape to make sure it is all in order
print("Size of x_train: {} \t Size of x_test: {} \nSize of y_train: {} \t Size of y_test: {}".format(
    x_train.shape, x_test.shape, y_train.shape, y_test.shape))

#print(y_train.value_counts(), '\n', y_test.value_counts())
```

    Size of x_train: (103, 18) 	 Size of x_test: (45, 18) 
    Size of y_train: (103,) 	 Size of y_test: (45,)
    

### Upsample / Oversample

We've seen above that the dataset is imbalanced. As a workaround, we will upsample/oversample minority classes so that its count is same as the major classes

To do this, we should first merge the x_train and y_train. Shown below is the value counts of target variable before upsampling.


```python
df_train = pd.concat([x_train, y_train], axis=1)
print("DF Train shape:", df_train.shape, "\nDF Train value counts:\n",df_train['class'].value_counts())
df_train.head()
```

    DF Train shape: (103, 19) 
    DF Train value counts:
     2    53
    3    47
    4     2
    1     1
    Name: class, dtype: int64
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lymphatics</th>
      <th>block of affere</th>
      <th>bl. of lymph. c</th>
      <th>bl. of lymph. s</th>
      <th>by pass</th>
      <th>extravasates</th>
      <th>regeneration of</th>
      <th>early uptake in</th>
      <th>lym.nodes dimin</th>
      <th>lym.nodes enlar</th>
      <th>changes in lym.</th>
      <th>defect in node</th>
      <th>changes in node</th>
      <th>changes in stru</th>
      <th>special forms</th>
      <th>dislocation of</th>
      <th>exclusion of no</th>
      <th>no. of nodes in</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>119</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>




After merging the x_train and y_train, let's create a dataframe for the upsampled data and initialize it with the dominating class. 

Then, we can upsample the minority classes so that its value counts will be same as of the dominating class. In this case, we will upsample the classes 1, 3 and 4 to have 53 rows each.


```python
### Append the class with highest value counts
df_train_up = df_train[df_train["class"]==2]

### Upsample minority classes
for n in [1, 3, 4]:
    upsampled = resample(df_train[df_train["class"]==n],
                        replace=True, # sample with replacement
                        n_samples=53, # match number in majority class
                        random_state=1) # reproducible results
    df_train_up = pd.concat([df_train_up, upsampled]) 

### Print upsampled training set to check
print("Size of df_train_up:", df_train_up.shape, "\nValue counts for class:\n", df_train_up["class"].value_counts())
df_train_up.head()
```

    Size of df_train_up: (212, 19) 
    Value counts for class:
     4    53
    3    53
    2    53
    1    53
    Name: class, dtype: int64
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lymphatics</th>
      <th>block of affere</th>
      <th>bl. of lymph. c</th>
      <th>bl. of lymph. s</th>
      <th>by pass</th>
      <th>extravasates</th>
      <th>regeneration of</th>
      <th>early uptake in</th>
      <th>lym.nodes dimin</th>
      <th>lym.nodes enlar</th>
      <th>changes in lym.</th>
      <th>defect in node</th>
      <th>changes in node</th>
      <th>changes in stru</th>
      <th>special forms</th>
      <th>dislocation of</th>
      <th>exclusion of no</th>
      <th>no. of nodes in</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>124</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>123</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>




Now, we have a perfectly balanced data!

Let's separate the predictors from the target variable again to be used for modelling.


```python
x_train_up = df_train_up.drop(["class"], axis=1)
y_train_up = df_train_up["class"]

print("Size of x_train_up: {}\nSize of y_train_up: {}".format(x_train_up.shape, y_train_up.shape))
```

    Size of x_train_up: (212, 18)
    Size of y_train_up: (212,)
    

## Build the Models

Before building the models, the function "confmatrix" is initialized for easy plotting of confusion matrix.


```python
def confmatrix(y_pred, title):
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title(title)
    
    sns.set(font_scale=1.4) # For label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # Font size
```

### K-Nearest Neighbors

#### Build/Train the KNN model

Iterate through KNN to find the optimal Ks for respective KNN models that use regular data and upsampled data.


```python
### Finding the best k

best_k = {"Regular":0, "Upsampled":0}
best_score = {"Regular":0, "Upsampled":0}

for k in range(3, 50, 2):
    
    ## Using Regular / Not upsampled training set
    knn_temp = KNeighborsClassifier(n_neighbors=k)              # Instantiate the model
    knn_temp.fit(x_train, y_train)                              # Fit the model to the training set
    knn_temp_pred = knn_temp.predict(x_test)                    # Predict on the test set
    score = metrics.accuracy_score(y_test, knn_temp_pred) * 100 # Get accuracy
    if score >= best_score["Regular"] and score < 100:          # Store best params
        best_score["Regular"] = score
        best_k["Regular"] = k
        
    ## Using Upsampled training set
    knn_temp = KNeighborsClassifier(n_neighbors=k)              # Instantiate the model
    knn_temp.fit(x_train_up, y_train_up)                        # Fit the model to the training set
    knn_temp_pred = knn_temp.predict(x_test)                    # Predict on the test set
    score = metrics.accuracy_score(y_test, knn_temp_pred) * 100 # Get accuracy
    if score >= best_score["Upsampled"] and score < 100:        # Store best params
        best_score["Upsampled"] = score
        best_k["Upsampled"] = k
        
### Print the best score and best k
print("---Best results---\nK: {}\nScore: {}".format(best_k, best_score))
```

    ---Best results---
    K: {'Regular': 5, 'Upsampled': 29}
    Score: {'Regular': 71.11111111111111, 'Upsampled': 77.77777777777779}
    

Now that the optimal K is found, the final KNN models are initialized below.


```python
### Build final models using the best k

## Instantiate the models
knn = KNeighborsClassifier(n_neighbors=best_k["Regular"])
knn_up = KNeighborsClassifier(n_neighbors=best_k["Upsampled"])

## Fit the model to the training set
knn.fit(x_train, y_train)
knn_up.fit(x_train_up, y_train_up)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=29, p=2,
                         weights='uniform')



#### Validate the KNN model


```python
### Predict on the test set
knn_pred = knn.predict(x_test)
knn_pred_up = knn_up.predict(x_test)
```

###### Classification Report


```python
### Get performance metrics
knn_score = metrics.accuracy_score(y_test, knn_pred) * 100
knn_score_up = metrics.accuracy_score(y_test, knn_pred_up) * 100

### Print classification report for regular
print("----- Regular Training Set Used -----")
print("Classification report for {}:\n{}".format(knn, metrics.classification_report(y_test, knn_pred)))
print("Accuracy score:", knn_score)

### Print classification report for upsampled
print("\n----- Upsampled Training Set Used -----")
print("Classification report for {}:\n{}".format(knn_up, metrics.classification_report(y_test, knn_pred_up)))
print("Accuracy score:", knn_score_up)
```

    ----- Regular Training Set Used -----
    Classification report for KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform'):
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         1
               2       0.79      0.82      0.81        28
               3       0.56      0.64      0.60        14
               4       0.00      0.00      0.00         2
    
        accuracy                           0.71        45
       macro avg       0.34      0.37      0.35        45
    weighted avg       0.67      0.71      0.69        45
    
    Accuracy score: 71.11111111111111
    
    ----- Upsampled Training Set Used -----
    Classification report for KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=29, p=2,
                         weights='uniform'):
                  precision    recall  f1-score   support
    
               1       0.50      1.00      0.67         1
               2       0.83      0.86      0.84        28
               3       0.67      0.57      0.62        14
               4       1.00      1.00      1.00         2
    
        accuracy                           0.78        45
       macro avg       0.75      0.86      0.78        45
    weighted avg       0.78      0.78      0.77        45
    
    Accuracy score: 77.77777777777779
    

Based on the accuracy scores, the KNN model that used upsampled data performs better than the one that used regular data.

Shown below are the confusion matrices for each KNN model.

##### Confusion Matrix


```python
### Plot confusion matrix
confmatrix(knn_pred, "Confusion Matrix\nKNN - Regular Training Set")
confmatrix(knn_pred_up, "Confusion Matrix\nKNN - Upsampled Training Set")
```


![png](Images/output_39_0.png)



![png](Images/output_39_1.png)


If we observed the confusion matrices above, we can see that the KNN model that used the upsampled data predicted the minority classes (classes 1 and 4) well.

##### Cross-Validation of the KNN model


```python
### Perform cross-validation then get the mean
knn_cv = np.mean(cross_val_score(knn, x, y, cv=10) * 100)
print("10-Fold Cross-Validation score for KNN fit in Regular Training Set:", knn_cv)
```

    10-Fold Cross-Validation score for KNN fit in Regular Training Set: 74.48412698412697
    

### Decision Tree Classifier

#### Build/Train the DTree model

Two Decision Tree models are built. The variable *dtree* is fit on the regular data, while *dtree_up* is fit on upsampled data


```python
### Instantiate the model
dtree = tree.DecisionTreeClassifier()
dtree_up = tree.DecisionTreeClassifier()

### Fit the model to the training set
dtree.fit(x_train, y_train)
dtree_up.fit(x_train_up, y_train_up)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



#### Validate the DTree model


```python
### Predict on the test set
dtree_pred = dtree.predict(x_test)
dtree_pred_up = dtree_up.predict(x_test)
```

##### Classification Report


```python
### Get performance metrics
dtree_score = metrics.accuracy_score(y_test, dtree_pred) * 100
dtree_score_up = metrics.accuracy_score(y_test, dtree_pred_up) * 100

### Print classification report for regular
print("----- Regular Training Set Used -----")
print("Classification report for {}:\n{}".format(dtree, metrics.classification_report(y_test, dtree_pred)))
print("Accuracy score:", dtree_score)

### Print classification report for upsampled
print("\n----- Upsampled Training Set Used -----")
print("Classification report for {}:\n{}".format(dtree_up, metrics.classification_report(y_test, dtree_pred_up)))
print("Accuracy score:", dtree_score_up)
```

    ----- Regular Training Set Used -----
    Classification report for DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best'):
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         1
               2       0.75      0.75      0.75        28
               3       0.59      0.71      0.65        14
               4       0.00      0.00      0.00         2
    
        accuracy                           0.69        45
       macro avg       0.33      0.37      0.35        45
    weighted avg       0.65      0.69      0.67        45
    
    Accuracy score: 68.88888888888889
    
    ----- Upsampled Training Set Used -----
    Classification report for DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best'):
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00         1
               2       0.88      0.82      0.85        28
               3       0.71      0.86      0.77        14
               4       1.00      0.50      0.67         2
    
        accuracy                           0.82        45
       macro avg       0.90      0.79      0.82        45
    weighted avg       0.84      0.82      0.82        45
    
    Accuracy score: 82.22222222222221
    

Same as in KNN, the accuracy of *dtree_up* is way better than of the *dtree*. 

Plotted below are the confusion matrices for each models.

##### Confusion Matrix


```python
### Plot confusion matrix
confmatrix(dtree_pred, "Confusion Matrix\nDtree - Regular Training Set")
confmatrix(dtree_pred_up, "Confusion Matrix\nDTree - Upsampled Training Set")
```


![png](Images/output_53_0.png)



![png](Images/output_53_1.png)


As expected, *dtree_up* predicted classes 1 and 4 better than *dtree*.

##### Cross-Validation of the DTree model


```python
### Perform cross-validation then get the mean
dtree_cv = np.mean(cross_val_score(dtree, x, y, cv=10) * 100)
print("Cross-Validation score for DTree (10 folds):", knn_cv)
```

    Cross-Validation score for DTree (10 folds): 74.48412698412697
    

#### Feature Importance

Let's look and compare the feature importances based on both decision trees.

The first table shows the complete and unsorted feature importances set by the decision trees on each predictors.

The second table shows the rank of feature importances of the decision trees.


```python
### Extract Feature importance
### Then put into a DataFrame along with Feature Names for easier understanding.
df_feature_importance = pd.DataFrame(dtree.feature_importances_, index=x_train.columns, 
                                     columns=["Regular-Importance"])
df_feature_importance_up = pd.DataFrame(dtree_up.feature_importances_, index=x_train_up.columns, 
                                        columns=["Upsampled-Importance"])

### Merge the regular and upsampled feature importance
df_feature_importance_merged = pd.concat([df_feature_importance, df_feature_importance_up], axis=1)

### Print
print("Feature Importance - Complete")
df_feature_importance_merged.sort_values(["Regular-Importance"],ascending=False)
```

    Feature Importance - Complete
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regular-Importance</th>
      <th>Upsampled-Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>special forms</th>
      <td>0.239643</td>
      <td>0.013045</td>
    </tr>
    <tr>
      <th>lym.nodes enlar</th>
      <td>0.233469</td>
      <td>0.051908</td>
    </tr>
    <tr>
      <th>block of affere</th>
      <td>0.174549</td>
      <td>0.068307</td>
    </tr>
    <tr>
      <th>changes in lym.</th>
      <td>0.103258</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>changes in node</th>
      <td>0.051547</td>
      <td>0.010063</td>
    </tr>
    <tr>
      <th>changes in stru</th>
      <td>0.051103</td>
      <td>0.003250</td>
    </tr>
    <tr>
      <th>early uptake in</th>
      <td>0.034041</td>
      <td>0.036235</td>
    </tr>
    <tr>
      <th>lymphatics</th>
      <td>0.025982</td>
      <td>0.342155</td>
    </tr>
    <tr>
      <th>defect in node</th>
      <td>0.024585</td>
      <td>0.007471</td>
    </tr>
    <tr>
      <th>exclusion of no</th>
      <td>0.024585</td>
      <td>0.016771</td>
    </tr>
    <tr>
      <th>extravasates</th>
      <td>0.020115</td>
      <td>0.009704</td>
    </tr>
    <tr>
      <th>no. of nodes in</th>
      <td>0.017122</td>
      <td>0.107759</td>
    </tr>
    <tr>
      <th>lym.nodes dimin</th>
      <td>0.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>regeneration of</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>by pass</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bl. of lymph. s</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>bl. of lymph. c</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dislocation of</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>





```python
### Show top 5 important features
## Get top features for each model
df_top_features_reg = df_feature_importance.sort_values(["Regular-Importance"],ascending=False).reset_index(level=0).rename(columns={'index':'Regular-Feature'})
df_top_features_up = df_feature_importance_up.sort_values(["Upsampled-Importance"],ascending=False).reset_index(level=0).rename(columns={'index':'Upsampled-Feature'})

## Merge the top features
df_top_features = pd.concat([df_top_features_reg, df_top_features_up], axis=1)

## Print results
print("Ranked Feature Importance")
df_top_features
```

    Ranked Feature Importance
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regular-Feature</th>
      <th>Regular-Importance</th>
      <th>Upsampled-Feature</th>
      <th>Upsampled-Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>special forms</td>
      <td>0.239643</td>
      <td>lymphatics</td>
      <td>0.342155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lym.nodes enlar</td>
      <td>0.233469</td>
      <td>lym.nodes dimin</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>block of affere</td>
      <td>0.174549</td>
      <td>no. of nodes in</td>
      <td>0.107759</td>
    </tr>
    <tr>
      <th>3</th>
      <td>changes in lym.</td>
      <td>0.103258</td>
      <td>block of affere</td>
      <td>0.068307</td>
    </tr>
    <tr>
      <th>4</th>
      <td>changes in node</td>
      <td>0.051547</td>
      <td>lym.nodes enlar</td>
      <td>0.051908</td>
    </tr>
    <tr>
      <th>5</th>
      <td>changes in stru</td>
      <td>0.051103</td>
      <td>early uptake in</td>
      <td>0.036235</td>
    </tr>
    <tr>
      <th>6</th>
      <td>early uptake in</td>
      <td>0.034041</td>
      <td>exclusion of no</td>
      <td>0.016771</td>
    </tr>
    <tr>
      <th>7</th>
      <td>lymphatics</td>
      <td>0.025982</td>
      <td>special forms</td>
      <td>0.013045</td>
    </tr>
    <tr>
      <th>8</th>
      <td>defect in node</td>
      <td>0.024585</td>
      <td>changes in node</td>
      <td>0.010063</td>
    </tr>
    <tr>
      <th>9</th>
      <td>exclusion of no</td>
      <td>0.024585</td>
      <td>extravasates</td>
      <td>0.009704</td>
    </tr>
    <tr>
      <th>10</th>
      <td>extravasates</td>
      <td>0.020115</td>
      <td>defect in node</td>
      <td>0.007471</td>
    </tr>
    <tr>
      <th>11</th>
      <td>no. of nodes in</td>
      <td>0.017122</td>
      <td>changes in stru</td>
      <td>0.003250</td>
    </tr>
    <tr>
      <th>12</th>
      <td>lym.nodes dimin</td>
      <td>0.000000</td>
      <td>bl. of lymph. s</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>regeneration of</td>
      <td>0.000000</td>
      <td>by pass</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>by pass</td>
      <td>0.000000</td>
      <td>regeneration of</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>bl. of lymph. s</td>
      <td>0.000000</td>
      <td>bl. of lymph. c</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>bl. of lymph. c</td>
      <td>0.000000</td>
      <td>changes in lym.</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dislocation of</td>
      <td>0.000000</td>
      <td>dislocation of</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>




The top features of *dtree* are way different than that of *dtree_up*. Only the predictors *block of affere* and *lym.nodes enlar* remain on the top 5 features of both models.

These rankings can be used for future optimizations of the model and feature engineering.

#### Visualize DTrees


```python
### Visualize Dtree model that used regular training set

print("Decision Tree Plot - Regular Training Set Used")
print("Depth: {} \t N-Leaves: {}".format(dtree.get_depth(), dtree.get_n_leaves()))

## Get the feature/attribute columns
feature_col = x_train.columns

## Get the class column
class_col = pd.unique(y_train)
class_col = np.array(class_col)
class_col = str(class_col).replace(" ", "")

## Plot tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
               feature_names=feature_col, class_names=class_col)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

    Decision Tree Plot - Regular Training Set Used
    Depth: 8 	 N-Leaves: 20
    




![png](Images/output_63_1.png)




```python
### Visualize Dtree model that used upsampled training set

print("Decision Tree Plot - Upsampled Training Set Used")
print("Depth: {} \t N-Leaves: {}".format(dtree_up.get_depth(), dtree_up.get_n_leaves()))

## Get the feature/attribute columns
feature_col = x_train_up.columns

## Get the class column
class_col = pd.unique(y_train_up)
class_col = np.array(class_col)
class_col = str(class_col).replace(" ", "")

## Plot tree
dot_data = StringIO()
export_graphviz(dtree_up, out_file=dot_data, filled=True, rounded=True, special_characters=True,
               feature_names=feature_col, class_names=class_col)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```

    Decision Tree Plot - Upsampled Training Set Used
    Depth: 9 	 N-Leaves: 17
    




![png](Images/output_64_1.png)



The depth of *dtree_up* is greater than that of *dtree* but it has fewer leaves.

### Random Forest Classifier

#### Build/Train the RF model

Same as in the previous models, two random forest models are created: *rf* fit on not upsampled data and *rf_up* fit on upsampled data.


```python
### Instantiate algorithm
rf = RandomForestClassifier()
rf_up = RandomForestClassifier()

### Fit the model to the data
rf.fit(x_train, y_train)
rf_up.fit(x_train_up, y_train_up)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



#### Validate the RF model


```python
### Predict on the test set
rf_pred = rf.predict(x_test)
rf_pred_up = rf_up.predict(x_test)
```

##### Classification Report


```python
### Get performance metrics
rf_score = metrics.accuracy_score(y_test, rf_pred) * 100
rf_score_up = metrics.accuracy_score(y_test, rf_pred_up) * 100

### Print classification report for regular
print("----- Regular Training Set Used -----")
print("Classification report for {}:\n{}".format(rf, metrics.classification_report(y_test, rf_pred)))
print("Accuracy score:", dtree_score)

### Print classification report for upsampled
print("\n----- Upsampled Training Set Used -----")
print("Classification report for {}:\n{}".format(rf_up, metrics.classification_report(y_test, rf_pred_up)))
print("Accuracy score:", rf_score_up)
```

    ----- Regular Training Set Used -----
    Classification report for RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False):
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00         1
               2       0.79      0.93      0.85        28
               3       0.82      0.64      0.72        14
               4       1.00      0.50      0.67         2
    
        accuracy                           0.80        45
       macro avg       0.65      0.52      0.56        45
    weighted avg       0.79      0.80      0.78        45
    
    Accuracy score: 68.88888888888889
    
    ----- Upsampled Training Set Used -----
    Classification report for RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False):
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00         1
               2       0.86      0.86      0.86        28
               3       0.67      0.71      0.69        14
               4       1.00      0.50      0.67         2
    
        accuracy                           0.80        45
       macro avg       0.88      0.77      0.80        45
    weighted avg       0.81      0.80      0.80        45
    
    Accuracy score: 80.0
    

The model fit on upsampled data also performs better.

##### Confusion Matrix


```python
### Plot confusion matrix
confmatrix(rf_pred, "Confusion Matrix\nRandom Forest - Regular Training Set")
confmatrix(rf_pred_up, "Confusion Matrix\nRandom Forest - Upsampled Training Set")
```


![png](Images/output_76_0.png)



![png](Images/output_76_1.png)


When it comes to classifying the minority classes, the Random Forest model fit in upsampled data did not perform better than the previous models.

##### Cross-Validation of the RF model


```python
### Perform cross-validation then get the mean
rf_cv = np.mean(cross_val_score(rf, x, y, cv=10) * 100)
print("Cross-Validation score for RandomForest (10 folds):", rf_cv)
```

    Cross-Validation score for RandomForest (10 folds): 81.25198412698413
    

## Summary of Results


```python
df_results = pd.DataFrame.from_dict({
    'Regular - Accuracy Score':{'KNN':knn_score, 'Decision Tree':dtree_score, 'Random Forest':rf_score},
    'Upsampled - Accuracy Score':{'KNN':knn_score_up, 'Decision Tree':dtree_score_up, 'Random Forest':rf_score_up},
    'Cross-Validation Score':{'KNN':knn_cv, 'Decision Tree':dtree_cv, 'Random Forest':rf_cv}
    })
df_results
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regular - Accuracy Score</th>
      <th>Upsampled - Accuracy Score</th>
      <th>Cross-Validation Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decision Tree</th>
      <td>68.888889</td>
      <td>82.222222</td>
      <td>74.849206</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>71.111111</td>
      <td>77.777778</td>
      <td>74.484127</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>81.251984</td>
    </tr>
  </tbody>
</table>




**Upsampling / Oversampling data improves the model performance, especially on cases like this where it is hard to gather data.**

The Decision Tree fit on upsampled data has the highest accuracy score. But this does not mean that this is the overall best model for this dataset.

**It still depends on the use case whether which models and metrics to consider.** 

For example, if we want to focus on classifying metastases (class == 2), it may be better to use Random forest than Decision Tree because the Random Forest model fit on upsampled data classifies metastases better than all the other models tried here.

Most of the data out there are imbalanced, and oversampling is just one of the many techniques to have a work around on this problem.

## Special Thanks
* [FTW Foundation](https://ftwfoundation.org)
