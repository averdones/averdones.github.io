---
layout: post
title: >-
  Reading tabular data in Pytorch and training a multilayer perceptron
comments: true
published: true
---

Pytorch is a library that is normally used to train models that leverage unstructured data, such as images or text. However, it can also be used to train models that have tabular data as their input. This is nothing more than classic tables, where each row represents an observation and each column holds a variable. While deep learning models may not be the best suited to fit tabular data, since they might contain simple enough relationships such that a shallow linear regression or a support vector machine could lead to better predictions, you still may want to use Pytorch to solve your tabular shaped problem. For example, you might think that your problem is complex enough to try a deep network approach, or perhaps you are building a larger pipeline in Pytorch and you don't think it's worth introducing a new framework in your code.

Reading data in Pytorch can be very easy to do thanks to some already [implemented methods](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). However, if your data is not one of the famous datasets, such as MNIST, or is not stored in a specific way, instead of having a one-liner to read your data, you will have to code a whole new class. The jump from using a single line to read your data, to code an entire class for it can be daunting. Fortunately, Pytorch has tried to make things as easier as possible for the user, allowing us to inherit from their [*Dataset* class](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset), where things have been structured appropriately. In doing so, we will only have to **override two methods**:

- **\_\_len\_\_**: it has to return the number of rows of the input table.
- **\_\_getitem\_\_**: it has to return a single observation, including both the independent variables and the dependent variable. For example, the return could be something like $[[100, 2, -5], 0]$, where there are three independent variables that take the values 100, 2 and -5 respectively and the value of the dependent variable is 0. This method has a mandatory argument, which is the index of the table row. We will need to write our code thinking that this index can take any integer value, such as 10, which would mean *get row number 10* from our table. 

Additionaly, we could use the method **\_\_init\_\_** of our class to load and transform our data.

## A simple example

An example will make things much clearer. I will be reading a dataset from Kaggle called [Students Performance in Exams](https://www.kaggle.com/spscientist/students-performance-in-exams), that you can easily download and try for yourself. In this particular example, I will try to predict the variable *math score* based on all the other variables available. However, the specifics of the dataset are not relevant at the moment. The following code shows the complete class needed to read the downloaded table:

```python
class StudentsPerformanceDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.

        Args:
            csv_file (str): Path to the csv file with the students data.

        """
        df = pd.read_csv(csv_file)

        # Grouping variable names
        self.categorical = ["gender", "race/ethnicity", "parental level of education", "lunch",
                           "test preparation course"]
        self.target = "math score"

        # One-hot encoding of categorical variables
        self.students_frame = pd.get_dummies(df, prefix=self.categorical)

        # Save target and predictors
        self.X = self.students_frame.drop(self.target, axis=1)
        self.y = self.students_frame[self.target]

    def __len__(self):
        return len(self.students_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]
```
 
As you can see, the table is in *csv* format, so I decided to use *pandas* to load it. Also, there are some categorical variables that need to be one-hot encoded, and as previously stated, I used the \_\_init\_\_ method to do so. Since my target variable is *math score* I will save it in the python variable *y*, which represents the dependent variable of our problem. The rest of the variables are saved in the python variable called *X*.

The classical neural network to fit tabular data is the **Multilayer Perceptron**, which could be thought of as an extension of the linear and logistic regressions, depending on the activation function of the last layer: the identity function for linear regression and the sigmoid function for logistic regression. In fact, a multilayer perceptron without any hidden layers is exactly either a linear or a logistic regression (as long as the activation functions of the last layer are either the identity or the sigmoid).

Following, I will present a small script that can be run in order to read and train a small multilayer perceptron on the Students Performance data. The rest of the code contains the definition of a small model, the dataloaders, the choice of a loss function and an optimization algorithm, and the usual loop to fit the data using backpropagation.  

{% gist averdones/ff8e2c04962f585168278d73e4b4a48a %}

Now you should be able to start playing around with structured data in Pytorch. 

Enjoy your tabular data!