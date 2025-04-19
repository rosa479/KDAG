# [Statistical Modeling: The Two Cultures](https://www2.math.uu.se/~thulin/mm/breiman.pdf)

# Introduction
The goal of statistics is to study how nature converts input variables to output and predict outputs and extract meaningful data about the process.
<center>x -> [Nature] -> y</center>
There are two approaches to this :
### The Data Modelling Culture
We assume a stochastic data model (one that incorporates probability and randomness in its structure) :
<center>response variables = f(predictor variables, random noise, parameters)</center>
Methods like linear regression, logistic regression and Cox Model are used.

### The Algorithmic Modelling Culture
The analysis in this culture considers the inside of the box complex and unknown. Decision Trees and Neural Nets are used. It does not rely on strict assumptions about the underlying data distribution but instead lets the data determine the best model.

The author argues that the focus of statistical community on data models has led to irrelevant theory and questionable scientific conclusions, kept them from using more suitable algorithmic models and prevented them from working on new exciting problems.


# Projects in Consulting
As a consultant, the author worked on a diverse set of prediction projects like :
- Predicting next-day ozone levels.  
- Using mass spectra to identify halogen-containing compounds.  
- Predicting the class of a ship from high-altitude radar returns.  
- Using sonar returns to predict the class of a submarine.  
- Identity of hand-sent Morse Code.  
- Toxicity of chemicals.  
- On-line prediction of the cause of a freeway traffic breakdown.  
- Speech recognition.  
- The sources of delay in criminal trials in state court systems.  
First two are discussed below.
### The Ozone Project
In the mid to late 1960s ozone levels became a serious health problem in the LA Basin and alerts were issued. The major source of ozone at that time was automobile tailpipe emissions. The alert warnings were issued in the morning, but would be more effective if they could be issued 12 hours in advance. In the mid-1970s, the EPA funded a large effort to see if ozone levels could be accurately predicted 12 hours in advance.

With the total amount of emissions about constant, the resulting ozone levels depend on the meteorology of the preceding days. A large data base was assembled consisting of lower and upper air measurements at U.S. weather stations as far away as Oregon and Arizona, together with hourly readings of surface temperature, humidity, and wind speed at the dozens of air pollution stations in the Basin and nearby areas.

Altogether, there were daily and hourly readings of over 450 meteorological variables for a period of seven years, with corresponding hourly values of ozone and other pollutants in the Basin. Let **x** be the predictor vector of meteorological variables on the nth day. There are more than 450 variables in **x** since information several days back is included. Let _y_ be the ozone level on the (_n_ + 1)st day. Then the problem was to construct a function _f_(**x**) such that for any future day and future predictor variables **x** for that day, _f_(**x**) is an accurate predictor of the next day’s ozone level _y_.

First five years of data were used as training set and the last two as testing set. Large linear regressions were run, followed by variable selection. In the end the project was a failure with too many false alarms and the author regrets it cannot be revisited with the tools available today.
### The Chlorine Project
The EPA samples thousands of compounds a year and tries to determine their potential toxicity. Mass spectra was measured to determine the chemical structure. This was cheap but the examination by a trained chemist was not. So the EPA funded a project of determining presence of chlorine from the mass spectra.
The problem is to construct a function _f_(**x**) that is an accurate predictor of _y_ where **x** is the mass spectrum of the compound.

To measure predictive accuracy the data set was randomly divided into a 25,000 member training set and a 5,000 member test set. Linear discriminant analysis was tried, then quadratic discriminant analysis. These were difficult to adapt to the variable dimensionality. By this time I was thinking about decision trees. The hallmarks of chlorine in mass spectra were researched. This domain knowledge was incorporated into the decision tree algorithm by the design of the set of 1,500 yes–no questions that could be applied to a mass spectra of any dimensionality. The result was a decision tree that gave 95% accuracy on both chlorines and nonchlorines.

### Perceptions of Statistical Analysis
- Live with the data before you plunge into modelling it
- Search for a model that gives the best solution, either algorithmic or data
- Computers are an indispensable partner

Note : A **goodness of fit test** is a statistical test used to determine how well a model fits the observed data. It helps assess whether the assumed probability distribution or statistical model adequately represents the given dataset.
# The use of data models
Statisticians in applied research consider data modelling as the template for statistical analysis.
If the model is a poor emulation of nature, the conclusions may be wrong.
Statisticians have been overly focused on **parametric models**, limiting their ability to adapt to complex real-world problems.
Traditional models often focus on **interpretability** but sacrifice **predictive accuracy**.
These truisms have often been ignored in the enthusiasm for fitting data models. A few decades ago, the commitment to data models was such that even simple precautions such as residual analysis or goodness-of-fit tests were not used. The belief in the infallibility of data models was almost religious. It is a strange phenomenon—once a model is made, then it becomes truth and the conclusions from it are infallible.

### Problems in Current Data Modelling
A model can fit past data well but still produce misleading conclusions.
Many published studies **fail to adequately discuss** how well their models fit the data.
The focus in statistical research often shifts toward creating "ingenious stochastic models" rather than ensuring their accuracy in real-world applications.

Misleading conclusions may follow from data models that pass goodness-of-fit tests and residual checks. But published applications to data often show little care in checking model fit using these methods or any other. For instance, many of the current application articles in JASA that fit data models have very little discussion of how well their model fits the data. The question of how well the model fits the data is of secondary importance compared to the construction of an ingenious stochastic model.

### The Multiplicity of Data Models
The greatest plus of data modelling is that it presents a simple and understandable picture of the relationship between the input variables and the responses. Two different models can fit the data but paint different pictures of the nature's mechanism. They may different relations between predictor and response variables. The question of which one most accurately reflects the data is difficult to resolve. One reason for this multiplicity is that goodness-of-fit tests and other methods give a yes-or-no answer.

### Predictive Accuracy
Prediction is rarely perfect. If the model has too many parameters, then it may overfit the data and give a biased estimate of accuracy. Cross-validation can be used to remedy this, put aside a test set.


# The Limitations of Data Models

<center><b>If all a man has is a hammer, then every problem looks like a nail.</b></center>

It is assumed the data fits traditional statistical models without any verification which leads to confirmation bias, where the model is tweaked to fit this narrative.
Imposition of simple parametric models on complex systems like medical and financial data can result in loss of information and accuracy in comparison to algorithmic models.

# Algorithmic Modelling
The list of statisticians in the algorithmic modeling business is short, and applications to data are seldom seen in the journals. The development of algorithmic methods was taken up by a community outside statistics.

### Theory in Algorithmic Modelling
The approach is that nature produces data in a black box whose insides are complex, mysterious, and, at least, partly unknowable. What is observed is a set of x's that go in and a subsequent set of y's that come out. The problem is to find an algorithm f(x) such that for future x in a test set, f(x) will be a good predictor of y.
The one assumption made in the theory is that the data is drawn from an unknown multivariate distribution.

### Recent Lessons?
There have been particularly exciting developments in the last five years. What has been learned? The three lessons that seem most important to one: 
- Rashomon: the multiplicity of good models; 
- Occam: the conflict between simplicity and accuracy; 
- Bellman: dimensionality-curse or blessing

# Rashomon and the multiplicity of good models
Rashomon Effect means that there is often a multitude of different descriptions in a class of functions giving about the same minimum error rate.
This effect is closely connected to what I call instability that occurs when there are many different models crowded together that have about the same training or test set error. Then a slight perturbation of the data or in the model construction will cause a skip from one model to another. The two models are close to each other in terms of error, but can be distant in terms of the form of the model. Aggregating over a large set of competing models can reduce the nonuniqueness while improving accuracy.

# Occam and Simplicity vs. Accuracy
Occam's Razor, long admired, is usually interpreted to mean that simpler is better. Unfortunately, in prediction, accuracy and simplicity (interpretability) are in conflict. For instance, linear regression gives a fairly interpretable picture of the y, x relation. But its accuracy is usually less than that of the less interpretable neural nets.

### The Occam Dilemma
Accuracy generally requires more complex prediction methods. Simple and interpretable functions do not make the most accurate predictor. Using complex predictors may be unpleasant, but the soundest path is to go for predictive accuracy first, then try to understand why.

# Bellman and the Curse of Dimensionality
The title of this section refers to Richard Bellman's famous phrase, "the curse of dimensionality." For decades, the first step in prediction methodology was to avoid the curse. If there were too many prediction variables, the recipe was to find a few features (functions of the predictor variables) that "contain most of the information" and then use these features to replace the original variables. But recent work has shown that dimensionality can be a blessing.

### Digging it out in small pieces

# Final Remarks
The goals in statistics are to use data to predict and to get information about the underlying data mechanism. Nowhere is it written on a stone tablet what kind of model should be used to solve problems involving data. In some situations they are the most appropriate way to solve the problem. But the emphasis needs to be on the problem and on the data.
The roots of statistics, as in science, lie in working with data and checking theory against data. There are signs that this hope is not illusory. Over the last ten years, there has been a noticeable move toward statistical work on real world problems and reaching out by statisticians toward collaborative work with other disciplines.