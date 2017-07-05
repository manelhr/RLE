
# *R*obust *L*ocal *E*xplainers
As the machine learning community creates increasingly complex models and pervases areas of societal impact, we are struck with the problem of understanding decisions made by machine learning models without sacrificing their accuracy.

This repo contains some experiments with the concept of a local explainer, as presented by Ribeiro et. al in 
*["Why Should I Trust You":Explainining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)*. We explore the role of locality in the results with some toy examples, attempting to help answering the question: *How can we be sure we can trust a local explainer?*

The local explainers, as implemented in *[LIME](https://github.com/marcotcr/lime)* work in the following fashion. Given a decision *d*, a global model *M* we want to explain and a local model *M'* we want to use locally, we:

- sample around a decision.

- apply the global model to classify the generated samples.

- apply the local model to the labeled samples.

- extract information out of the local model to explain the decision. 

These steps are illustrated in letters *(a-d)* in the figure bellow:

 ![Steps in a local model explainer.](https://raw.githubusercontent.com/manelhr/RLE/master/tests/imgs/_steps.png)
 
 The possible problems you can have is that (as illustrated in the figure bellow):
  
- Hard to know the how local the local model should be *(a-b)*.
 
- Multiple explanations may exist *(c-d)*.
  
 
![Steps in a local model explainer.](https://raw.githubusercontent.com/manelhr/RLE/master/tests/imgs/_models.png)
 
 
## The Stack
 
 Notice that the steps in local explanation suggest a stack:
 
- A specific decision *d*, characterized by a vector of features.
- A global model *M*, which is the model we want to explain the decisions from.
- A local model *M'*, which will be locally applied on the neighborhood of a decision.
- A sampler *S*, which will sample the neighborhood around a specific decision.
- A depicter *D*, which will present the local in a humanly understandable fashion.

In this repo we implement a logistic regression local model (*M'*), a gaussian sampler (*S*) and a bar chart depicter (*D*). We try to keep it generic. Notice that our stack only works for two labels and numerical features.

### Sampler

The *sampler* samples from a multivariate gaussian distribution *N(d,Sigma)*, where the covariance the different random variables is 0. It also produces an exponential kernel, weighting the sampled distances. The steepness of the kernel *l* indicates how local the explanation is.
 
 Sampling is as easy as:

    decision = np.array([-0.42, 0.62])

    sampler = GaussianExponentialSampler(X, ["Feature 1, Feature 2"], ["numerical", "numerical"],
                                         y, "Label", "Categorical",
                                         100, 0.05,
                                         rf.predict_proba)
    
    sample_f, sample_l, weights = sampler.sample(decision)
    
An intuition behind sampling can be seen in the image bellow. Given the data in *(a)* we train the model in *(b)* and do the sampling in *(c)*, labeling the instances with the model trained in *(b)*.

![Steps in a local model explainer.](https://raw.githubusercontent.com/manelhr/RLE/master/tests/imgs/_gaussian_exponential_sampler.png)
 
### Explainer

Our *explainer*, or in other words, the local interpretable model that we fit in the more complex machine learning one is as weighted logistic regression. 

Explaining is as easy as:

    decision = np.array([-0.42, 0.62])

    explainer = LogisticRegressionExplainer(sampler)

    explainer.explain(decision)
    
### Depicter

Our *depicter* saves or shows a simple bar chart with the weights of each one of the features. It can be tuned to plot on an axis/destination.
