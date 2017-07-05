
# *R*obust *L*ocal *E*xplainers

This repo contains some experiments with the concept of a local explainer, as presented by Ribeiro et. al in 
*["Why Should I Trust You":Explainining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)*. We explore the role of locality in the results with some toy examples.
 
 # Problem Intuition
 
As the machine learning community creates increasingly complex models and pervases areas of societal impact, we are struck with the problem of understanding decisions made by machine learning models without sacrificing their accuracy.

This understandability is important for the creation of models that are reliable, fair and transparent. 
A method to locally understand models that are hard to interpret is to apply an explainable model to regions of the complex one. 
We call this local model an *explainer* However, this leaves us with another problem: *how can we be sure we can trust the explainer?*
