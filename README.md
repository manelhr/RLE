
# *R*obust *L*ocal *E*xplainers
As the machine learning community creates increasingly complex models and pervases areas of societal impact, we are struck with the problem of understanding decisions made by machine learning models without sacrificing their accuracy.

This repo contains some experiments with the concept of a local explainer, as presented by Ribeiro et. al in 
*["Why Should I Trust You":Explainining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)*. We explore the role of locality in the results with some toy examples, attempting to help answering the question: *How can we be sure we can trust a local explainer?*

The local explainers, as implemented in *[LIME](https://github.com/marcotcr/lime)* work in the following fashion. Given a decision *d*, a global model *G*" we want to explain and a local model *G'* we want to use locally, we:

- sample around a decision.

- apply the global model to classify the generated samples.

- apply the local model to the labeled samples.

- extract information out of the local model to explain the decision. 

These steps are illustrated in letters (a-d) in the figure bellow:

 ![Steps in a local model explainer.](https://raw.githubusercontent.com/manelhr/RLE/master/tests/imgs/steps.png)