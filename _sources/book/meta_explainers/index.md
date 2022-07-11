(text:meta-explainers)=
# Meta-Explainers #

We consider *{term}`meta-explainers<meta-explainer>`* to be a family of approaches that are highly modular, hence can be adopted to a number of unique use cases across a range of distinct problems, thus addressing diverse explainability needs.
These techniques support building a broad range of bespoke explainability tools within a shared algorithmic framework, offering the desired type of an explanation and accounting for its operational context.
Their properties may vary depending on their composition -- based on a selection of individual components and their instantiations -- hence the effectiveness of such methods may not transfer across different modelling problems and data sets.
Nonetheless, having the flexibility to design a tailor-made meta-explainer can offer much better transparency and robustness in comparison to using a generic composition thereof.
By creating them with a specific purpose in mind and aware of their strengths and limitations, they can become a powerful inspection tool that does not suffer from problems exhibited by their most generic variants {cite:p}`rudin2019stop,sokol2021explainability`.

````{admonition} Meta-Explainers Covered by the Book
:class: info

The following explainability approaches are included in the family of meta-explainers:
```{tableofcontents}
```
<!--
```{contents}
:local:
:depth: 2
```
-->
````

A highly popular example of a meta-explainer are *{term}`surrogates<surrogate explainer>`* {cite:p}`sokol2019blimey`.
They construct a simple model to approximate a more complex decision boundary in a desired -- local, cohort or global -- subspace {cite:p}`craven1996extracting,ribeiro2016why`.
By using different surrogate models we can generate a wide array of explanation types; for example, counterfactuals with decision trees {cite:p}`waa2018contrastive,sokol2020limetree` and feature influence with linear classifiers {cite:p}`ribeiro2016why`.
In addition to tweaking the scope and type of the explanation, their flexibility enables adjusting and customising the concepts used to express it, which in turn allows to target diverse audiences {cite:p}`sokol2020towards`.
While these explanations are derived from a separate model built on top of the explained {term}`black box` -- whose poor quality may undermine their trustworthiness {cite:p}`rudin2019stop` -- they can become reliable when designed and operationalised with care.
We will discuss common pitfalls and best practice throughout {numref}`Chapter %s <text:meta-explainers:surrogates>` to help you get the most out of surrogate explainers.
