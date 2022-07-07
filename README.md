# NLP Transformers' Interpretability

The purpose of this repository is to demonstrate how to use NLP explanation/interpretability tools. In this project, I use the stance detection task, but you can change it to your own custom NLP task if you wish. This repository will be updated in the future, but for now, I just use [SHAP](https://github.com/slundberg/shap) as an explanation tool.

# Model Explanation (SHAP)

![The result of SHAP explanation on the Persian stance detection](https://github.com/Zarharan/NLP-Transformers-Interpretability/blob/main/images/FA_Shap_explanation.png)
A red area increases the probability of that class, and a blue area decreases it [(SHAP)](https://github.com/slundberg/shap).
