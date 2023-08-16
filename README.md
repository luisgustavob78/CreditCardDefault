# CreditCardDefault
The goal of this project is to develop a simple machine learning app that generates score predictions for credit default analysis. The app was built using FastAPI and Heroku. The workflow is to receive json files containing data about customers and then predicts which of them is likely to default a bill payment next month. This repo aims to show and to share some concepts of model development, model deployment and CI/CD.

### Dataset
The dataset used in this project and explanations about the data can be found here: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

### Concepts
Some concepts used in this project:
* EDA;
* Feature Engineering;
* Imbalanced datasets;
* Hyperparemeter tuning;
* Model deployment and CI/CD.

### Clone
If you wanna reproduce what is presented here, just run:
```
git clone https://github.com/luisgustavob78/CreditCardDefault
```
Make sure that you have all dependencies needed here:

```
pip install requirements.txt
```

### Using the app
If you wanna try it out, select one of the json files stored in "batches" foldes and follow the gif:
![](https://github.com/luisgustavob78/CreditCardDefault/blob/main/gifs/credit_default_app.gif)

The app can be found here: https://shorturl.at/bknR0
