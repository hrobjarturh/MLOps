# Animals-10

### Description (First Draft)

**Goals:**

The goal of this project is to create an Image Classification model that can classify animals. The end result should be a production ready environment containing a trained image classification model. We will use the material provided in the course.

**Data:**

We will use the dataset provided [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10). This open dataset contains 26k images of animals seperated into 10 classes. 
dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant.

**Framework:**

We will use the framework: (to-be-decided).

How we will include the framework in the project:
* The Images will be preprocessed to be compatible with the given framework
* The framework is used to build a model
* We will train the selected model on the prepared dataset, optimizing hyperparameters for optimal performance.

**Workflow:**



# Checklist

### Week 1
- [x] (Hroi) Create a git repository
- [x] (Hroi) Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages
- [x] (Hroi) Create the initial file structure using cookiecutter
- [ ] (Hroi) Write Project description
- [ ] (Jakob) Fill out the make_dataset.py file such that it downloads whatever data you need and
- [ ] (Jakob) Add a model file and a training script and get that running
- [ ] Remember to fill out the requirements.txt file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (pep8) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] (Magnus) Setup version control for your data or part of your data
- [ ] (Nael) Construct one or multiple docker files for your code
- [ ] (Nael) Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] (Nael) Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally, consider running a hyperparameter optimization sweep.
- [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── Project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
