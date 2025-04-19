# uc-no-response

[![Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
![MLflow](https://img.shields.io/badge/MLflow-active-blue)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## Architecture Overview

```mermaid
graph TD
    A[Raw Data CSV/Parquet] --> B(Data Preprocessing)
    B --> C[Processed Data]
    
    subgraph Kedro Pipeline
        B -->|"1. fill_null_values()"| C
        B -->|"2. create_dummy_vars()"| C
        B -->|"3. normalize()"| C
        C --> D{Model Training}
    end
    
    subgraph Model Training
        D -->|XGBoost| E[Hyperparameter Tuning<br/>RandomizedSearchCV]
        E -->|Best Model| F[Evaluation]
        F --> G[Metrics: Recall, F2, Cost]
        F --> H[Confusion Matrix]
    end
    
    subgraph MLflow Tracking
        G --> I[(MLflow Server)]
        H --> I
        E -->|"Log params/metrics"| I
    end
    
    subgraph Output Artifacts
        I --> J[Model Registry<br/>Pickle/MLflow]
        I --> K[Performance Reports<br/>JSON/PNG]
    end
    
    style A fill:#F9E79F,stroke:#F1C40F
    style B fill:#AED6F1,stroke:#3498DB
    style D fill:#A2D9CE,stroke:#16A085
    style E fill:#F5B7B1,stroke:#E74C3C
    style I fill:#D2B4DE,stroke:#9B59B6
```

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
