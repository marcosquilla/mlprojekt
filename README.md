Directory Structure
--------------------

    .
    ├── AUTHORS.md
    ├── README.md
    ├── models  <- model checkpoints
    ├── config  <- any configuration files
    ├── data
    │   ├── interim <- data in intermediate processing stage
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── docs  <- usage documentation or reference papers
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src
        ├── data <- scripts for processing data eg. transformations, dataset merges etc. 
        ├── models    <- scripts for generating models, training and testing performance.
    |─── requirements.txt <- file with libraries and library versions for recreating the analysis environment
   
<p><small>Project based on the <a target="_blank" href="https://github.com/jeannefukumaru/cookiecutter-ml">cookiecutter "Reproducible ML for Minimalists" template</a>. #cookiecutterdatascience</small></p>