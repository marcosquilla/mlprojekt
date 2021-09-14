Directory Structure
--------------------

    .
    ├── AUTHORS.md
    ├── README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
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
        ├── visualisation  <- scripts for visualisation during EDA, modelling, error analysis etc. 
        ├── modeling    <- scripts for generating models
    |─── requirements.txt <- file with libraries and library versions for recreating the analysis environment
   
<p><small>Project based on the <a target="_blank" href="https://github.com/jeannefukumaru/cookiecutter-ml">cookiecutter "Reproducible ML for Minimalists" template</a>. #cookiecutterdatascience</small></p>