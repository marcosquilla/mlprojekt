Shared mobility systems provide a means of transport that is less polluting and that lowers the number of cars in cities, thus improving traffic flow and parking availability. The most widely spread of these systems is the Free-floating type. Car-sharing falls into this category. It is so due to the flexibility it provides, by allowing users to instantly rent vehicles in an area and then leave them in another one. This system is prone to unbalancing by design, as most users share the same travel patterns (travelling to work in the morning and returning in the afternoon). Currently there are two ways of balancing this system: changing the price of low demand vehicles (user-based) and manually moving them from low to high demand areas (vehicle-based).
Reinforcement learning algorithms attempt to train agents that can learn from the environment and take actions based on what they perceive. This thesis uses reinforcement learning algorithms to decide more optimally on the vehicle-based method by considering supply and demand. More specifically, offline reinforcement learning is used to prevent interaction with the environment, as letting an agent learn by trial and error would become awfully expensive.

The following algorithms were implemented in pytorch lightning: DQN, Double DQN and Conservative Q learning

Directory Structure
--------------------

    .
    ├── AUTHORS.md
    ├── README.md
    ├── models  <- model checkpoints
    ├── data
    │   ├── interim <- data in intermediate processing stage
    │   ├── processed <- data after all preprocessing has been done
    │   └── raw <- original unmodified data acting as source of truth and provenance
    ├── docs  <- documents used
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src
        ├── data <- scripts for processing data eg. transformations, dataset merges etc. 
        ├── models    <- scripts for generating models, training and testing performance.
    |─── requirements.txt <- file with libraries and library versions for recreating the analysis environment
   
<p><small>Project based on the <a target="_blank" href="https://github.com/jeannefukumaru/cookiecutter-ml">cookiecutter "Reproducible ML for Minimalists" template</a>. #cookiecutterdatascience</small></p>