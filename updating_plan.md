# Updating plan

The plan here, is to do a complete overhaul of the rppg-toolbox. It's now going to have extreme deviations from the original rppg-toolbox. So much so that we should consider renaming the repo to remote-physiology or similar. 

The general goal is the following: have the same set of models as implemented here, not only attempt to predict the original BVP (PPG) signal, but to also be able to work for pressure traces (ABP and CVP), respiratory (RR) and others as they appear.

As far as dependencies go, we can essentially remove `requirements.txt` and `setup.sh`. `pyproject.toml` will be our source of truth. When installing new packages, consideration should be given to whether the packages exist across architecture. Primarily, we will want it to work on Windows (this machine) for dev, linux for the HPC runs, and macos (also for dev + demonstrations).

The overall repo is built on multiple stages, each will need modification.

1. Data Caching
2. Dataset and Dataloading
3. Model training
4. Evaluation
5. Config Files

## Caching 
To begin with, we have implemented a new "cache", as shown with the Neckflix dataset. This is going to be the mandatory setup for all new datasets, overriding the format as described in @./README.md.
Broadly, the tasks for updating caching will be as follows:
- Document the cache contract in README.md. Use the Neckflix cache as an example.
- For each dataset implementation, as found in `C:\Users\20759193\source\repos\rPPG-Toolbox\dataset\data_loader`, read through the description of the dataset, and then, we can delete the loaders, replacing them with markdown files which should explain what the zarr cache of that dataset should look like.
- Make a note in `CLAUDE.md` that if asked to implement a cache/loader for a new dataset, to look up the respective file in `./data_loader/`.

## Dataset and DataLoading
The dataloaders should, by default, load the whole dataset, and much like the neckflix loader, should then drop by series of filters.
The data loading should load into dictionaries, with standard structure and labels, so new neural networks can train and predict on these repeatedly.

## Model
This is a challenging task. We want to reconfigure these models to now attempt to estimate multiple signals, instead of just one. We aim to keep the main architecture of the models as close as we can to the original. When we get to each new model, I think we will need to work, collaboratively to decide how to modify the new model such that the model is generally true to form as the original, but does manage to predict multiple signals. Furthermore, we also want to have consistency in the plots that get generated across all the models.

## Evaluation
The original purpose of this repo was to estimate heart rate. So, a simple mean absolute error, bland altman analysis was fine. Now, we're adding in additional metrics, particularly ABP and CVP, where the clinically useful metrics are more than just the frequency information. We had began doing something like this in @metrics.ipynb, @neckflix_metrics.ipynb and @neckflix_metrics.py. But these are still short of what we actually want. There exists a few standards for the validation of non-invasive blood pressure monitoring, including the IEEE 1708-2014 and 1708a-2019, ISO 81060-2:2018 and 81060-3:2022, and the ESH 2023 recommendations. We should explicitly think about not only how to evaluate the model from a deep learning perspective, but to also try to establish if any of these models might fit into the clinically acceptable measurement levels.

## Config Files
The current setup for the config is too complex, and every time a new dataset or model is added, the config file keeps growing. What we'd ideally like, is to simplify/streamline the config file. It may not be possible, but it'd be nice. I also don't like the way it gets loaded in @main.py.