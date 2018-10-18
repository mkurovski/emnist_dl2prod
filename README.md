# From Exploration to Production - Bridging the Deployment Gap for Deep Learning

This repository contains the sourcecode related to my blogpost [series on Medium](https://medium.com/@marcel.kurovski/from-exploration-to-production-bridging-the-deployment-gap-for-deep-learning-8b59a5e1c819) on deep learning model exploration, translation and deployment using the [EMNIST dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset).

## Usage

1. Clone the repository and change to the folder
2. If you are using `conda`, just create a new environment from  `env.yml` with `conda env create -f env.yml`. This will create the environment `emnist_dl`.
3. Activate the environment with `conda activate emnist_dl`
3. Install the package with `python setup.py install`
4. Work through the JuPyter notebooks provided in `notebooks/`

## JuPyter Notebooks

I will provide five JuPyter notebooks for guidance through the code. I advise you to set everything up and start reading each blogpost and go trough the referenced notebooks to try things out yourself.

Part 1:

* [`1_exploration_data_EMNIST.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/1_exploration_data_EMNIST.ipynb)
* [`2_exploration_model.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/2_exploration_model.ipynb)
* [`3_translation_ONNX_GraphPipe.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/3_translation_ONNX_GraphPipe.ipynb)

Part 2:

* [`4_production_TFServing.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/4_production_TFServing.ipynb)
* [`5_production_Webserver.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/5_production_Webserver.ipynb)
* [`6_conclusion_Serving_Performance_Comparison.ipynb`](https://github.com/squall-1002/emnist_dl2prod/blob/master/notebooks/6_conclusion_Serving_Performance_Comparison.ipynb)

The code has been tested with Docker `Version 18.06.1-ce-mac73 (26764)`

## Note

This project has been set up using PyScaffold 3.1rc2. For details and usage
information on PyScaffold see https://pyscaffold.org/.