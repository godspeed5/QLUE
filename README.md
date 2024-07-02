# QLUE

This is the repository for the qLUE clustering algorithm inspired by CERN's CLUE event reconstruction algorithm. The associated paper can be found at [text](https://arxiv.org/abs/2407.00357)

The code for the noise analysis in Fig 5(a-c) is in ``main_noise.py`` and can be run using ``python main_noise.py --cq q``

The code for the overlap analysis in Fig 5(d-f) is in ``main.py`` and can be run using ``python main.py --cq q``

The datasets for Fig 5(g-j) are based on SKLearn's moons and circles datasets and can be generated using the code in ``Viz.ipynb``. qLUE can be run on this using ``python main_general_dataset.py``. As the name suggests, this is the code to run qLUE on a general dataset.

Several example datasets are provided in the ``datasets/`` folder
