# Simple Black Box Adversarial Perturbations for Deep Networks
Implementation of 'Simple Black-Box Adversarial Perturbations for Deep Networks' https://openreview.net/pdf?id=SJCscQcge in Keras


### Running it

* `python cifar100.py` to train a basic CNN for cifar100 and save that file.
* `python find_good.py <model>` to go through cifar100 test dataset and find a good image (as defined in the paper).
* `python main.py <KERAS_MODEL> <IMAGE_in_NUMPY>` : currently works for cifar images only. 
