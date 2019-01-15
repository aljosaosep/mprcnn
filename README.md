# THIS README IS STILL BEING WRITTEN. STAY TUNED.
# Mask Proposal R-CNN (for 4D Generic Video Object Proposals)

This repository contains code for the Mask Proposal R-CNN network, as described in

**4D Generic Video Object Proposals** (Under review, coming to arxiv soon)

By [Aljosa Osep](https://www.vision.rwth-aachen.de/person/13/), [Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/), [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/), Mark Weber, [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), Computer Vision Group, RWTH Aachen University

## Pretrained Models
* Get them here (TODO link)

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):
* Python 3.5.5
* Tensorflow 1.9.0 (note: won't work with CPU version)
* Additional dependencies (get using conda install / pip install / ...): `termcolor, tqdm, tabulate, opencv-python, msgpack_numpy, scipy, pillow`
* [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) (need to add to PYTHONPATH)

## Install
* Clone tensorpack `git clone https://github.com/ppwwyyxx/tensorpack /home/${USER}/vision/tensorpack/`
* To make sure your tensorpack is compatible, better checkout the commit on which we base this implementation:
  * `cd /home/${USER}/vision/tensorpack/`
  * `git checkout 6fdde15deac4f23eb31ffef9e8d405d9153a51cd`
  
* Add tensorpack and pycocotools to your PTHONPATH: 
  * `export PYTHONPATH=$PYTHONPATH:/home/${USER}/vision/tensorpack/:/home/${USER}/vision/coco3/PythonAPI/`
  
* Create a folder on your local disk on which models and logs will be stored and link it here under the name train_log:
  * `mkdir -p /work/${USER}/data/tensorpack_models` (or wherever your models are)
  * `ln -s /work/${USER}/data/tensorpack_models/ train_log`
  
## Run
* To use the pre-trained MP R-CNN model trained on COCO dataset, simply run the provided `predict_test.sh`; it will forward all images in `$REPO\sample_images` and store segmentations to `/tmp/proposals_test` (jsons).

## Train
* TODO

## Citing

If you find this repo useful in your research, please cite:

    @article{Osep19arxiv,
      author = {O\v{s}ep, Aljo\v{s}a and Voigtlaender, Paul and Luiten, Jonathon and Weber, Mark and Leibe, Bastian},
      title = {4D Generic Video Object Proposals},
      journal = {arXiv preprint arXiv:TBA},
      year = {2019}
    }
    
## Potential Issues
* TODO

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2017 Aljosa Osep, Paul Voigtlaender, Jonathon Luiten
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
