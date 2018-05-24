# OpenSlideGenerator for Keras


## What's this?

This data generator enables training DNN (e.g., convnets) with very large images such as pathological slides with polygonal region annotation.
It generates many randomly cropped image patches from any images that [OpenSlide](http://openslide.org/api/python/) can handle.

It also provides several data augmentation features (see below).


## Features

- random cropping from polygonal regions
- multiple sampling options (based on area / label / slide)
- box-blurring with random kernel size
- H&E-stain augmentation (Tellez et al., 2018)
- scale augmentation (+-20%)

## Dependency

- Keras
- OpenCV3
- tripy
- OpenSlide (can be installed by `pip install openslide-python`)
- scikit-image

## Usage example

Usage of `OpenSlideGenerator` is very similar to that of official `ImageDataGenerator`, except for that you should prepare custom input file that contains the names of slide images, polygonal regions in them and their labels.
Please see the demo files and sample scripts for the detailed usage.

You can try training of convnet for trivial discrimination task (sand vs. sky in desert images) as follows.

```bash
# prepare input slide (tiled tif) from jpg
$ cd test_slides
$ for i in 1 2 3 4; do ./convert_to_tiled_tif.sh desert${i}.jpg desert${i}.tif; done

# run example script
$ cd ..
$ python train_example.py

```

## ToDo

- Add annotation visualization script
- Write detailed manual

