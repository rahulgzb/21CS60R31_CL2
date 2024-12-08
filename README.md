# G-CNN: Nucleus Segmentation and classification of matastetic tissue 

The repository can be used for training G-CNN network for classification and segmentation task and to process image tiles or whole-slide images. As part of this repository:

- [Kumar](https://ieeexplore.ieee.org/abstract/document/7872382)
- [PCAM] (https://paperswithcode.com/dataset/pcam)


Links to the checkpoints can be found in the inference description below.

## Set Up Environment

```
pip install torch==1.6.0 torchvision==0.7.0
```

Above, we install PyTorch version 1.6 with CUDA 10.2. 

## Repository Structure

Below are the main directories in the repository: 

- `Nucleus_segmentation/dataloader/`: the data loader and augmentation pipeline for kumar dataset
- `Nucleus_segmentation/metrics/`: scripts for metric calculation
- `Nucleus_segmentation/misc/`: utils that are
- `Nucleus_segmentation/models/`: model definition of G-CNN , along with the main run step and hyperparameter settings  
- `Nucleus_segmentation/run_utils/`: defines the train/validation loop and callbacks 

Below are the main executable scripts in the repository:

- `Nucleus_segmentation/config.py`: configuration file
- `Nucleus_segmentation/dataset.py`: defines the dataset classes 
- `Nucleus_segmentation/extract_patches.py`: extracts patches from original images
- `Nucleus_segmentation/compute_stats.py`: main metric computation script
- `Nucleus_segmentation/run_train.py`: main training script
- `Nucleus_segmentation/run_infer.py`: main inference script for tile and WSI processing
- `Nucleus_segmentation/convert_chkpt_tf2pytorch`: convert tensorflow `.npz` model trained in original repository to pytorch supported `.tar` format.

# Running the Code

## Training

### Data Format
For training, patches must be extracted using `extract_patches.py`. For instance segmentation, patches are stored as a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. 

For simultaneous instance segmentation and classification, patches are stored as a 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

Before training:

- Set path to the data directories in `config.py`
- Set path where checkpoints will be saved  in `config.py`


### Usage and Options
 
Usage: <br />
```
  python Nucleus_segmentation/run_train.py [--gpu=<id>] [--view=<dset>]
  python Nucleus_segmentation/run_train.py (-h | --help)
  python Nucleus_segmentation/run_train.py --version
```

Options:
```
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list.  
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
```

Examples:

To visualise the training dataset as a sanity check before training use:
```
python Nucleus_segmentation/run_train.py --view='train'
```

To initialise the training script with GPUs 0 and 1, the command is:
```
python Nucleus_segmentation/run_train.py --gpu='0,1' 
```

## Inference

### Data Format
Input: <br />
- Standard images files, including `png`, `jpg` and `tiff`.
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- Both image tiles and whole-slide images output a `json` file with keys:
    - 'bbox': bounding box coordinates for each nucleus
    - 'centroid': centroid coordinates for each nucleus
    - 'contour': contour coordinates for each nucleus 
    - 'type_prob': per class probabilities for each nucleus (default configuration doesn't output this)
    - 'type': prediction of category for each nucleus
- Image tiles output a `mat` file, with keys:
    - 'raw': raw output of network (default configuration doesn't output this)
    - 'inst_map': instance map containing values from 0 to N, where N is the number of nuclei
    - 'inst_type': list of length N containing predictions for each nucleus
 - Image tiles output a `png` overlay of nuclear boundaries on top of original RGB image
  


### Usage and Options

Usage: <br />
```
  Nucleus_segmentation/run_infer.py [options] [--help] <command> [<args>...]
  Nucleus_segmentation/run_infer.py --version
  Nucleus_segmentation/run_infer.py (-h | --help)
```

Options:
```
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlay color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used in PanNuke / MoNuSAC, 'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
```

Tile Processing Options: <br />
```
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
```

WSI Processing Options: <br />
```
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
```

The above command can be used from the command line or via an executable script. We supply two example executable scripts: one for tile processing and one for WSI processing. To run the scripts, first make them executable by using `chmod +x Nucleus_segmentation/run_tile.sh` and `chmod +x Nucleus_segmentation/run_tile.sh`.
Intermediate results are stored in cache. Therefore ensure that the specified cache location has enough space! Preferably ensure that the cache location is SSD.

Note, it is important to select the correct model mode when running inference. 'original' model mode refers to the method described in the original medical image analysis paper with a 270x270 patch input and 80x80 patch output. 'fast' model mode uses a 256x256 patch input and 164x164 patch output. Model checkpoints trained on Kumar, CPM17 and CoNSeP are from our original publication and therefore the 'original' mode **must** be used. For PanNuke and MoNuSAC, the 'fast' mode **must** be selected. The model mode for each checkpoint that we provide is given in the filename. Also, if using a model trained only for segmentation, `nr_types` must be set to 0.

`type_info.json` is used to specify what RGB colours are used in the overlay. Make sure to modify this for different datasets and if you would like to generally control overlay boundary colours.

As part of our tile processing implementation, we add an option to save the output in a form compatible with QuPath. 



