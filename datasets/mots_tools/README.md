# mots_tools
Tools for evaluating and visualizing results for the Multi Object Tracking and Segmentation (MOTS) task.

For the TrackR-CNN code please visit https://github.com/VisualComputingInstitute/TrackR-CNN

## Project website (including annotations)
https://www.vision.rwth-aachen.de/page/mots

## Paper
https://www.vision.rwth-aachen.de/media/papers/mots-multi-object-tracking-and-segmentation/MOTS.pdf

## Using the mots_tools
Please install the cocotools (https://github.com/cocodataset/cocoapi), which we use with run-length encoded binary masks. If you want to visualize your results using this script, please also install FFmpeg.

In order to evaluate or visualize the results of your MOTS method, please export them in one of the two formats we use for the ground truth annotations: png or txt (see https://www.vision.rwth-aachen.de/page/mots). When using png, we expect the result images to be in subfolders corresponding to the sequences (e.g. tracking_results/0002/000000.png, tracking_results/0002/000001.png, ...). When using txt, we expect filenames corresponding to the sequences (e.g. tracking_results/0002.txt, tracking_results/0006.txt, ...).

### Evaluating a tracking result
Clone this repository, navigate to the mots_tools directory and make sure it is in your Python path. 
Now suppose your tracking results are located in a folder "tracking_results". Suppose further the ground truth annotations are located in a folder "gt_folder". Then you can evaluate your results using the commands
```
python mots_eval/eval.py tracking_results gt_folder seqmap
```
where "seqmap" is a textfile containing the sequences which you want to evaluate on. Several seqmaps are already provided in the mots_eval repository: val.seqmap, train.seqmap, fulltrain.seqmap, val_MOTSchallenge.seqmap which correspond to the KITTI MOTS validation set, the KITTI MOTS training set, both KITTI MOTS sets combined and the four annotated MOTSChallenge sequences respectively.

Parts of the evaluation logic are built upon the KITTI 2D tracking evaluation devkit from http://www.cvlibs.net/datasets/kitti/eval_tracking.php

### Visualizing a tracking result
Similarly to evaluating tracking results, you can also create visualizations using
```
python mots_eval/visualize_mots.py tracking_results img_folder output_folder seqmap
```
where "img_folder" is a folder containing the original KITTI tracking images (http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and "output_folder" is a folder where the resulting visualization will be created.
## Citation
If you use this code, please cite:
```
@inproceedings{Voigtlaender19CVPR_MOTS,
 author = {Paul Voigtlaender and Michael Krause and Aljo\u{s}a O\u{s}ep and Jonathon Luiten and Berin Balachandar Gnana Sekar and Andreas Geiger and Bastian Leibe},
 title = {{MOTS}: Multi-Object Tracking and Segmentation},
 booktitle = {CVPR},
 year = {2019},
}
```

## License
MIT License

## Contact
If you find a problem in the code, please open an issue.

For general questions, please contact Paul Voigtlaender (voigtlaender@vision.rwth-aachen.de) or Michael Krause (michael.krause@rwth-aachen.de)
