import sys
import os
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as rletools

from PIL import Image
from multiprocessing import Pool
from mots_common.io import load_sequences, load_seqmap
from functools import partial
from subprocess import call


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  N = 30
  brightness = 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
  colors = [colors[idx] for idx in perm]
  return colors


# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] * (1 - alpha) + alpha * color[c],
                              image[:, :, c])
  return image


def process_sequence(seq_id, tracks_folder, img_folder, output_folder, max_frames, draw_boxes=False,
                     create_video=True):
  print("Processing sequence", seq_id)
  os.makedirs(output_folder + "/" + seq_id, exist_ok=True)
  tracks = load_sequences(tracks_folder, [seq_id])[seq_id]
  max_frames_seq = max_frames[seq_id]
  visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, output_folder, draw_boxes, create_video)


def visualize_sequences(seq_id, tracks, max_frames_seq, img_folder, output_folder, draw_boxes=False, create_video=True):
  colors = generate_colors()
  dpi = 100.0
  frames_with_annotations = [frame for frame in tracks.keys() if len(tracks[frame]) > 0]
  img_sizes = next(iter(tracks[frames_with_annotations[0]])).mask["size"]
  for t in range(max_frames_seq + 1):
    print("Processing frame", t)
    filename_t = img_folder + "/" + seq_id + "/%06d" % t
    if os.path.exists(filename_t + ".png"):
      filename_t = filename_t + ".png"
    elif os.path.exists(filename_t + ".jpg"):
      filename_t = filename_t + ".jpg"
    else:
      print("Image file not found for " + filename_t + ".png/.jpg, continuing...")
      continue
    img = np.array(Image.open(filename_t), dtype="float32") / 255
    fig = plt.figure()
    fig.set_size_inches(img_sizes[1] / dpi, img_sizes[0] / dpi, forward=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.subplots()
    ax.set_axis_off()

    if t in tracks:
      for obj in tracks[t]:
        color = colors[obj.track_id % len(colors)]
        if obj.class_id == 1:
          category_name = "Car"
        elif obj.class_id == 2:
          category_name = "Pedestrian"
        else:
          category_name = "Ignore"
          color = (0.7, 0.7, 0.7)
        if obj.class_id == 1 or obj.class_id == 2:  # Don't show boxes or ids for ignore regions
          x, y, w, h = rletools.toBbox(obj.mask)
          if draw_boxes:
            import matplotlib.patches as patches
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                     edgecolor=color, facecolor='none', alpha=1.0)
            ax.add_patch(rect)
          category_name += ":" + str(obj.track_id)
          ax.annotate(category_name, (x + 0.5 * w, y + 0.5 * h), color=color, weight='bold',
                      fontsize=7, ha='center', va='center', alpha=1.0)
        binary_mask = rletools.decode(obj.mask)
        apply_mask(img, binary_mask, color)

    ax.imshow(img)
    fig.savefig(output_folder + "/" + seq_id + "/%06d" % t + ".jpg")
    plt.close(fig)
  if create_video:
    os.chdir(output_folder + "/" + seq_id)
    call(["ffmpeg", "-framerate", "10", "-y", "-i", "%06d.jpg", "-c:v", "libx264", "-profile:v", "high", "-crf", "20",
          "-pix_fmt", "yuv420p", "-vf", "pad=\'width=ceil(iw/2)*2:height=ceil(ih/2)*2\'", "output.mp4"])


def main():
  if len(sys.argv) != 5:
    print("Usage: python visualize_mots.py tracks_folder(gt or tracker results) img_folder output_folder seqmap")
    sys.exit(1)

  tracks_folder = sys.argv[1]
  img_folder = sys.argv[2]
  output_folder = sys.argv[3]
  seqmap_filename = sys.argv[4]

  seqmap, max_frames = load_seqmap(seqmap_filename)
  process_sequence_part = partial(process_sequence, max_frames=max_frames,
                                  tracks_folder=tracks_folder, img_folder=img_folder, output_folder=output_folder)

  with Pool(10) as pool:
    pool.map(process_sequence_part, seqmap)
  #for seq in seqmap:
  #  process_sequence_part(seq)


if __name__ == "__main__":
  main()
