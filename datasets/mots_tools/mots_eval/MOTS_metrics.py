import math
from collections import defaultdict
import pycocotools.mask as rletools
from mots_common.io import SegmentedObject


class MOTSResults:
  def __init__(self):
    self.n_gt_trajectories = 0
    self.n_tr_trajectories = 0
    self.total_num_frames = 0

    # Evaluation metrics
    self.n_gt = 0  # number of ground truth detections
    self.n_tr = 0  # number of tracker detections minus ignored tracker detections
    self.n_itr = 0  # number of ignored tracker detections
    self.tp = 0  # number of true positives
    self.fp = 0  # number of false positives
    self.fn = 0  # number of false negatives
    self.MOTSA = 0
    self.sMOTSA = 0
    self.MOTSP = 0
    self.MOTSAL = 0
    self.MODSA = 0
    self.MODSP = 0
    self.recall = 0
    self.precision = 0
    self.F1 = 0
    self.FAR = 0
    self.total_cost = 0
    self.fragments = 0
    self.id_switches = 0
    self.MT = 0
    self.PT = 0
    self.ML = 0


# go through all frames and associate ground truth and tracker results
def compute_MOTS_metrics_per_sequence(seq_name, gt_seq, results_seq, max_frames, class_id,
                                      ignore_class, overlap_function):
  results_obj = MOTSResults()
  results_obj.total_num_frames = max_frames + 1
  seq_trajectories = defaultdict(list)

  # To count number of track ids
  gt_track_ids = set()
  tr_track_ids = set()

  # Statistics over the current sequence
  seqtp = 0
  seqfn = 0
  seqfp = 0
  seqitr = 0

  n_gts = 0
  n_trs = 0

  # Iterate over frames in this sequence
  for f in range(max_frames + 1):
    g = []
    dc = []
    t = []

    if f in gt_seq:
      for obj in gt_seq[f]:
        if obj.class_id == ignore_class:
          dc.append(obj)
        elif obj.class_id == class_id:
          g.append(obj)
          gt_track_ids.add(obj.track_id)
    if f in results_seq:
      for obj in results_seq[f]:
        if obj.class_id == class_id:
          t.append(obj)
          tr_track_ids.add(obj.track_id)

    # Handle ignore regions as one large ignore region
    dc = SegmentedObject(mask=rletools.merge([d.mask for d in dc], intersect=False),
                         class_id=ignore_class, track_id=ignore_class)

    tracks_valid = [False for _ in range(len(t))]

    # counting total number of ground truth and tracker objects
    results_obj.n_gt += len(g)
    results_obj.n_tr += len(t)

    n_gts += len(g)
    n_trs += len(t)

    # tmp variables for sanity checks and MODSP computation
    tmptp = 0
    tmpfp = 0
    tmpfn = 0
    tmpc = 0  # this will sum up the overlaps for all true positives
    tmpcs = [0] * len(g)  # this will save the overlaps for all true positives
    # the reason is that some true positives might be ignored
    # later such that the corrsponding overlaps can
    # be subtracted from tmpc for MODSP computation

    # To associate, simply take for each ground truth the (unique!) detection with IoU>0.5 if it exists

    # all ground truth trajectories are initially not associated
    # extend groundtruth trajectories lists (merge lists)
    for gg in g:
      seq_trajectories[gg.track_id].append(-1)
    num_associations = 0
    for row, gg in enumerate(g):
      for col, tt in enumerate(t):
        c = overlap_function(gg, tt)
        if c > 0.5:
          tracks_valid[col] = True
          results_obj.total_cost += c
          tmpc += c
          tmpcs[row] = c
          seq_trajectories[g[row].track_id][-1] = t[col].track_id

          # true positives are only valid associations
          results_obj.tp += 1
          tmptp += 1

          num_associations += 1

    # associate tracker and DontCare areas
    # ignore tracker in neighboring classes
    nignoredtracker = 0  # number of ignored tracker detections

    for i, tt in enumerate(t):
      overlap = overlap_function(tt, dc, "a")
      if overlap > 0.5 and not tracks_valid[i]:
        nignoredtracker += 1

    # count the number of ignored tracker objects
    results_obj.n_itr += nignoredtracker

    # false negatives = non-associated gt instances
    #
    tmpfn += len(g) - num_associations
    results_obj.fn += len(g) - num_associations

    # false positives = tracker instances - associated tracker instances
    # mismatches (mme_t)
    tmpfp += len(t) - tmptp - nignoredtracker
    results_obj.fp += len(t) - tmptp - nignoredtracker
    # tmpfp   = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
    # self.fp += len(t) - tmptp - nignoredtp

    # update sequence data
    seqtp += tmptp
    seqfp += tmpfp
    seqfn += tmpfn
    seqitr += nignoredtracker

    # sanity checks
    # - the number of true positives minues ignored true positives
    #   should be greater or equal to 0
    # - the number of false negatives should be greater or equal to 0
    # - the number of false positives needs to be greater or equal to 0
    #   otherwise ignored detections might be counted double
    # - the number of counted true positives (plus ignored ones)
    #   and the number of counted false negatives (plus ignored ones)
    #   should match the total number of ground truth objects
    # - the number of counted true positives (plus ignored ones)
    #   and the number of counted false positives
    #   plus the number of ignored tracker detections should
    #   match the total number of tracker detections
    if tmptp < 0:
      print(tmptp)
      raise NameError("Something went wrong! TP is negative")
    if tmpfn < 0:
      print(tmpfn, len(g), num_associations)
      raise NameError("Something went wrong! FN is negative")
    if tmpfp < 0:
      print(tmpfp, len(t), tmptp, nignoredtracker)
      raise NameError("Something went wrong! FP is negative")
    if tmptp + tmpfn != len(g):
      print("seqname", seq_name)
      print("frame ", f)
      print("TP    ", tmptp)
      print("FN    ", tmpfn)
      print("FP    ", tmpfp)
      print("nGT   ", len(g))
      print("nAss  ", num_associations)
      raise NameError("Something went wrong! nGroundtruth is not TP+FN")
    if tmptp + tmpfp + nignoredtracker != len(t):
      print(seq_name, f, len(t), tmptp, tmpfp)
      print(num_associations)
      raise NameError("Something went wrong! nTracker is not TP+FP")

    # compute MODSP
    MODSP_f = 1
    if tmptp != 0:
      MODSP_f = tmpc / float(tmptp)
    results_obj.MODSP += MODSP_f

  assert len(seq_trajectories) == len(gt_track_ids)
  results_obj.n_gt_trajectories = len(gt_track_ids)
  results_obj.n_tr_trajectories = len(tr_track_ids)

  # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
  if len(seq_trajectories) != 0:
    for g in seq_trajectories.values():
      # all frames of this gt trajectory are not assigned to any detections
      if all([this == -1 for this in g]):
        results_obj.ML += 1
        continue
      # compute tracked frames in trajectory
      last_id = g[0]
      # first detection (necessary to be in gt_trajectories) is always tracked
      tracked = 1 if g[0] >= 0 else 0
      for f in range(1, len(g)):
        if last_id != g[f] and last_id != -1 and g[f] != -1:
          results_obj.id_switches += 1
        if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
          results_obj.fragments += 1
        if g[f] != -1:
          tracked += 1
          last_id = g[f]
      # handle last frame; tracked state is handled in for loop (g[f]!=-1)
      if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
        results_obj.fragments += 1

      # compute MT/PT/ML
      tracking_ratio = tracked / float(len(g))
      if tracking_ratio > 0.8:
        results_obj.MT += 1
      elif tracking_ratio < 0.2:
        results_obj.ML += 1
      else:  # 0.2 <= tracking_ratio <= 0.8
        results_obj.PT += 1

  return results_obj


def compute_MOTS_metrics(gt, results, max_frames, class_id, ignore_class, overlap_function):
  """
      Like KITTI tracking eval but with simplified association (when we assume non overlapping masks)
  """
  results_per_seq = {}
  for seq in gt.keys():
    results_seq = {}
    if seq in results:
      results_seq = results[seq]
    results_per_seq[seq] = compute_MOTS_metrics_per_sequence(seq, gt[seq], results_seq, max_frames[seq], class_id,
                                                             ignore_class, overlap_function)

  # Sum up results for all sequences
  results_for_all_seqs = MOTSResults()
  mots_results_attributes = [a for a in dir(results_for_all_seqs) if not a.startswith('__')]
  for attr in mots_results_attributes:
    results_for_all_seqs.__dict__[attr] = sum(obj.__dict__[attr] for obj in results_per_seq.values())

  # Compute aggregate metrics
  for res in results_per_seq.values():
    compute_prec_rec_clearmot(res)
  compute_prec_rec_clearmot(results_for_all_seqs)

  print_summary(list(gt.keys()), results_per_seq, results_for_all_seqs)

  return results_per_seq, results_for_all_seqs


def compute_prec_rec_clearmot(results_obj):
  # precision/recall etc.
  if (results_obj.fp + results_obj.tp) == 0 or (results_obj.tp + results_obj.fn) == 0:
    results_obj.recall = 0.
    results_obj.precision = 0.
  else:
    results_obj.recall = results_obj.tp / float(results_obj.tp + results_obj.fn)
    results_obj.precision = results_obj.tp / float(results_obj.fp + results_obj.tp)
  if (results_obj.recall + results_obj.precision) == 0:
    results_obj.F1 = 0.
  else:
    results_obj.F1 = 2. * (results_obj.precision * results_obj.recall) / (results_obj.precision + results_obj.recall)
  if results_obj.total_num_frames == 0:
    results_obj.FAR = "n/a"
  else:
    results_obj.FAR = results_obj.fp / float(results_obj.total_num_frames)
  # compute CLEARMOT
  if results_obj.n_gt == 0:
    results_obj.MOTSA = -float("inf")
    results_obj.MODSA = -float("inf")
    results_obj.sMOTSA = -float("inf")
  else:
    results_obj.MOTSA = 1 - (results_obj.fn + results_obj.fp + results_obj.id_switches) / float(results_obj.n_gt)
    results_obj.MODSA = 1 - (results_obj.fn + results_obj.fp) / float(results_obj.n_gt)
    results_obj.sMOTSA = (results_obj.total_cost - results_obj.fp - results_obj.id_switches) / float(results_obj.n_gt)
  if results_obj.tp == 0:
    results_obj.MOTSP = float("inf")
  else:
    results_obj.MOTSP = results_obj.total_cost / float(results_obj.tp)
  if results_obj.n_gt != 0:
    if results_obj.id_switches == 0:
      results_obj.MOTSAL = 1 - (results_obj.fn + results_obj.fp + results_obj.id_switches) / float(results_obj.n_gt)
    else:
      results_obj.MOTSAL = 1 - (results_obj.fn + results_obj.fp + math.log10(results_obj.id_switches)) / float(
        results_obj.n_gt)
  else:
    results_obj.MOTSAL = -float("inf")

  if results_obj.total_num_frames == 0:
    results_obj.MODSP = "n/a"
  else:
    results_obj.MODSP = results_obj.MODSP / float(results_obj.total_num_frames)

  if results_obj.n_gt_trajectories == 0:
    results_obj.MT = 0.
    results_obj.PT = 0.
    results_obj.ML = 0.
  else:
    results_obj.MT /= float(results_obj.n_gt_trajectories)
    results_obj.PT /= float(results_obj.n_gt_trajectories)
    results_obj.ML /= float(results_obj.n_gt_trajectories)

  return results_obj


def print_summary(seq_names, results_per_seq, results_for_all_seqs, column_width=14):
  metrics = [("sMOTSA", "sMOTSA"), ("MOTSA", "MOTSA"),
             ("MOTSP", "MOTSP"), ("MOTSAL", "MOTSAL"), ("MODSA", "MODSA"), ("MODSP", "MODSP"),
             ("Recall", "recall"), ("Prec", "precision"), ("F1", "F1"), ("FAR", "FAR"),
             ("MT", "MT"), ("PT", "PT"), ("ML", "ML"),
             ("TP", "tp"), ("FP", "fp"), ("FN", "fn"),
             ("IDS", "id_switches"), ("Frag", "fragments"),
             ("GT Obj", "n_gt"), ("GT Trk", "n_gt_trajectories"),
             ("TR Obj", "n_tr"), ("TR Trk", "n_tr_trajectories"), ("Ig TR Tck", "n_itr")]
  metrics_names = [tup[0] for tup in metrics]
  metrics_keys = [tup[1] for tup in metrics]
  row_format = "{:>4}" + "".join([("{:>" + str(max(len(name), 4)+2) + "}") for name in metrics_names])
  print(row_format.format("", *metrics_names))

  def format_results_entries(results_obj):
    res = []
    for key in metrics_keys:
      entry = results_obj.__dict__[key]
      if isinstance(entry, float):
        res.append("%.1f" % (entry * 100.0))
      else:
        res.append(str(entry))
    return res

  all_results = format_results_entries(results_for_all_seqs)
  print(row_format.format("all", *all_results))
  for seq in seq_names:
    all_results = format_results_entries(results_per_seq[seq])
    print(row_format.format(seq, *all_results))


def create_summary_KITTI_style(results_obj):
  summary = ""

  summary += "tracking evaluation summary".center(80, "=") + "\n"
  summary += print_entry("Multiple Object Tracking Segmentation Accuracy (sMOTSA)", results_obj.sMOTSA) + "\n"
  summary += print_entry("Multiple Object Tracking Accuracy (MOTSA)", results_obj.MOTSA) + "\n"
  summary += print_entry("Multiple Object Tracking Precision (MOTSP)", results_obj.MOTSP) + "\n"
  summary += print_entry("Multiple Object Tracking Accuracy (MOTSAL)", results_obj.MOTSAL) + "\n"
  summary += print_entry("Multiple Object Detection Accuracy (MODSA)", results_obj.MODSA) + "\n"
  summary += print_entry("Multiple Object Detection Precision (MODSP)", results_obj.MODSP) + "\n"
  summary += "\n"
  summary += print_entry("Recall", results_obj.recall) + "\n"
  summary += print_entry("Precision", results_obj.precision) + "\n"
  summary += print_entry("F1", results_obj.F1) + "\n"
  summary += print_entry("False Alarm Rate", results_obj.FAR) + "\n"
  summary += "\n"
  summary += print_entry("Mostly Tracked", results_obj.MT) + "\n"
  summary += print_entry("Partly Tracked", results_obj.PT) + "\n"
  summary += print_entry("Mostly Lost", results_obj.ML) + "\n"
  summary += "\n"
  summary += print_entry("True Positives", results_obj.tp) + "\n"
  summary += print_entry("False Positives", results_obj.fp) + "\n"
  summary += print_entry("False Negatives", results_obj.fn) + "\n"
  summary += print_entry("Missed Targets", results_obj.fn) + "\n"
  summary += print_entry("ID-switches", results_obj.id_switches) + "\n"
  summary += print_entry("Fragmentations", results_obj.fragments) + "\n"
  summary += "\n"
  summary += print_entry("Ground Truth Objects (Total)", results_obj.n_gt) + "\n"
  summary += print_entry("Ground Truth Trajectories", results_obj.n_gt_trajectories) + "\n"
  summary += "\n"
  summary += print_entry("Tracker Objects (Total)", results_obj.n_tr) + "\n"
  summary += print_entry("Ignored Tracker Objects", results_obj.n_itr) + "\n"
  summary += print_entry("Tracker Trajectories", results_obj.n_tr_trajectories) + "\n"
  summary += "=" * 80

  return summary


def print_entry(key, val, width=(70, 10)):
  s_out = key.ljust(width[0])
  if type(val) == int:
    s = "%%%dd" % width[1]
    s_out += s % val
  elif type(val) == float:
    s = "%%%df" % (width[1])
    s_out += s % val
  else:
    s_out += ("%s" % val).rjust(width[1])
  return s_out
