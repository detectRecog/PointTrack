from models.BranchedERFNet import *


def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    if name == "tracker_offset_emb":
        model = TrackerOffsetEmb(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))