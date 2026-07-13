import os
import yaml

def load_cfg(path):
    assert os.path.isfile(path), path
    cfg = yaml.safe_load(open(path, "r"))

    # model.global is the single source of truth: fan its keys out to every
    # model sub-dict, the dataset, and the train section (so n_bins/min_date/
    # max_date can't drift between them).
    for key, val in cfg["model"]["global"].items():
        for name in cfg["model"].keys():
            if name == "global":
                continue
            cfg["model"][name][key] = val
        cfg["dataset"][key] = val
        cfg["train"][key] = val
    return cfg
