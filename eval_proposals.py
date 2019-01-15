#!/usr/env/bin python3
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from pycocotools.mask import encode, iou, area

GT_DIR = "/fastwork/voigtlaender/data/DAVIS2017/Annotations/480p/"
PROPOSAL_DIR = "train_log"


def eval_proposals(n_max_proposal_values, proposals_folder, gt):
    all_results = []
    n_proposals = 0
    for seq, gt_masks in gt.items():
        proposals = load_proposals(proposals_folder, seq)
        n_proposals += len(proposals)
        for idx, gt_mask in enumerate(gt_masks):
            r = eval_object(proposals, gt_mask, n_max_proposal_values)
            all_results.append(r)
    all_results = np.array(all_results)
    print(proposals_folder, "n_proposals", n_proposals, "avg per frame", n_proposals / len(gt))
    return all_results


def eval_object(proposals, gt_mask, n_max_proposal_values):
    gt_mask = encode(np.asfortranarray(gt_mask))
    segmentations = [prop["segmentation"] for prop in proposals]
    if len(proposals) == 0:
        return [0.0 for _ in n_max_proposal_values]
    assert area(gt_mask) > 0
    ious = iou(segmentations, [gt_mask], np.array([0], np.uint8))
    assert ious.shape[1] == 1
    ious = ious[:, 0]
    max_ious = [0.0 if n_max == 0 else ious[:n_max].max() for n_max in n_max_proposal_values]
    return max_ious


def load_proposals(folder, seq):
    filename = folder + "/" + seq + "/00000.json"
    try:
        with open(filename, "r") as f:
            proposals = json.load(f)
    except FileNotFoundError:
        alternate_filename = folder + "/" + seq + "/00000.jpg.json"
        with open(alternate_filename, "r") as f:
            proposals = json.load(f)
    # sort by score
    proposals.sort(key=lambda x: x["score"], reverse=True)
    return proposals


def load_gt():
    files = glob.glob(GT_DIR + "*/00000.png")
    seq_tags = [f.split("/")[-2] for f in files]
    imgs = [np.array(Image.open(f)) for f in files]
    ids = [np.unique(img) for img in imgs]
    gt = [[(img == id_).astype(np.uint8) for id_ in im_ids if id_ != 0] for im_ids, img in zip(ids, imgs)]
    gt = {tag: g for tag, g in zip(seq_tags, gt)}
    return gt


def analyze_recall(gt):
    n_max_proposal_values = [1, 10, 100, 1000]
    folders = ["fwd_original1", "fwd_agnostic1", "fwd_DAVIS2017_sharpmask"]
    for folder in folders:
        print(folder)
        ious = eval_proposals(n_max_proposal_values, PROPOSAL_DIR + "/" + folder, gt)
        mious = ious.mean(axis=0)
        for n, iou in zip(n_max_proposal_values, mious):
            print(n, iou)


def recall_curve(gt):
    n_max_proposal_values = range(1001)
    folders = ["fwd_DAVIS2017_maskrcnn",
               #"fwd_DAVIS2017_maskrcnn_tuned",
               "fwd_DAVIS2017_maskrcnn_test2",
               "fwd_DAVIS2017_maskrcnn_test3",
               "fwd_DAVIS2017_maskrcnn_agnostic",
               "fwd_DAVIS2017_maskrcnn_agnostic_tuned",
               "fwd_DAVIS2017_maskrcnn_merged",
               "fwd_DAVIS2017_maskrcnn_tuned_merged",
               "fwd_DAVIS2017_sharpmask"
               ]
    #folders += ["fwd_DAVIS2017_maskrcnn_agnostic_test" + str(idx) for idx in [9, 10, 11, 12]]

    legend_map = {"fwd_DAVIS2017_maskrcnn": "Mask R-CNN (original)",
                  #"fwd_DAVIS2017_maskrcnn_tuned": "Mask R-CNN (original + tuned)",
                  "fwd_DAVIS2017_maskrcnn_agnostic": "Mask R-CNN (category agnostic)",
                  "fwd_DAVIS2017_maskrcnn_agnostic_tuned": "Mask R-CNN (category agnostic, tuned)",
                  "fwd_DAVIS2017_maskrcnn_merged": "Mask R-CNN (original + category agnostic)",
                  "fwd_DAVIS2017_maskrcnn_tuned_merged": "Mask R-CNN (original + category agnostic, both tuned)",
                  "fwd_DAVIS2017_sharpmask": "SharpMask"}
    legend_map.update({"fwd_DAVIS2017_maskrcnn_agnostic_test" + str(idx):
                       "Mask R-CNN (category agnostic new" + str(idx) + ")" for idx in range(2, 20)})
    legend_map.update({"fwd_DAVIS2017_maskrcnn_test" + str(idx):
                        "Mask R-CNN (original new" + str(idx) + ")" for idx in range(1, 20)})
    legends = [legend_map[f] for f in folders]
    for folder in folders:
        res = eval_proposals(n_max_proposal_values, PROPOSAL_DIR + "/" + folder, gt)
        res = res.mean(axis=0)
        plt.plot(n_max_proposal_values, res)
    plt.legend(legends)
    plt.xlabel("n proposals")
    plt.ylabel("mIoU")
    plt.title("IoU of best proposal on DAVIS 2017 first frame")
    plt.show()


def analyze_difference(gt):
    n_max_proposal_values = [1, 10, 100, 1000]
    folder1 = "fwd_DAVIS2017_maskrcnn_tuned"
    folder2 = "fwd_DAVIS2017_maskrcnn_agnostic_tuned"
    folder3 = "fwd_DAVIS2017_sharpmask"
    res1 = eval_proposals(n_max_proposal_values, PROPOSAL_DIR + "/" + folder1, gt)
    res2 = eval_proposals(n_max_proposal_values, PROPOSAL_DIR + "/" + folder2, gt)
    res3 = eval_proposals(n_max_proposal_values, PROPOSAL_DIR + "/" + folder3, gt)
    for idx, n_max_proposals in enumerate(n_max_proposal_values):
        print("max proposals", n_max_proposals)
        r1 = res1[:, idx]
        r2 = res2[:, idx]
        r3 = res3[:, idx]
        print("original", r1.mean())
        print("agnostic", r2.mean())
        print("sharpmask", r3.mean())
        print("max(original, agnostic)", np.maximum(r1, r2).mean())
        print("max(original, agnostic, sharpmask)", np.maximum(np.maximum(r1, r2), r3).mean())
        plt.hist(r2 - r1, bins=20)
        #plt.figure()
        #plt.hist(r2 - r3, bins=20)
        plt.show()


def filter_gt_to_new_categories(gt):
    seqs = ["blackswan", "camel", "carousel", "chamaleon", "deer", "demolition", "dolphins", "drone", "flamingo",
            "gold-fish", "grass-chopper", "guitar-violin", "helicopter", "hoverboard", "kart-turn", "koala", "lions",
            "lock", "longboard", "mallard-fly", "mallard-water", "monkeys", "monkeys-trees", "paragliding", "pigs",
            "rhino", "seasnake", "stroller", "tractor", "turtle", "varanus-cage", "varanus-tree"]
    return {k: v for k, v in gt.items() if k in seqs}


def main():
    gt = load_gt()
    #gt = filter_gt_to_new_categories(gt)
    #analyze_recall(gt)
    #analyze_difference(gt)
    recall_curve(gt)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print("elapsed", elapsed)
