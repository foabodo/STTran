import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
import yaml

# from dataloader.action_genome import ActionGenome as Dataset, cuda_collate_fn
from dataloader.kitchen_genome import KitchenGenome as Dataset, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

with open(conf.config_path) as config_file:
    yaml_config = yaml.load(config_file, yaml.SafeLoader)

dataset_train = Dataset(
    mode="train", 
    datasize=conf.datasize, 
    data_path=conf.data_path, 
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == 'predcls' else True,
    config_path=conf.config_path
)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train, 
    shuffle=True, 
    num_workers=2,
    collate_fn=cuda_collate_fn,
    pin_memory=False)

dataset_test = Dataset(
    mode="test",
    datasize=conf.datasize,
    data_path=conf.data_path,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == 'predcls' else True,
    config_path=conf.config_path)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    shuffle=False,
    num_workers=2,
    collate_fn=cuda_collate_fn,
    pin_memory=False)

cpu_device = torch.device("cpu")
sttran_device = torch.device("cuda:1")
object_detector_device = torch.device("cuda:0")
# freeze the detection backbone

# Prepare a partial mapping
checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
ignorable_keys = [
    'RCNN_cls_score.weight',
    'RCNN_cls_score.bias',
    'RCNN_bbox_pred.weight',
    'RCNN_bbox_pred.bias'
]
state_dict = {
    k: v for k, v in checkpoint['model'].items() if k not in ignorable_keys
}

object_detector = detector(
    train=True,
    object_classes=dataset_train.object_classes,
    use_SUPPLY=True,
    mode=conf.mode,
    state_dict=state_dict,
    ignore_missing_keys=True,
    device=object_detector_device,
    batch_size=yaml_config["batch_size"]
).to(device=object_detector_device)

object_detector.eval()

model = STTran(
    mode=conf.mode,
    source_class_num=len(dataset_train.source_relationships),
    target_class_num=len(dataset_train.target_relationships),
    obj_classes=dataset_train.object_classes,
    enc_layer_num=conf.enc_layer,
    dec_layer_num=conf.dec_layer,
    word_vec_dir=yaml_config["word_vec_dir"],
    word_vec_dim=yaml_config["word_vec_dim"],
    device=sttran_device
).to(device=sttran_device)

evaluator = BasicSceneGraphEvaluator(
    mode=conf.mode,
    object_classes=dataset_train.object_classes,
    all_predicates=dataset_train.relationship_classes,
    source_predicates=dataset_train.source_relationships,
    target_predicates=dataset_train.target_relationships,
    iou_threshold=yaml_config["iou_threshold"],
    constraint=yaml_config["constraint"]
)

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []

for epoch in range(int(conf.nepoch)):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    
    for b in range(len(dataloader_train)):
        print(f"Fetching train data {b}")
        data = next(train_iter)

        gt_annotation = dataset_train.gt_annotations[data[4]]
        num_gt_annotations = sum([len(anno) for anno in gt_annotation])

        print(f"GT_ANNOTATION_LEN: {num_gt_annotations}")
        print(f"GT_ANNOTATION_LEN: {len(gt_annotation)}")

        # # we can't fit too many bboxes in GPU ram at the same time
        # if num_gt_annotations >= 3625:
        #     continue

        im_data = copy.deepcopy(data[0]).to(object_detector_device)
        im_info = copy.deepcopy(data[1]).to(object_detector_device)
        gt_boxes = copy.deepcopy(data[2]).to(object_detector_device)
        num_boxes = copy.deepcopy(data[3]).to(object_detector_device)

        print(f"im_data.shape: {im_data.size()}")
        print(f"im_info.shape: {im_info.size()}")
        print(f"gt_boxes.shape: {gt_boxes.size()}")
        print(f"num_boxes.shape: {num_boxes.size()}")

        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        entry = {k: v.to(sttran_device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}

        # # Try to avoid GPU OOM
        # im_data.to(cpu_device)
        # im_info.to(cpu_device)
        # gt_boxes.to(cpu_device)
        # num_boxes.to(cpu_device)
        #
        # del im_data
        # del im_info
        # del gt_boxes
        # del num_boxes

        pred = model(entry)

        source_distribution = pred["source_distribution"]
        target_distribution = pred["target_distribution"]

        # attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            source_label = -torch.ones([len(pred["source_gt"]), dataset_train.num_source_relationships], dtype=torch.long).to(device=source_distribution.device)
            target_label = -torch.ones([len(pred["target_gt"]), dataset_train.num_target_relationships], dtype=torch.long).to(device=source_distribution.device)
            for i in range(len(pred["source_gt"])):
                source_label[i, : len(pred["source_gt"][i])] = torch.tensor(pred["source_gt"][i])
                target_label[i, : len(pred["target_gt"][i])] = torch.tensor(pred["target_gt"][i])

        else:
            # bce loss
            # TODO: what are these magic numbers?
            source_label = torch.zeros([len(pred["source_gt"]), dataset_train.num_source_relationships], dtype=torch.float32).to(device=source_distribution.device)
            target_label = torch.zeros([len(pred["target_gt"]), dataset_train.num_target_relationships], dtype=torch.float32).to(device=source_distribution.device)
            for i in range(len(pred["source_gt"])):
                source_label[i, pred["source_gt"][i]] = 1
                target_label[i, pred["target_gt"][i]] = 1

        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        # losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        if not conf.bce_loss:
            losses["source_relation_loss"] = mlm_loss(source_distribution, source_label)
            losses["target_relation_loss"] = mlm_loss(target_distribution, target_label)

        else:
            losses["source_relation_loss"] = bce_loss(source_distribution, source_label)
            losses["target_relation_loss"] = bce_loss(target_distribution, target_label)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start = time.time()

        # del pred
        #
        # torch.cuda.empty_cache()

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(len(dataloader_test)):
        # for b in range(2):
            print(f"Fetching test data {b}")
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].to(object_detector_device))
            im_info = copy.deepcopy(data[1].to(object_detector_device))
            gt_boxes = copy.deepcopy(data[2].to(object_detector_device))
            num_boxes = copy.deepcopy(data[3].to(object_detector_device))
            gt_annotation = dataset_test.gt_annotations[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            entry = {k: v.to(sttran_device) if isinstance(v, torch.Tensor) else v for k, v in entry.items()}

            # Try to avoid GPU OOM
            im_data.to(cpu_device)
            im_info.to(cpu_device)
            gt_boxes.to(cpu_device)
            num_boxes.to(cpu_device)

            del im_data
            del im_info
            del gt_boxes
            del num_boxes

            torch.cuda.empty_cache()

            pred = model(entry)

            del entry

            torch.cuda.empty_cache()

            evaluator.evaluate_scene_graph(gt_annotation, pred)

        del pred

        torch.cuda.empty_cache()
            
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)



