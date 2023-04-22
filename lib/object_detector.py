import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from lib.funcs import assign_relations
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fasterRCNN.lib.model.roi_layers import nms


class detector(nn.Module):
    '''first part: object detection (image/video)'''

    def __init__(
            self,
            train,
            object_classes,
            use_SUPPLY,
            mode='predcls',
            state_dict=None,
            ignore_missing_keys=False,
            device=None,
            batch_size=10
    ):
        super(detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode
        self.device = torch.device("cuda:0") if device is None else device
        self.batch_size = batch_size
        self.cpu_device = torch.device("cpu")

        self.fasterRCNN = resnet(
            classes=self.object_classes,
            pretrained=False,
            class_agnostic=False
        )
        self.fasterRCNN.create_architecture()

        if state_dict:  # we're using Torchserve
            self.fasterRCNN.load_state_dict(
                state_dict,
                strict=not ignore_missing_keys
            )
        else:
            checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
            self.fasterRCNN.load_state_dict(
                checkpoint['model'],
                strict=not ignore_missing_keys
            )

        # self.fasterRCNN = self.fasterRCNN.to(torch.device("cuda:2"))

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

    def forward(self,
                im_data,
                im_info,
                gt_boxes=None,
                num_boxes=None,
                gt_annotation=None,
                im_all=None):
        if self.is_train:
            assert gt_boxes is not None
            assert num_boxes is not None
            assert gt_annotation is not None

        if self.mode == 'sgdet':
            counter = 0
            counter_image = 0

            # create saved-bbox, labels, scores, features
            FINAL_BBOXES = torch.tensor([]).to(self.device)
            FINAL_LABELS = torch.tensor([], dtype=torch.int64).to(self.device)
            FINAL_SCORES = torch.tensor([]).to(self.device)
            FINAL_FEATURES = torch.tensor([]).to(self.device)
            FINAL_BASE_FEATURES = torch.tensor([]).to(self.device)

            inputs_gtboxes = None
            inputs_numboxes = None

            while counter < im_data.shape[0]:
                #compute 10 images in batch and  collect all frames data in the video
                if counter + self.batch_size < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + self.batch_size]
                    inputs_info = im_info[counter:counter + self.batch_size]

                    if self.is_train:
                        inputs_gtboxes = gt_boxes[counter:counter + self.batch_size]
                        inputs_numboxes = num_boxes[counter:counter + self.batch_size]

                else:
                    inputs_data = im_data[counter:]
                    inputs_info = im_info[counter:]

                    if self.is_train:
                        inputs_gtboxes = gt_boxes[counter:]
                        inputs_numboxes = num_boxes[counter:]

                rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(
                    inputs_data, inputs_info, inputs_gtboxes, inputs_numboxes)

                SCORES = cls_prob.data
                boxes = rois.data[:, :, 1:5]
                # bbox regression (class specific)
                box_deltas = bbox_pred.data
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(self.device) \
                             + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).to(self.device)  # the first is normalize std, the second is mean
                box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                PRED_BOXES = clip_boxes(pred_boxes, im_info.data, 1)

                PRED_BOXES /= inputs_info[0, 2] # original bbox scale!!!!!!!!!!!!!!

                #traverse frames
                for i in range(rois.shape[0]):
                    # images in the batch
                    scores = SCORES[i]
                    pred_boxes = PRED_BOXES[i]

                    for j in range(1, len(self.object_classes)):
                        # NMS according to obj categories
                        inds = torch.nonzero(scores[:, j] > 0.1).view(-1) #0.05 is score threshold
                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[:, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.4) # NMS threshold
                            cls_dets = cls_dets[keep.view(-1).long()]

                            if j == 1:
                                # for person we only keep the highest score for person!
                                final_bbox = cls_dets[0,0:4].unsqueeze(0)
                                final_score = cls_dets[0,4].unsqueeze(0)
                                final_labels = torch.tensor([j]).to(self.device)
                                final_features = roi_features[i, inds[order[keep][0]]].unsqueeze(0)
                            else:
                                final_bbox = cls_dets[:, 0:4]
                                final_score = cls_dets[:, 4]
                                final_labels = torch.tensor([j]).repeat(keep.shape[0]).to(self.device)
                                final_features = roi_features[i, inds[order[keep]]]

                            final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(final_bbox.shape[0], 1).to(self.device),
                                                    final_bbox), 1)
                            FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                            FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                            FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                            FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)
                    FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat[i].unsqueeze(0)), 0)

                    counter_image += 1

                counter += self.batch_size

            FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)

            if self.is_train:
                prediction = {'FINAL_BBOXES': FINAL_BBOXES, 'FINAL_LABELS': FINAL_LABELS, 'FINAL_SCORES': FINAL_SCORES,
                              'FINAL_FEATURES': FINAL_FEATURES, 'FINAL_BASE_FEATURES': FINAL_BASE_FEATURES}

                DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(
                    prediction, gt_annotation, assign_IOU_threshold=0.5)

                if self.use_SUPPLY:
                    # supply the unfounded gt boxes by detector into the scene graph generation training
                    FINAL_BBOXES_X = torch.tensor([]).to(self.device)
                    FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).to(self.device)
                    FINAL_SCORES_X = torch.tensor([]).to(self.device)
                    FINAL_FEATURES_X = torch.tensor([]).to(self.device)
                    assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(FINAL_BBOXES_X.device)

                    for i, j in enumerate(SUPPLY_RELATIONS):
                        if len(j) > 0:
                            unfound_gt_bboxes = torch.zeros([len(j), 5]).to(self.device)
                            unfound_gt_classes = torch.zeros([len(j)], dtype=torch.int64).to(self.device)
                            one_scores = torch.ones([len(j)], dtype=torch.float32).to(self.device)  # probability
                            for m, n in enumerate(j):
                                # if person box is missing or objects
                                if 'bbox' in n.keys():
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = n['class']
                                else:
                                    # here happens always that IOU <0.5 but not unfounded
                                    unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']) * im_info[
                                        i, 2]  # don't forget scaling!
                                    unfound_gt_classes[m] = 1  # person class index

                            DETECTOR_FOUND_IDX[i] = list(np.concatenate((DETECTOR_FOUND_IDX[i],
                                                                         np.arange(
                                                                             start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                                                             stop=int(
                                                                                 sum(FINAL_BBOXES[:, 0] == i)) + len(
                                                                                 SUPPLY_RELATIONS[i]))), axis=0).astype(
                                'int64'))

                            GT_RELATIONS[i].extend(SUPPLY_RELATIONS[i])

                            # compute the features of unfound gt_boxes
                            pooled_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES[i].unsqueeze(0),
                                                                         unfound_gt_bboxes.to(self.device))
                            pooled_feat = self.fasterRCNN._head_to_tail(pooled_feat)
                            cls_prob = F.softmax(self.fasterRCNN.RCNN_cls_score(pooled_feat), 1)

                            unfound_gt_bboxes[:, 0] = i
                            unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                            FINAL_BBOXES_X = torch.cat(
                                (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i],
                                                        unfound_gt_classes))  # final label is not gt!
                            FINAL_SCORES_X = torch.cat(
                                (FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                            FINAL_FEATURES_X = torch.cat(
                                (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
                        else:
                            FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                            FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                            FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
                global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices

                im_idx = []  # which frame are the relations belong to
                pair = []
                s_rel = []
                t_rel = []
                for i, j in enumerate(DETECTOR_FOUND_IDX):

                    for k, kk in enumerate(GT_RELATIONS[i]):
                        if 'person_bbox' in kk.keys():
                            kkk = k
                            break
                    localhuman = int(global_idx[FINAL_BBOXES_X[:, 0] == i][kkk])

                    for m, n in enumerate(j):
                        if 'class' in GT_RELATIONS[i][m].keys():
                            im_idx.append(i)

                            pair.append([localhuman, int(global_idx[FINAL_BBOXES_X[:, 0] == i][int(n)])])

                            s_rel.append(GT_RELATIONS[i][m]['source_relationship'].tolist())
                            t_rel.append(GT_RELATIONS[i][m]['target_relationship'].tolist())

                pair = torch.tensor(pair).to(self.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(self.device)
                union_boxes = torch.cat((im_idx[:, None],
                                         torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                                   FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)

                union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

                del FINAL_BASE_FEATURES

                pair_rois = torch.cat((FINAL_BBOXES_X[pair[:,0],1:],FINAL_BBOXES_X[pair[:,1],1:]), 1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES_X,
                         'labels': FINAL_LABELS_X,
                         'scores': FINAL_SCORES_X,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'features': FINAL_FEATURES_X,
                         'union_feat': union_feat,
                         'spatial_masks': spatial_masks,
                         'source_gt': s_rel,
                         'target_gt': t_rel}

                return entry

            else:
                FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                PRED_LABELS = PRED_LABELS + 1

                entry = {'boxes': FINAL_BBOXES,
                         'scores': FINAL_SCORES,
                         'distribution': FINAL_DISTRIBUTIONS,
                         'pred_labels': PRED_LABELS,
                         'features': FINAL_FEATURES,
                         'fmaps': FINAL_BASE_FEATURES,
                         'im_info': im_info[0, 2]}

                return entry
        else:
            # how many bboxes we have
            bbox_num = 0

            im_idx = []  # which frame are the relations belong to
            pair = []
            s_rel = []
            t_rel = []

            for i in gt_annotation:
                bbox_num += len(i)
            FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).to(self.device)
            FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).to(self.device)
            FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).to(self.device)
            HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).to(self.device)

            bbox_idx = 0
            for i, j in enumerate(gt_annotation):
                for m in j:
                    if 'person_bbox' in m.keys():
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = 1
                        HUMAN_IDX[i] = bbox_idx
                        bbox_idx += 1
                    else:
                        FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                        FINAL_BBOXES[bbox_idx, 0] = i
                        FINAL_LABELS[bbox_idx] = m['class']
                        im_idx.append(i)
                        pair.append([int(HUMAN_IDX[i]), bbox_idx])
                        s_rel.append(m['source_relationship'].tolist())
                        t_rel.append(m['target_relationship'].tolist())
                        bbox_idx += 1
            pair = torch.tensor(pair).to(self.device)
            im_idx = torch.tensor(im_idx, dtype=torch.float).to(self.device)

            counter = 0
            
            FINAL_BASE_FEATURES_0 = torch.tensor([]).to(torch.device("cuda:2"))
            FINAL_BASE_FEATURES_1 = torch.tensor([]).to(torch.device("cuda:3"))
            # print(f"FINAL_BASE_FEATURES: {FINAL_BASE_FEATURES.size()}")

            while counter < im_data.shape[0]:
                #compute 10 images in batch and  collect all frames data in the video
                if counter + self.batch_size < im_data.shape[0]:
                    inputs_data = im_data[counter:counter + self.batch_size]
                else:
                    inputs_data = im_data[counter:]
                    
                base_feat = self.fasterRCNN.RCNN_base(inputs_data)
                print(f"base_feat.device: {base_feat.device}")
                
                if counter <= int(im_data.shape[0] / 2): 
                    FINAL_BASE_FEATURES_0 = torch.cat((FINAL_BASE_FEATURES_0, base_feat.to(torch.device("cuda:2"))), 0)
                else:
                    FINAL_BASE_FEATURES_1 = torch.cat((FINAL_BASE_FEATURES_1, base_feat.to(torch.device("cuda:3"))), 0)
                    
                counter += self.batch_size

            # FINAL_BASE_FEATURES = FINAL_BASE_FEATURES.to(self.device)
            print(f"FINAL_BASE_FEATURES_0: {FINAL_BASE_FEATURES_0.size()}")
            print(f"FINAL_BASE_FEATURES_0.device: {FINAL_BASE_FEATURES_0.device}")

            print(f"FINAL_BASE_FEATURES_1: {FINAL_BASE_FEATURES_1.size()}")
            print(f"FINAL_BASE_FEATURES_1.device: {FINAL_BASE_FEATURES_1.device}")

            FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]

            FINAL_BBOXES_0 = FINAL_BBOXES[:len(FINAL_BASE_FEATURES_0)].to(torch.device("cuda:2"))
            print(f"FINAL_BBOXES_0: {FINAL_BBOXES_0.size()}")
            print(f"FINAL_BBOXES_0.device: {FINAL_BBOXES_0.device}")
            
            FINAL_BBOXES_1 = FINAL_BBOXES[len(FINAL_BASE_FEATURES_0):].to(torch.device("cuda:3"))
            print(f"FINAL_BBOXES_1: {FINAL_BBOXES_1.size()}")
            print(f"FINAL_BBOXES_1.device: {FINAL_BBOXES_1.device}")
            
            FINAL_FEATURES_0 = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES_0, FINAL_BBOXES_0)
            print(f"FINAL_FEATURES_0 (roi_align): {FINAL_FEATURES_0.size()}")
            print(f"FINAL_FEATURES_0.device: {FINAL_FEATURES_0.device}")
            
            FINAL_FEATURES_1 = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES_1, FINAL_BBOXES_1)
            print(f"FINAL_FEATURES_1 (roi_align): {FINAL_FEATURES_1.size()}")
            print(f"FINAL_FEATURES_1.device: {FINAL_FEATURES_1.device}")

            FINAL_FEATURES_0 = self.fasterRCNN._head_to_tail(FINAL_FEATURES_0).to(self.device)
            print(f"FINAL_FEATURES_0 (head_to_tail): {FINAL_FEATURES_0.size()}")
            print(f"FINAL_FEATURES_0.device: {FINAL_FEATURES_0.device}")

            FINAL_FEATURES_1 = self.fasterRCNN._head_to_tail(FINAL_FEATURES_1).to(self.device)
            print(f"FINAL_FEATURES_1 (head_to_tail): {FINAL_FEATURES_1.size()}")
            print(f"FINAL_FEATURES_1.device: {FINAL_FEATURES_1.device}")

            FINAL_FEATURES = torch.cat((FINAL_FEATURES_0, FINAL_FEATURES_1), 0)

            if self.mode == 'predcls':
                union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES_0, FINAL_BASE_FEATURES_1), 0).to(self.device)
                union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
                print(f"union_feat.device: {union_feat.device}")

                FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                entry = {'boxes': FINAL_BBOXES,
                         'labels': FINAL_LABELS, # here is the groundtruth
                         'scores': FINAL_SCORES,
                         'im_idx': im_idx,
                         'pair_idx': pair,
                         'human_idx': HUMAN_IDX,
                         'features': FINAL_FEATURES,
                         'union_feat': union_feat,
                         'union_box': union_boxes,
                         'spatial_masks': spatial_masks,
                         'source_gt': s_rel,
                         'target_gt': t_rel
                        }

                return entry
            elif self.mode == 'sgcls':
                if self.is_train:

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    union_boxes = torch.cat(
                        (im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                         torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
                    union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
                    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                                          1).data.cpu().numpy()
                    spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'union_feat': union_feat,
                             'union_box': union_boxes,
                             'spatial_masks': spatial_masks,
                             'source_gt': s_rel,
                             'target_gt': t_rel}

                    return entry
                else:
                    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]

                    FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
                    FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
                    PRED_LABELS = PRED_LABELS + 1

                    entry = {'boxes': FINAL_BBOXES,
                             'labels': FINAL_LABELS,  # here is the groundtruth
                             'scores': FINAL_SCORES,
                             'distribution': FINAL_DISTRIBUTIONS,
                             'pred_labels': PRED_LABELS,
                             'im_idx': im_idx,
                             'pair_idx': pair,
                             'human_idx': HUMAN_IDX,
                             'features': FINAL_FEATURES,
                             'source_gt': s_rel,
                             'target_gt': t_rel,
                             'fmaps': FINAL_BASE_FEATURES,
                             'im_info': im_info[0, 2]}

                    return entry

