import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
from imageio import imread
import numpy as np
import pickle
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
import yaml


class KitchenGenome(Dataset):

    def __init__(
            self,
            mode,
            datasize,
            config_path,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        with open(config_path) as config_file:
            config = yaml.load(config_file, yaml.SafeLoader)

        root_path = data_path

        self.frames_path = os.path.join(root_path, 'frames')

        self.object_classes = config["object_classes"]
        self.object_classes[self.object_classes.index('Nothing')] = '__background__'

        self.source_relationships = config["source_relationship_classes"]
        self.target_relationships = config["target_relationship_classes"]

        self.relationship_classes = self.source_relationships + self.target_relationships

        self.num_relationships = len(self.relationship_classes)
        self.num_source_relationships = len(self.source_relationships)
        self.num_target_relationships = len(self.target_relationships)

        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
                person_bbox = pickle.load(f)

            with open('dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
                person_bbox = pickle.load(f)

            with open(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
                object_bbox = pickle.load(f)

        print('--------------------finish!-------------------------')

        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.rsplit('/', maxsplit=1)
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if len(person_bbox[j]['bbox']) == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1

                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]

                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] is not None, f'warning! The object is visible without bbox'

                        k['class'] = self.object_classes.index(k['class'])
                        # k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['source_relationship'] = torch.tensor([self.source_relationships.index(r) for r in k['source_relationship']], dtype=torch.long)
                        k['target_relationship'] = torch.tensor([self.target_relationships.index(r) for r in k['target_relationship']], dtype=torch.long)

                        gt_annotation_frame.append(k)

                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(self.non_heatmap_nums))
        print('x' * 60)

    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            im = imread(os.path.join(self.frames_path, name)) # channel h,w,3
            im = im[:, :, ::-1] # rgb -> bgr
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 360, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
