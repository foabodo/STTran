import numpy as np
np.set_printoptions(precision=4)
import copy
from os import makedirs, path, cpu_count
from tqdm import tqdm
import torch

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test",
                datasize=conf.datasize,
                data_path=conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset,
                                         shuffle=False,
                                         num_workers=int(cpu_count() / 2),
                                         collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False,
                           object_classes=AG_dataset.object_classes,
                           use_SUPPLY=True,
                           mode=conf.mode).to(device=gpu_device)
object_detector.eval()


model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               word_vec_dir=conf.word_vec_dir).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='no')

if conf.result_path and not path.exists(conf.result_path):
    makedirs(conf.result_path)

with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)):
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        # print("im_data: {}".format(im_data))
        print("         {}".format(im_data.size()))
        # print("im_info: {}".format(im_info))
        print("         {}".format(im_info.size()))

        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

        # for k, v in entry.items():
        print(f"im_info: {entry['im_info'].size()}")

        print(f"boxes: {entry['boxes'].size()}")
        pred = model(entry)

        video_name, frame_num = AG_dataset.video_list[b][0].split('/')

        evaluator1.evaluate_scene_graph(
            gt_annotation,
            dict(pred),
            result_path=path.join(conf.result_path, f'{video_name}_{frame_num}_constraint=with.json')
            if conf.result_path else None
        )
        evaluator2.evaluate_scene_graph(
            gt_annotation,
            dict(pred),
            result_path=path.join(conf.result_path, f'{video_name}_{frame_num}_constraint=semi.json')
            if conf.result_path else None
        )
        evaluator3.evaluate_scene_graph(
            gt_annotation,
            dict(pred),
            result_path=path.join(conf.result_path, f'{video_name}_{frame_num}_constraint=no.json')
            if conf.result_path else None
        )


print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()
