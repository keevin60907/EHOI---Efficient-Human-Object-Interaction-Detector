import cv2
import math
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import xgboost as xgb
import torch.nn as nn
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from PIL import Image, ImageDraw
from pocket.data import HICODet
from pocket.utils import DetectionAPMeter, BoxPairAssociation
from torchvision.ops.boxes import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from labeling import spatial_feat

from multiforest_v1 import masking, hoi_dict
import multiforest_v1
import multiforest_XGB
from greenlab.dft import Conditional_DFT, DiscriminantFeatureTest
#from multi_model import Spatial_Classifier, softmax

def masking(hoi_list, obj_list, verb_list):
    ''' Binary masking to eliminating the impossible object and relation pair
    Arg(s):
        hoi_list(list)   : the str list of hoi pairs (verb + ' ' + obj)
        obj_list(list)   : the str list of objects
        verb_list(list)  : the str list of verbs
    Return(s):
        masking(np.array): the binary array with size (obj, verb)
    '''
    masking = np.zeros((len(obj_list), len(verb_list)))
    for hoi in hoi_list:
        verb, obj = hoi.split()
        masking[obj_list.index(obj), verb_list.index(verb)] = 1
    return masking

def hoi_dict(hoi_list, obj_list, verb_list):
    ''' Query the hoi index by the (verb, obj) pair
    Arg(s):
        hoi_list(list)   : the str list of hoi pairs (verb + ' ' + obj)
        obj_list(list)   : the str list of objects
        verb_list(list)  : the str list of verbs
    Return(s):
        hoi_dict(dict)   : the dict with key (vb, obj) and value (hoi)
    '''
    hoi_dict = {}
    for idx, hoi in enumerate(hoi_list):
        verb, obj = hoi.split()
        hoi_dict[(verb_list.index(verb), obj_list.index(obj))] = idx
    return hoi_dict

@torch.no_grad()
def eval_ref(tree, object_info,
             dataset, masking, hoi_dict, datasplit='test2015',
             threshold=0.1, use_gpu=False):

    print(f'Referenced mAP by using ground truth bounding boxes')
    associate = BoxPairAssociation(min_iou=0.5)

    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )
    use_bg = True
    use_bg = True
    if use_bg:
        import pocket.models as models
        from torchvision import transforms
        backbone_name = 'resnet50'
        detector = models.fasterrcnn_resnet_fpn(backbone_name,
                pretrained=True)
        backbone = detector.backbone
        from torchvision.models import resnet50, ResNet50_Weights
        background = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        background.fc = nn.Identity()
        from torchvision.ops import MultiScaleRoIAlign
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=3,
            sampling_ratio=2)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    n_pred, n_gt = 0, 0
    metric = MeanAveragePrecision(class_metrics=True)
    preds, target = [], []
    for idx, (image, gt) in enumerate(dataset):
        #if (idx%100) == 0:
        #    print(f'Updating... {idx:0>5d} ({100*idx/len(dataset):.2f}%)')
        img_w, img_h = image.width, image.height
        if use_bg:
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            whole = torch.Tensor([0, 0, img_w, img_h]).reshape((1, -1))

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                backbone = backbone.to('cuda')
                whole = whole.to('cuda')
                background = background.to('cuda')

            with torch.no_grad():
                #bg_info = box_roi_pool(backbone(input_batch), [whole], [image.size])
                bg_info = background(input_batch)
                bg_info = bg_info.cpu().numpy()

        filename = dataset.filename(idx).split('.')[0]
        # Read detection annotations
        obj_anno = torch.load(f'./obj_anno/{datasplit}/{filename}.pth')
        # Do NMS for the output of detection
        keep_o = batched_nms(
            torch.Tensor(gt['boxes_o']),
            torch.ones_like(torch.Tensor(gt['object'])),
            torch.Tensor(gt['object']),
            iou_threshold=0.7
        )
        keep_h = batched_nms(
            torch.Tensor(gt['boxes_h']),
            torch.ones_like(torch.Tensor(gt['object'])),
            torch.zeros_like(torch.Tensor(gt['object'])),
            iou_threshold=0.7
        )
        hum_blist = torch.Tensor(gt['boxes_h'])[keep_h].tolist()
        obj_blist = torch.Tensor(gt['boxes_o'])[keep_o].tolist()
        hum_slist = torch.ones_like(torch.Tensor(gt['object']))[keep_h].tolist()
        obj_slist = torch.ones_like(torch.Tensor(gt['object']))[keep_o].tolist()
        obj_entity = torch.IntTensor(gt['object'])[keep_o].tolist()
        x_test_hum, x_test_obj = [], []
        x_score = []
        hoi_box, det_obj = [], []
        hum_boxes, obj_boxes = [], []

        img = preprocess(image).unsqueeze(0)
        if torch.cuda.is_available():
            with torch.no_grad():
                hum_feats = box_roi_pool(backbone(img.to('cuda')),
                                         [torch.Tensor(gt['boxes_h'])[keep_h].to('cuda')],
                                         [image.size]).cpu()
                obj_feats = box_roi_pool(backbone(img.to('cuda')),
                                         [torch.Tensor(gt['boxes_o'])[keep_o].to('cuda')],
                                         [image.size]).cpu()
        else:
            with torch.no_grad():
                hum_feats = box_roi_pool(backbone(img),
                                         [torch.Tensor(gt['boxes_h'])[keep_h]],
                                         [image.size])
                obj_feats = box_roi_pool(backbone(img),
                                         [torch.Tensor(gt['boxes_o'])[keep_o]],
                                         [image.size])
        entity_coding = np.zeros(80)
        for label in obj_entity:
            entity_coding[label] += 1

        for hum_box, hum_feat, hum_score in zip(hum_blist, hum_feats, hum_slist):
            hum_feat = hum_feat.numpy().reshape(-1)
            for idx, (obj_box, obj_feat, obj_score) in enumerate(zip(obj_blist, obj_feats, obj_slist)):
                sp_feat = spatial_feat(hum_box, obj_box, img_w, img_h)
                obj_feat = obj_feat.numpy().reshape(-1)
                if use_bg:
                    bg_info = bg_info.flatten()
                    #feat = np.concatenate([hum_feat, bg_info, sp_feat])
                    test_hum = np.concatenate([hum_feat, bg_info, sp_feat, entity_coding])
                    test_obj = np.concatenate([obj_feat, bg_info, sp_feat, entity_coding])
                else:
                    #feat = np.concatenate([hum_feat, bg_info, sp_feat])
                    feat = np.concatenate([hum_feat, obj_feat, sp_feat])
                x_test_hum.append(test_hum)
                x_test_obj.append(test_obj)
                x_score.append(np.array([hum_score*obj_score]))
                hoi_box.append([min(hum_box[0], obj_box[0]),
                                min(hum_box[1], obj_box[1]),
                                max(hum_box[2], obj_box[2]),
                                max(hum_box[3], obj_box[3])])
                hum_boxes.append(hum_box)
                obj_boxes.append(obj_box)
                det_obj.append(obj_entity[idx])

        x_test_hum = np.array(x_test_hum)
        x_test_obj = np.array(x_test_obj)
        x_score = np.array(x_score)
        if x_test_hum.shape[0] != 0:
            #y_num = surpressor.predict(x_test)
            y_pred_hum = tree[0].predict_prob(x_test_hum, det_obj) * masking[det_obj, :]
            y_pred_obj = tree[1].predict_prob(x_test_obj, det_obj) * masking[det_obj, :]
            if len(tree) > 2:
                y_pred = np.zeros_like(y_pred_hum)
                for idx, ent in enumerate(det_obj):
                    y_pred[idx, :] = tree[2][ent, :, 0]*y_pred_hum[idx, :] + tree[2][ent, :, 1]*y_pred_obj[idx, :]
                y_pred[y_pred > 1] = 1
            else:
                y_pred = np.maximum(y_pred_hum, y_pred_obj)
            #y_pred = softmax(y_pred, axis=1)
            #y_pred = y_pred / (y_pred.max(axis=1, keepdims=True)+1e-6) 
            y_pred  = y_pred
            index = np.argwhere(y_pred > threshold)
            pred_box, pred_score, pred_hoi = [], [], []
            pred_hbox, pred_obox = [], []

            # index is in the form k pairs of (i, j)
            # i is the ith object from the detection, 
            # j is the possible verb,
            # Thus, the index array is composed of (i1, j1), (i1, j2), (i2, j3)...
            for i in range(index.shape[0]):
                pred_hoi.append(hoi_dict[(index[i, 1], det_obj[index[i, 0]])])
                pred_box.append(hoi_box[index[i, 0]])
                pred_hbox.append(hum_boxes[index[i, 0]])
                pred_obox.append(obj_boxes[index[i, 0]])
                pred_score.append(y_pred[index[i, 0], index[i, 1]])

            pred_box = torch.tensor(pred_box)
            pred_hbox = torch.tensor(pred_hbox)
            pred_obox = torch.tensor(pred_obox)
            pred_score = torch.tensor(pred_score, dtype=torch.float)
            pred_hoi = torch.tensor(pred_hoi)

            labels = torch.zeros_like(pred_score)
            unique_hoi = pred_hoi.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(torch.tensor(gt['hoi']) == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(pred_hoi == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (torch.tensor(gt['boxes_h'])[gt_idx].view(-1, 4),
                        torch.tensor(gt['boxes_o'])[gt_idx].view(-1, 4)),
                        (pred_hbox[det_idx].view(-1, 4),
                        pred_obox[det_idx].view(-1, 4)),
                        pred_score[det_idx].view(-1)
                    )

            meter.append(pred_score, pred_hoi, labels)
    return meter.eval()

@torch.no_grad()
def eval(tree, object_info,
         dataset, masking, hoi_dict, datasplit='test2015',
         threshold=0.1, use_gpu=False):

    associate = BoxPairAssociation(min_iou=0.5)
    meter = DetectionAPMeter(
        600, nproc=1,
        num_gt=testset.anno_interaction,
        algorithm='11P'
    )

    # The output of the detector is in the form of COCO labeling
    use_bg = True
    if use_bg:
        roi_size = 5
        print(f'The feature is aligned to {roi_size}*{roi_size} grids.')
        from torchvision.models import resnet50, ResNet50_Weights
        background = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        background.fc = nn.Identity()

        from torchvision import transforms
        
        import pocket.models as models
        backbone_name = 'resnet50'
        detector = models.fasterrcnn_resnet_fpn(backbone_name,
                pretrained=True)
        backbone = detector.backbone
        '''
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        backbone = detector.backbone
        '''
        from torchvision.ops import MultiScaleRoIAlign
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=roi_size,
            sampling_ratio=2)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    #n_pred, n_gt = 0, 0
    #metric = MeanAveragePrecision(class_metrics=True)
    #preds, target = [], []
    for idx, (image, gt) in enumerate(dataset):
        #if (idx%100) == 0:
        #    print(f'Updating... {idx:0>5d} ({100*idx/len(dataset):.2f}%)')
        img_w, img_h = image.width, image.height
        if use_bg:
            input_batch = preprocess(image).unsqueeze(0)
            whole = torch.Tensor([0, 0, img_w, img_h]).reshape((1, -1))

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                backbone = backbone.to('cuda')
                whole = whole.to('cuda')
                background = background.to('cuda')

            with torch.no_grad():
                bg_info = background(input_batch)
                bg_info = bg_info.cpu().numpy()
    
        filename = dataset.filename(idx).split('.')[0]
        # Read detection annotations
        obj_anno = torch.load(f'./obj_anno/{datasplit}/{filename}_detr-r101.pth')
        #obj_anno = torch.load(f'./obj_anno/{datasplit}/{filename}.pth')
        # Do NMS for the output of detection
        keep = batched_nms(
            obj_anno['boxes'], obj_anno['scores'],
            obj_anno['labels'], iou_threshold=0.5
        )
        boxes, labels, scores = obj_anno['boxes'][keep], obj_anno['labels'][keep], obj_anno['scores'][keep]

        img = preprocess(image).unsqueeze(0)
        if torch.cuda.is_available():
            with torch.no_grad():
                feat = box_roi_pool(backbone(img.to('cuda')), [boxes.to('cuda')], [image.size])
        else:
            with torch.no_grad():
                feat = box_roi_pool(backbone(img), [boxes], [image.size])

        hum_blist, obj_blist = boxes[labels==0].tolist(), boxes.tolist()
        hum_feats, obj_feats = feat[labels==0].cpu(), feat.cpu()
        hum_slist, obj_slist = scores[labels==0].tolist(), scores.tolist()
        avg_feats = obj_feats.numpy().mean(axis=0).reshape(-1)
        obj_entity = tree[0][0].coco2hico(labels)
        entity_coding = np.zeros(80)
        for label in obj_entity:
            entity_coding[label] += 1
        x_test_hum, x_test_obj = [], []
        x_score = []
        hoi_box, det_obj = [], []
        hum_boxes, obj_boxes = [], []

        for hum_box, hum_feat, hum_score in zip(hum_blist, hum_feats, hum_slist):
            hum_feat = hum_feat.numpy().reshape(-1)
            for idx, (obj_box, obj_feat, obj_score) in enumerate(zip(obj_blist, obj_feats, obj_slist)):
                obj_feat = obj_feat.numpy().reshape(-1)
                if obj_box != hum_box:
                    sp_feat = spatial_feat(hum_box, obj_box, img_w, img_h)
                    if use_bg:
                        bg_info = bg_info.flatten()
                        test_hum = np.concatenate([hum_feat, bg_info, sp_feat, entity_coding])
                        test_obj = np.concatenate([obj_feat, bg_info, sp_feat, entity_coding])
                    else:
                        #feat = np.concatenate([hum_feat, sp_feat])
                        feat = np.concatenate([hum_feat, obj_feat, sp_feat])
                    x_test_hum.append(test_hum)
                    x_test_obj.append(test_obj)
                    x_score.append(np.array([hum_score*obj_score]))
                    hoi_box.append([min(hum_box[0], obj_box[0]),
                                    min(hum_box[1], obj_box[1]),
                                    max(hum_box[2], obj_box[2]),
                                    max(hum_box[3], obj_box[3])])
                    hum_boxes.append(hum_box)
                    obj_boxes.append(obj_box)
                    det_obj.append(obj_entity[idx])

        x_test_hum = np.array(x_test_hum)
        x_test_obj = np.array(x_test_obj)
        x_score = np.array(x_score)
        if x_test_hum.shape[0] != 0:
            onehot_pred_hum = tree[0][0].predict_prob(x_test_hum, det_obj) * masking[det_obj, :]
            onehot_pred_obj = tree[1][0].predict_prob(x_test_obj, det_obj) * masking[det_obj, :]
            hamming_pred_hum = tree[0][1].predict_prob(x_test_hum, det_obj) * masking[det_obj, :]
            hamming_pred_obj = tree[1][1].predict_prob(x_test_obj, det_obj) * masking[det_obj, :]
            
            onehot_pred = np.zeros_like(onehot_pred_hum)
            hamming_pred = np.zeros_like(onehot_pred_hum)
            y_pred = np.zeros_like(onehot_pred_hum)

            for idx, ent in enumerate(det_obj):
                onehot_pred[idx, :] = tree[2][0][ent, :, 0]*onehot_pred_hum[idx, :] +\
                                      tree[2][0][ent, :, 1]*onehot_pred_obj[idx, :]
                # greater than 1 is used in the attempt
                onehot_pred[onehot_pred > 1] = 1
                onehot_pred[onehot_pred < 0] = 0
                hamming_pred[idx, :] = tree[2][1][ent, :, 0]*hamming_pred_hum[idx, :] +\
                                       tree[2][1][ent, :, 1]*hamming_pred_obj[idx, :]
                hamming_pred[hamming_pred > 1] = 1
                hamming_pred[hamming_pred < 0] = 0
                y_pred[idx, :] = tree[2][2][ent, :, 0]*onehot_pred[idx, :] +\
                                 tree[2][2][ent, :, 1]*hamming_pred[idx, :]
            y_pred[y_pred > 1] = 1

            y_pred = y_pred
            index = np.argwhere(y_pred > threshold)
            pred_box, pred_score, pred_hoi = [], [], []
            pred_hbox, pred_obox = [], []
            pred_rel = []

            # index is in the form k pairs of (i, j)
            # i is the ith object from the detection, 
            # j is the possible verb,
            # Thus, the index array is composed of (i1, j1), (i1, j2), (i2, j3)...
            for i in range(index.shape[0]):
                pred_hoi.append(hoi_dict[(index[i, 1], det_obj[index[i, 0]])])
                pred_rel.append(index[i, 1])
                pred_box.append(hoi_box[index[i, 0]])
                pred_hbox.append(hum_boxes[index[i, 0]])
                pred_obox.append(obj_boxes[index[i, 0]])
                pred_score.append(y_pred[index[i, 0], index[i, 1]])
            #import pdb; pdb.set_trace()

            pred_box = torch.tensor(pred_box)
            pred_hbox = torch.tensor(pred_hbox)
            pred_obox = torch.tensor(pred_obox)
            pred_score = torch.tensor(pred_score, dtype=torch.float)
            pred_hoi = torch.tensor(pred_hoi)
            pred_rel = torch.tensor(pred_rel)
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(pred_score)

            unique_hoi = pred_hoi.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(torch.tensor(gt['hoi']) == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(pred_hoi == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (torch.tensor(gt['boxes_h'])[gt_idx].view(-1, 4),
                        torch.tensor(gt['boxes_o'])[gt_idx].view(-1, 4)),
                        (pred_hbox[det_idx].view(-1, 4),
                        pred_obox[det_idx].view(-1, 4)),
                        pred_score[det_idx].view(-1)
                    )
            meter.append(pred_score, pred_hoi, labels)
    return meter.eval()

if __name__ == '__main__':

    print(f'============ Loading Embeddings ============')
    print('Using fastText word embedding with dimension 300')
    with open('./spatial_feat/fastText.npy', 'rb') as f:
        object_info = np.load(f)
        print('Performing SVD for (80, 300) embeddings...')
        _, _, d = np.linalg.svd(object_info, full_matrices=False)
    object_info = object_info@d.T
    print(f'================= End Loading =================')

    test_path = '../hico_20160224_det/images/test2015'
    test_anno = '../hico_20160224_det/instances_test2015.json'
    testset = HICODet(root=test_path, anno_file=test_anno)
    #import pdb; pdb.set_trace()
    print('HICO-Det dataset loaded')

    num_anno = torch.tensor(HICODet(None, anno_file='../hico_20160224_det/instances_train2015.json').anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    '''
    num_anno = torch.tensor(HICODet(None, anno_file='../hico_20160224_det/instances_train2015.json').anno_action)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    '''
    #print(rare)
    #print(non_rare)
    
    hoi_list = testset.interactions
    verb_list = testset.verbs
    obj_list = testset.objects

    mask = masking(hoi_list, obj_list, verb_list)
    with open(f'./nms_feat/masking.npy', 'wb') as f:
        np.save(f, mask)
    rel_dict = hoi_dict(hoi_list, obj_list, verb_list)
    print(f'========== Initiation one-hot classifier ==========')
    onehot_hum2ent = multiforest_v1.forest_cls(mask, obj_list, n_neighbors=1)
    onehot_ent2hum = multiforest_v1.forest_cls(mask, obj_list)

    nestmodel_path = './two_head/'
    freq_path = './forest_model/frequency.npy'
    onehot_hum2ent.load_config(use_gpu=False)
    onehot_hum2ent.load(model_dir=f'{nestmodel_path}hum2ent_v13/')

    onehot_ent2hum.load_config(use_gpu=False)
    onehot_ent2hum.load(model_dir=f'{nestmodel_path}ent2hum_v13/')

    h2e_th_path = './two_head/h2e_threshold_focal_v13.npy'
    onehot_hum2ent.load_threshold(path=h2e_th_path)
    e2h_th_path = './two_head/e2h_threshold_focal_v13.npy'
    onehot_ent2hum.load_threshold(path=e2h_th_path)

    weight_path = './XGB_head/merging_weights_onehot_v2.npy'
    with open(weight_path, 'rb') as f:
        onehot_weights = np.load(f)

    print(f'Load classifier from the dir: {nestmodel_path}')
    print(f'Load thresholds from the file: {h2e_th_path}')
    print(f'Load thresholds from the file: {e2h_th_path}')
    print(f'Load conditional probability from the file: {freq_path}')
    print(f'Load the merging weights from the file: {weight_path}')

    print(f'========== Initiation Hamming classifier ==========')
    hamming_hum2ent = multiforest_XGB.forest_cls(mask, obj_list, n_neighbors=1)
    hamming_ent2hum = multiforest_XGB.forest_cls(mask, obj_list)

    nestmodel_path = './XGB_head/'
    freq_path = './forest_model/frequency.npy'
    hamming_hum2ent.load_config(use_gpu=False)
    hamming_hum2ent.load(model_dir=f'{nestmodel_path}hum2ent_v1/')

    hamming_ent2hum.load_config(use_gpu=False)
    hamming_ent2hum.load(model_dir=f'{nestmodel_path}ent2hum_v1/')

    h2e_th_path = './XGB_head/h2e_threshold_v1.npy'
    hamming_hum2ent.load_threshold(path=h2e_th_path)
    e2h_th_path = './XGB_head/e2h_threshold_v1.npy'
    hamming_ent2hum.load_threshold(path=e2h_th_path)

    weight_path = './XGB_head/merging_weights_hamming_v2.npy'
    with open(weight_path, 'rb') as f:
        hamming_weights = np.load(f)

    print(f'Load classifier from the dir: {nestmodel_path}')
    print(f'Load thresholds from the file: {h2e_th_path}')
    print(f'Load thresholds from the file: {e2h_th_path}')
    print(f'Load conditional probability from the file: {freq_path}')
    print(f'Load the merging weights from the file: {weight_path}')

    #tree = [hum2ent, ent2hum]
    weight_path = './XGB_head/merging_weights_hybrid_v2.npy'
    with open(weight_path, 'rb') as f:
        hybrid_weights = np.load(f)
    print('[*]')
    print(f'Load the two coding merging weights from the file: {weight_path}')
    tree = [(onehot_hum2ent, hamming_hum2ent),
            (onehot_ent2hum, hamming_ent2hum),
            (onehot_weights, hamming_weights, hybrid_weights)]
    print(f'================= End Loading =================')

    print('Start Evaluation...')
    threshold = 0.05
    print(f'threshold={threshold}')
    print(f'NMS thresholding = 0.5')
    metric = eval(tree, object_info,
                  testset, mask, rel_dict,
                  threshold=threshold, use_gpu=False)
    print("Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
        metric.mean(), metric[rare].mean(), metric[non_rare].mean()
    ))
    eval_path = './evals/hybrid_v2.pkl'
    index = eval_path.find('.pkl')
    with open(eval_path, 'wb') as f:
        pickle.dump(metric, f)
    print(f'The Evaluation is saved to :\n{eval_path}')

    #ref_metric = eval_ref(tree, object_info,
    #                      testset, mask, rel_dict,
    #                      threshold=threshold, use_gpu=False)
    #print("Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
    #    ref_metric.mean(), ref_metric[rare].mean(), ref_metric[non_rare].mean()
    #))
    #ref_path = eval_path[:index] + '_ref' + eval_path[index:]
    #with open(ref_path, 'wb') as f:
    #    pickle.dump(ref_metric, f)
    #print(f'The Referrenced Evaluation is saved to :\n{ref_path}')
