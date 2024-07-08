import os
import keras
import cv2
import sys
import numpy as np
sys.path.append("../../")
from infer_int_base import InferIntBase
from models.yolov83 import YOLOv8Light
from keras.models import Model
import argparse


def sigmoid(x):
    return 1/(1+np.exp(-x))


class InferPTQ(InferIntBase):
    def detect(self, inp, conf_soft=0.3, iou_thresh=0.4, save=True):
        if isinstance(inp, str):        ### img is image-path
            img_org = cv2.imread(inp)
        else:                           ### img is array
            img_org = inp
        h_org, w_org, _ = img_org.shape
        img = cv2.resize(img_org, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        all_output_model = self.infer(img)
#         out0 = all_output_model["ldetect_loc_pw0"][0]
#         out1 = all_output_model["ldetect1_loc_pw0"][0]
#         out2 = all_output_model["ldetect2_loc_pw0"][0]
#         preds = [out0, out1, out2]
#
#         M0 = self.model_quant["post_process"]["out1_multiplier"]
#         n0 = self.model_quant["post_process"]["out1_shift"]
#         Z0 = self.model_quant["post_process"]["out1_off"]
#         M1 = self.model_quant["post_process"]["out2_multiplier"]
#         n1 = self.model_quant["post_process"]["out2_shift"]
#         Z1 = self.model_quant["post_process"]["out2_off"]
#         M2 = self.model_quant["post_process"]["out3_multiplier"]
#         n2 = self.model_quant["post_process"]["out3_shift"]
#         Z2 = self.model_quant["post_process"]["out3_off"]
#         Ms = [M0, M1, M2]
#         ns = [n0, n1, n2]
#         Zs = [Z0, Z1, Z2]
#
#         conf_soft = np.log(conf_soft / (1 - conf_soft))     # scale conf_soft to before-sigmoid conf
#         print(conf_soft)
#         conf_soft0 = conf_soft * pow(2, 31 - n0) / M0 + Z0
#         conf_soft1 = conf_soft * pow(2, 31 - n1) / M1 + Z1
#         conf_soft2 = conf_soft * pow(2, 31 - n2) / M2 + Z2
#         conf_soft = [conf_soft0, conf_soft1, conf_soft2]
#         print(conf_soft)
#
#         anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
#         anchors = np.array(anchors).reshape(-1, 2)
#         anchors = anchors * 320 // 416
#         # print(anchors)
#         anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#         print(anchors)
#
#         propose = []
#         for map in range(3):
#             print(f"---{map}---")
#             grid = preds[map].shape[1]
#             print(preds[map].shape)
#             predict = np.reshape(preds[map], (grid, grid, 3, 6))
#             for y in range(grid):
#                 for x in range(grid):
#                     for anchor in range(3):
#                         conf = predict[y][x][anchor][4]
#                         if conf >= conf_soft[map]*6/7:
#                             print(conf, conf_soft[map]*6/7)
#                             conf_fp = (conf - Zs[map]) * Ms[map] / pow(2, 31 - ns[map])
#                             box_fp = (predict[y][x][anchor][:4] - Zs[map]) * Ms[map] / pow(2, 31 - ns[map])
#                             propose.append([box_fp, conf_fp, x, y, grid, anchors[anchor_mask[map]][anchor]])
#         print("num propose this map: ", len(propose))
#
#         re_propose = []
#         for box in propose:
#             x = (sigmoid(box[0][0]) + box[2]) / box[4]
#             y = (sigmoid(box[0][1]) + box[3]) / box[4]
#             print(box[1], x, y)
#             w = (np.exp(box[0][2]) * box[5][0]) / 320
#             h = (np.exp(box[0][3]) * box[5][1]) / 320
#
#             # w_org, h_org = image_shape[1], image_shape[0]
#             xmin, ymin, xmax, ymax = (x-w/2)*w_org, (y-h/2)*h_org, (x+w/2)*w_org, (y+h/2)*h_org
#             rebox = [int(np.floor(v+0.5)) for v in [xmin, ymin, xmax, ymax]]
#             rebox = [box[1]] + rebox
#
#             re_propose.append(rebox)
#
#         # print("---repropose---:", repropose)
#         re_propose = sorted(re_propose, key=itemgetter(0), reverse=True)
#         # print("re_propose:", re_propose)
#         final = []
#         while len(re_propose) > 0:
#             box_pass = re_propose[0]
#             final.append(box_pass)
#             for box in re_propose[1:]:
#                 iou = get_iou(box_pass[1:], box[1:])
#                 if iou > iou_thresh:
#                     re_propose.remove(box)
#             re_propose.remove(box_pass)
#
#         print("---final---:", len(final))
#         print("---final---:", final)
#
#         if save:
#             for box in final:
#                 box = box[1:]
#                 cv2.rectangle(img_org, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#             cv2.imwrite("quantize_infer_model/test.jpg", img_org)
#
#         return final
#
#     def track(self, video_path, thresh_iou=0.6, conf_soft=0.2, conf_hard=0.3):
#         video_name = "test"
#         cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
#         cap = cv2.VideoCapture(video_path)
#
#         ret, frame = cap.read()
#         h_org, w_org, _ = frame.shape
#         pre_boxes = []
#         count = 0
#         while ret:
#             predict_boxes = self.detect(frame, conf_soft=conf_soft, save=False)
#             print(predict_boxes)
#             boxes, count = matching_boxes(pre_boxes, predict_boxes, count, thresh_iou=thresh_iou, conf=conf_hard)
#             print(boxes)
#             pre_boxes = boxes
#             for box in boxes:
#                 frame = cv2.rectangle(frame, (box[1], box[2]), (box[3], box[4]), (0, 255, 0), 2)
#                 cv2.putText(frame, str(box[0]), (box[1], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             cv2.imshow(video_name, frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             ret, frame = cap.read()
#
#
# if __name__ == "__main__":
#     json_path = "quantize_infer_model/saved/light8n-1st-fix.json"
#     light8n_fused = YOLOv8Light(type="n", fuse=True)
#     gen_output = GenRefOutput(light8n_fused,
#                               json_path,
#                               "quantize_infer_model/saved/txt_ref_output",
#                               "quantize_infer_model/01b_0_104124_000002.jpg",
#                               )
#
#     def detect_task():
#         """
#         Test detect in integer-only
#         """
#         gen_output.detect("quantize_infer_model/test/inp/test1.jpg", conf_soft=0.3, iou_thresh=0.4, save=True)
#
#
#     def track_task():
#         """
#         Test track in integer-only
#         """
#         gen_output.track("quantize_infer_model/test/inp/01b_0_104124.mp4", thresh_iou=0.4, conf_soft=0.2, conf_hard=0.3)
#
#
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-m', '--mode', help='use mode', default="track")
#     args = ap.parse_args()
#
#     print("---USE MODE---: ", args.mode)
#     elif args.mode == "detect":
#         detect_task()
#     elif args.mode == "track":
#         track_task()