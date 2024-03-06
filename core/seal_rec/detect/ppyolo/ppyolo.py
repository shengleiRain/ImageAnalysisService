import cv2
import numpy as np
import onnxruntime as ort
from utils.image_utils import get_color_map_list, normalize
from core.seal_rec.detect.ppyolo.config import seal_detect_args
from utils.bbox_nms import multiclass_nms
from utils.image_utils import resize_image
from core.seal_rec.detect.ppyolo.infer import PredictConfig
from core.seal_rec.detect.ppyolo.preprocess import Compose


class PPYolo(object):
    def __init__(self, module_args=seal_detect_args):
        # load predictor
        self.predictor = ort.InferenceSession(module_args.model_path)
        
        # load infer config
        self.infer_config = PredictConfig(module_args.infer_cfg_path)
        
        # load preprocess transforms
        self.transforms = Compose(self.infer_config.preprocess_infos)
        
        # # 读取模型信息
        # self.classes = list(map(lambda x: x.strip(), open(module_args.label_path, 'r').readlines()))
        # self.score_thresh = module_args.score_thresh
        # self.nms_thresh = module_args.nms_thresh
        # self.mean = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)
        # # 设置推理配置
        # options = ort.SessionOptions()
        # options.log_severity_level = 3
        # # 构建推理会话
        # self.net = ort.InferenceSession(module_args.model_path, options)
        # inputs_name = [a.name for a in self.net.get_inputs()]
        # inputs_shape = {k: v.shape for k, v in zip(inputs_name, self.net.get_inputs())}
        # self.input_shape = inputs_shape['image'][2:]
        # # 绘制参数
        # self.color_list = get_color_map_list(len(self.classes))

    def detect(self, src_img):
        im_info = {
            "im_shape": np.array(
                src_img.shape[:2], dtype=np.float32),
            "scale_factor": np.array(
                [1., 1.], dtype=np.float32)
        }
        inputs = self.transforms(src_img, im_info)
        inputs_name = [var.name for var in self.predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}

        outputs = self.predictor.run(output_names=None, input_feed=inputs)
        
        print("ONNXRuntime predict: ")
        results = []
        if self.infer_config.arch in ["HRNet"]:
            print(np.array(outputs[0]))
        else:
            bboxes = np.array(outputs[0])
            # 过滤可能的背景类
            expect_boxes = bboxes[bboxes[:, 0] > -1]  
            expect_boxes = expect_boxes[expect_boxes[:, 1] > self.infer_config.draw_threshold]  
            results = expect_boxes
        return results

    # def detect_and_draw(self, src_img):
    #     boxes = self.detect(src_img)
    #     for i in range(boxes.shape[0]):
    #         class_id, conf = int(boxes[i, 0]), boxes[i, 1]
    #         x_min, y_min, x_max, y_max = int(boxes[i, 2]), int(boxes[i, 3]), int(boxes[i, 4]), int(boxes[i, 5])
    #         color = tuple(self.color_list[class_id])
    #         cv2.rectangle(src_img, (x_min, y_min), (x_max, y_max), color, thickness=2)
    #         print(self.classes[class_id] + ': ' + str(round(conf, 3)))
    #         cv2.putText(src_img,
    #                     self.classes[class_id] + ':' + str(round(conf, 3)), (x_min, y_min - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.8, (0, 255, 0),
    #                     thickness=2)
    #     return src_img


# if __name__ == '__main__':
#     import os
#     from conf.service_args import project_root
#     from utils.image_utils import read_image_file
#     # 构建检测器
#     net = PPYolo()
    
    
#     # 图像读取
#     test_image_path = r"/Users/rain/Downloads/seal/Images/1.jpeg"
#     test_data = read_image_file(test_image_path)
#     # 预测
#     res_image = net.detect_and_draw(test_data)
#     # 可视化结果
#     res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow("detect_res", res_image)
#     cv2.waitKey(0)
#     # test_dir = r"seal_dir"
#     # for one_name in os.listdir(test_dir):
        
