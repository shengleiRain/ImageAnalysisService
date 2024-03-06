from core.seal_rec.detect_ppyolo.ppyolo import PPYolo
from libs.base.image_analysis import BaseImageAnalysis
from core.seal_rec.config import seal_rec_args


class SealDetect(BaseImageAnalysis):
    def __init__(self, module_args=seal_rec_args):
        super().__init__(module_args)
        self.seal_det_module = self.init_analysis_module()

    @staticmethod
    def init_analysis_module():
        seal_det_module = PPYolo()
        return seal_det_module

    def analysis(self, image):
        seal_info = []
        # 印章检测
        boxes = self.seal_det_module.detect(image)
        for i in range(boxes.shape[0]):
            x_min, y_min, x_max, y_max = int(boxes[i, 2]), int(boxes[i, 3]), int(boxes[i, 4]), int(boxes[i, 5])
            one_seal_res = {"box": [x_min, y_min, x_max, y_max]}
            seal_info.append(one_seal_res)
        return seal_info

