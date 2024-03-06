import os
from libs.base.analysis_args import AnalysisArgs
from conf.service_args import project_root


def init_detect_args():
    args = AnalysisArgs()
    args.model_path = os.path.join(project_root, "data/seal/detect/ppyolo_seal_det.onnx")
    args.infer_cfg_path = os.path.join(project_root, "data/seal/detect/infer_cfg.yml")
    return args


# 初始化参数,使用需要用import导入，防止不必要的二次复制 例如 import **.config.**_args as **_args
seal_detect_args = init_detect_args()


if __name__ == "__main__":
    print(seal_detect_args)
