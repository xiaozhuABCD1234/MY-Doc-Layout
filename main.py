from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import DetectionModel


class MyTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model: DetectionModel = super().get_model(cfg, weights, verbose)
        return model


# 使用自定义 trainer
if __name__ == "__main__":
    trainer = MyTrainer(
        overrides={
            "model": "./yolo26n.yaml",
            "data": "coco128.yaml",
            "optimizer": "SGD",  # 强制使用 SGD
            "lr0": 0.01,
            "lrf": 0.01,  # 最终学习率 = lr0 * lrf = 0.0001
            "momentum": 0.937,  # SGD 动量
            "weight_decay": 0.0005,
            # 训练配置
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "device": 0,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            # 数据增强（关闭 mosaic 过早可能影响小数据集）
            "close_mosaic": 10,  # 最后10轮关闭 mosaic
        }
    )
    trainer.train()

# model = YOLO("/run/media/xiaozhu/移动/yolo/runs/detect/train10/weights/last.pt")