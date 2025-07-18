基于yolo-track的车辆的类型和颜色检测





安装所需环境，这个是针对CUDA12.2的，具体cuda版本不同有所影响

```
pip install -r requirements.txt
```

然后可以

```
python vehicle_detection.py -h
```

附带了yolov8s.pt，veri776数据集训练的shufflenet权重（效果不是很好），car-colors-1smyc数据集训练的mobilenet权重（效果还行）



用其它非coco数据集训练的yolo权重，请自行vehicle_detection.py中的yolo字典，color classifier同理

