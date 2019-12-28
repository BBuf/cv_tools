# cv_tools

- yolo-cluster-anchors.py 目标检测算法之YOLO系列算法的Anchor聚类，算法原理和使用方法见：https://blog.csdn.net/just_sort/article/details/103386047 。
- darknet2pb 这个文件夹是将https://github.com/mystic123/tensorflow-yolo-v3 这个darknet2pb的工具支持了深度可分离卷积，即支持以mobilenet做yolo的backbone的转换工具。在mystic123的基础上只改动了yolov3-tiny.py，想自己改哪些卷积层按照我的方式添加和修改就ok了。