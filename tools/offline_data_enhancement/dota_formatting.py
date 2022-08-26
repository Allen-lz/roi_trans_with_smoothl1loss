"""
随机旋转--->随机crop--->...
将数据转为dota的文件夹格式

这个脚本中的东西比较有用

"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def vis_landmarks(image, landmarks):
    """
    :return:
    """
    # 可视化
    landmarks_image = image.copy()
    plt.subplot(121), plt.imshow(image)
    for i, landmark in enumerate(landmarks):
        p = np.array(landmark, dtype=np.int32)
        cv2.circle(landmarks_image, p, radius=50, color=(0, 0, 0), thickness=100)
    plt.subplot(122), plt.imshow(landmarks_image)
    plt.show()


def random_rotation(image,
                    landmarks,
                    max_angle=45,
                    scale=1.0):
    """
    https://blog.csdn.net/qq_16564093/article/details/106000209

    之后想要加上bbox的旋转也是很简单的

    box为0~1的值，为[ymin, xmin, ymax, xmax]
    landmarks为0~1的值，为[y0,x0,y1,x1,y2,x2......yn,xn]
    :param image:
    :param box:
    :param landmarks:
    :param max_angle:
    :return:
    """

    # 随机生成一个角度
    angle = np.random.randint(-max_angle, max_angle)
    (h, w) = image.shape[:2]  # 10

    # 将w放在前面
    center = (w // 2, h // 2)  # 11
    scaler = np.stack([w, h], axis=0)

    # 生成一个仿射变换矩阵
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)  # 12
    rotated = cv2.warpAffine(image, M, (w, h))  # 13

    # 将角度转换为弧度, 并手动计算旋转矩阵
    theta = angle * (math.pi / 180.0)
    center = np.reshape(0.5 * scaler, [1, 2])
    rotation = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], axis=0)
    rotation_matrix = np.reshape(rotation, [2, 2])

    # 旋转关键点
    landmarks = np.matmul(landmarks - center, rotation_matrix) + center

    # 将点转化成int32的格式
    landmarks = np.array(landmarks, dtype=np.int32)

    # 可视化
    # vis_landmarks(rotated, landmarks)

    return rotated, landmarks

def random_crop(image, landmarks, jitter_range=0.15):
    """
    :param image:
    :param ratio:
    :param landmarks:
    :param jitter_range:
    :return:
    """
    h, w = image.shape[:2]
    xmin = 0
    ymin = 0
    xmax = w
    ymax = h
    bbox_h = ymax - ymin
    bbox_w = xmax - xmin

    # 对生成的bbox的四个角进行随机抖动(小范围抖动即可, 依旧要保证人脸占框的大部分)
    jitter_xmin = random.randint(0, int(jitter_range * bbox_w))
    jitter_ymin = random.randint(0, int(jitter_range * bbox_h))

    jitter_xmax = random.randint(-int(jitter_range * bbox_w), 0)
    jitter_ymax = random.randint(-int(jitter_range * bbox_h), 0)

    xmin = xmin + jitter_xmin
    xmax = xmax + jitter_xmax
    ymin = ymin + jitter_ymin
    ymax = ymax + jitter_ymax

    xmin = int(max(0, xmin))
    xmax = int(min(w, xmax))

    ymin = int(max(0, ymin))
    ymax = int(min(h, ymax))

    crop_img = image[ymin:ymax, xmin:xmax, :]

    landmarks[:, 0] = landmarks[:, 0] - xmin
    landmarks[:, 1] = landmarks[:, 1] - ymin

    # bbox[:, 0] = bbox[:, 0] - xmin
    # bbox[:, 1] = bbox[:, 1] - ymin
    # bbox[:, 2] = bbox[:, 2] - xmin
    # bbox[:, 3] = bbox[:, 3] - ymin

    # vis_landmarks(crop_img, landmarks)

    return crop_img, landmarks


def motion_blur(image, degree=15, angle=60):

    angle = random.randint(-angle, angle)
    degree = random.randint(2, degree)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


if __name__ == "__main__":
    # python .\tools\offline_data_enhancement\dota_formatting.py

    # name2id = {
    #     "scratch": 0,
    #     "deformation": 1
    # }

    name2id = {
        "bigship": 0
    }

    data_root = r"E:\datasets\detection\cracked_floor_tiles\images_txt"
    images = []
    labels = []

    items = os.listdir(data_root)
    for item in items:
        if item[-3:].lower() == "jpg":
            images.append(item)
        else:
            labels.append(item)

    images.sort()
    labels.sort()

    assert len(labels) == len(images)

    indexes = list(range(len(labels)))

    start_ids = 8368
    max_num = 2000
    max_num = start_ids + max_num
    count = start_ids
    phrase = "test"

    image_root = r"E:\datasets\detection\cracked_floor_tiles\{}\images".format(phrase)
    label_root = r"E:\datasets\detection\cracked_floor_tiles\{}\annfiles".format(phrase)
    yolo_label_root = r"E:\datasets\detection\cracked_floor_tiles\{}\annyolos".format(phrase)
    voc_txt = r"E:\datasets\detection\cracked_floor_tiles\{}.txt".format(phrase)

    if not os.path.exists(image_root):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(image_root)
    if not os.path.exists(label_root):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(label_root)
    if not os.path.exists(yolo_label_root):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(yolo_label_root)
    voc_txt_f = open(voc_txt, "w")

    while count < max_num:
        for index in indexes:
            print(count)
            img_path = os.path.join(data_root, images[index])
            label_path = os.path.join(data_root, labels[index])
            img = cv2.imread(img_path)
            polygons = []
            names = []
            lines = open(label_path, "r").readlines()
            for line in lines[2:]:
                line_list = line.strip().split(" ")[:-2]
                point = list(map(float, line_list))

                point_numpy = np.array(point).reshape(-1, 2)

                polygons.append(point)
                label_name = line.strip().split(" ")[-2]
                if label_name in ["deformatio", "deformation"]:
                    label_name = "deformation"
                names.append(label_name)

            num_bbox = len(polygons)
            polygons = np.array(polygons, dtype=np.int32).reshape(num_bbox, 4, 2)

            # 数据增强
            landmarks = polygons.reshape(-1, 2)
            # random rotate
            img, landmarks = random_rotation(img, landmarks)
            # random crop
            img, landmarks = random_crop(img, landmarks)
            # random blur
            # img = motion_blur(img)

            polygons = landmarks.reshape(num_bbox, 4, 2)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # 从中已经增强过后的数据中得到一个最大的外接矩形(这个部分是用来生成yolo格式的数据集的)
            # 使用的格式是: name_id, x1, y1, x2, y2
            bboxes = []
            for polygon in polygons:
                x1 = np.min(polygon[:, 0])
                y1 = np.min(polygon[:, 1])
                x2 = np.max(polygon[:, 0])
                y2 = np.max(polygon[:, 1])
                bbox = np.array([[x1, y1, x2, y2]])
                bboxes.append(bbox)
            obj_num = polygons.shape[0]
            if obj_num > 0:
                bboxes = np.concatenate(bboxes, axis=0)
                this_yolo_label_txt = open(os.path.join(yolo_label_root, str(count).zfill(4) + ".txt"), "w")
                voc_txt_f.write(str(count).zfill(4) + "\n")

                for index, bbox in enumerate(bboxes):
                    line = " ".join(list(map(str, bbox.tolist())))
                    this_yolo_label_txt.write(line + " ")
                    this_yolo_label_txt.write(str(name2id[names[index]]) + "\n")
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # 新建标注
            obj_num = polygons.shape[0]
            if obj_num > 0:
                # 保存信息
                # save_polygons = polygons.reshape(obj_num, -1)
                # this_label_txt = open(os.path.join(label_root, str(count).zfill(4) + ".txt"), "w")
                # for index, bbox in enumerate(save_polygons):
                #     if index == 0:
                #         this_label_txt.write("imagesource:GoogleEarth\n")
                #         this_label_txt.write("gsd:0.146343590398\n")
                #     line = " ".join(list(map(str, bbox.tolist())))
                #     this_label_txt.write(line + " ")
                #     this_label_txt.write(names[index] + " " + str(name2id[names[index]]) + "\n")
                #
                # # 保存图片
                # image_path = os.path.join(image_root, str(count).zfill(4) + ".jpg")
                # cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # 绘制标注并可视化
                salience_map = np.zeros((img.shape[0], img.shape[1]))
                cv2.fillPoly(salience_map, polygons, 255)
                for bbox in polygons:
                    # 线条绘制
                    cv2.polylines(img, [bbox.reshape(-1, 1, 2)], True, (0, 0, 0), 2)

                plt.subplot(121), plt.imshow(img)
                plt.subplot(122), plt.imshow(salience_map)
                plt.show()

                count += 1
