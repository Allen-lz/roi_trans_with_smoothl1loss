"""
有一些数据通过增强之后是没有标注的，还有可能就是标注在图片外面
"""
import os

if __name__ == "__main__":
    root = r"E:\datasets\detection\cracked_floor_tiles"

    trainval_dir = os.path.join(root, "trainval", "annfiles")
    test_dir = os.path.join(root, "test", "annfiles")

    train_sample_list = os.listdir(trainval_dir)
    test_sample_list = os.listdir(test_dir)

    for name in train_sample_list:
        f = open(os.path.join(trainval_dir, name), "r")
        lines = f.readlines()[2:]
        if len(lines) < 1:
            print(name)

    for name in test_sample_list:
        f = open(os.path.join(test_dir, name), "r")
        lines = f.readlines()[2:]
        if len(lines) < 1:
            print(name)
