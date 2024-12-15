import os
import numpy as np
import cv2

BASE_PATH = '/home/xhk010905/xhk/data'
    
class ORL(object):
    def __init__(self):
        self.dataset_path = os.path.join(BASE_PATH, 'FaceDB_orl')
        self.image_size = (112, 92)
        self.train_data = []
        self.train_labels = []  # 训练数据的标签
        self.test_data = []
        self.test_labels = []  # 测试数据的标签
        self.mean_face = None

    def load(self, test_size=2):
        folders = os.listdir(self.dataset_path)
        for folder in folders:
            folder_path = os.path.join(self.dataset_path, folder)
            files = os.listdir(folder_path)
            per_folder_file_count = 0
            for file in files:
                file_path = os.path.join(folder_path, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten()
                if per_folder_file_count >= test_size:
                    self.test_data.append(image)
                    self.test_labels.append(int(folder))  # 存储测试样本的标签
                    break
                self.train_data.append(image)
                self.train_labels.append(int(folder))  # 存储训练样本的标签
                per_folder_file_count += 1
        self.train_data = np.array(self.train_data)
        self.test_data = np.array(self.test_data)
        self.mean_face = np.mean(self.train_data, axis=0)

# class ORL(object):
#     def __init__(self):
#         self.dataset_path = os.path.join(BASE_PATH, 'FaceDB_orl')
#         self.image_size = (112, 92)
#         self.train_data = []
#         self.labels = []
#         self.test_data = []
#         self.mean_face = None

#     def load(self, test_size=2):
#         folders = os.listdir(self.dataset_path)
#         for folder in folders:
#             folder_path = os.path.join(self.dataset_path, folder)
#             files = os.listdir(folder_path)
#             per_folder_file_count = 0
#             for file in files:
#                 file_path = os.path.join(folder_path, file)
#                 image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten() # imread读图 flatten展开 我这里用的是行向量 其实是一样的 操作逻辑转置一下就行
#                 if per_folder_file_count >= test_size:
#                     self.test_data.append(image)
#                     break
#                 self.train_data.append(image)
#                 self.labels.append(int(folder))
#                 per_folder_file_count += 1
#         self.train_data = np.array(self.train_data)
#         self.test_data = np.array(self.test_data)
#         self.mean_face = np.mean(self.train_data, axis=0)