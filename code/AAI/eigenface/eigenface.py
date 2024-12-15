# import os
# import numpy as np
# import matplotlib.pyplot as plt
# # import cv2
# from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances

# from orl import ORL

# BASE_PATH = '/home/xhk010905/xhk/data'
# ORL_PATH = os.path.join(BASE_PATH, 'FaceDB_orl')
# TEST_SIZE = 2


# if __name__ == '__main__':
# # def eigenface():
#     # 读取 FaceDB_orl 数据集
#     orl = ORL()
#     orl.load()
#     # 中心化训练集
#     c_train = orl.train_data - orl.mean_face
#     # 计算特征脸 (取 n_components=80，即 80 个特征向量)
#     pca = PCA(n_components=80)
#     pca.fit(c_train)
#     eigenfaces = pca.components_.reshape((80, *orl.image_size))
#     # 对训练集进行投影
#     p_train = pca.transform(c_train)
#     # 中心化测试集
#     c_test = orl.test_data - orl.mean_face
#     # 对测试集进行投影
#     p_test = pca.transform(c_test)
#     # 计算测试集与训练集每一对投影点之间的距离
#     dists = pairwise_distances(p_test, p_train)
#     # 输出测试结果并绘制对比图
#     test_index = 0
#     for dist in dists:
#         n_index = np.argmin(dist)
#         n_label = orl.labels[n_index]
#         n_face = orl.train_data[n_index]
#         print(f"""{test_index}:
#         Label: {n_label}
#         Nearest Distance: {np.min(dist)}""")
#         plt.figure()
#         plt.title(f'Test Face and Nearest Face - {test_index}')
#         plt.axis('off')
#         plt.subplot(121)
#         plt.imshow(orl.test_data[test_index].reshape(orl.image_size), cmap='gray')
#         plt.axis('off')
#         plt.subplot(122)
#         plt.imshow(n_face.reshape(orl.image_size), cmap='gray')
#         plt.axis('off')
#         plt.savefig(f'{os.path.join(BASE_PATH, 'temp')}/{test_index}.png')
#         plt.close()
#         test_index += 1
import os
import numpy as np
import matplotlib.pyplot as plt
# import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from xhk.code.AAI.eigenface.orl import ORL

BASE_PATH = '/home/xhk010905/xhk/data'
ORL_PATH = os.path.join(BASE_PATH, 'FaceDB_orl')
TEST_SIZE = 2

if __name__ == '__main__':
    # 读取 FaceDB_orl 数据集
    orl = ORL()
    orl.load()
    
    # 中心化训练集
    c_train = orl.train_data - orl.mean_face
    
    # 设置 PCA 的主成分数量
    n_components = 80  # 选择一个足够大的值
    pca = PCA(n_components=n_components)
    pca.fit(c_train)
    eigenfaces = pca.components_.reshape((n_components, *orl.image_size))
    
    # 累计方差贡献率计算
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print("Cumulative Explained Variance for different k values:")
    for k in [20, 50, 80]:
        if k <= n_components:
            print(f"k={k}, Variance={explained_variance[k-1]:.4f}")
        else:
            print(f"k={k} exceeds the PCA component limit ({n_components}).")
    
    # 对训练集进行投影
    p_train = pca.transform(c_train)
    
    # 中心化测试集
    c_test = orl.test_data - orl.mean_face
    
    # 对测试集进行投影
    p_test = pca.transform(c_test)
    
    metric = 'euclidean' # 欧氏距离
    # metric = 'cosine'    # 余弦距离
    # metric = 'manhattan' # 曼哈顿距离
    # metric = 'chebyshev' # 切比雪夫距离
    
    # 计算测试集与训练集每一对投影点之间的距离
    dists = pairwise_distances(p_test, p_train, metric = metric) 

    
    # 初始化统计变量
    correct_count = 0
    total_test_samples = len(p_test)
    
    # 输出测试结果并绘制对比图
    for test_index, dist in enumerate(dists):
        # 找到距离最近的训练样本的索引和标签
        n_index = np.argmin(dist)
        predicted_label = orl.train_labels[n_index]  # 预测的标签
        true_label = orl.test_labels[test_index]     # 真实的标签
        
        # 判断预测是否正确
        if predicted_label == true_label:
            correct_count += 1
        
        # 打印匹配结果
        print(f"""{test_index}:
        True Label: {true_label}
        Predicted Label: {predicted_label}
        Nearest Distance: {np.min(dist)}""")
        
        # 绘制测试样本与匹配样本对比
        n_face = orl.train_data[n_index]
        plt.figure()
        plt.title(f'True Label and Predicted Label - {test_index}')
        plt.axis('off')
        plt.subplot(121)
        plt.imshow(orl.test_data[test_index].reshape(orl.image_size), cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(n_face.reshape(orl.image_size), cmap='gray')
        plt.axis('off')
        plt.savefig(f'{os.path.join(BASE_PATH, "temp")}/{test_index}.png')
        plt.close()
    
    # 计算识别准确率
    accuracy = correct_count / total_test_samples * 100
    print(f"\n度量距离: {metric}")
    print(f"Recognition Accuracy: {accuracy:.2f}%")