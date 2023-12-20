import numpy as np
import h5py
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm

"""

data_files = ['./test/16930.h5', './test/16931.h5', './test/16932.h5',
              './test/16933.h5', './test/16934.h5', './test/16935.h5',
              './test/16936.h5', './test/16937.h5', './test/16938.h5',
              './test/16939.h5', './test/16940.h5', './test/16941.h5',
              './test/16942.h5', './test/16943.h5', './test/16944.h5',
              './test/16945.h5', './test/16946.h5', './test/16947.h5',
              './test/16948.h5', './test/16949.h5']
              
"""

data_files = ['./test/16930.h5', './test/16931.h5', './test/16932.h5']



PETruth_list = []
ParticleTruth_list = []
event_offset = 0  # 初始化事件偏移量

for file_path in data_files:
    with h5py.File(file_path, 'r') as data:
        # 查看数据集
        print(data.keys())
        print(data['PETruth'].dtype)
        print(data['ParticleTruth'].dtype)

        # 将文件中的表读入内存（对更大的数据慎用，有炸内存风险）
        PETruth_temp = data['PETruth'][:]
        ParticleTruth_temp = data['ParticleTruth'][:]

        # 为当前文件的数据添加事件偏移量
        PETruth_temp['EventID'] += event_offset
        ParticleTruth_temp['EventID'] += event_offset

        # 将当前文件的数据添加到总数组中
        PETruth_list.append(PETruth_temp)
        ParticleTruth_list.append(ParticleTruth_temp)

        # 更新事件偏移量
        event_offset += len(np.unique(PETruth_temp['EventID']))


# 使用 numpy.concatenate 整合数据
PETruth = np.concatenate(PETruth_list, axis=0)
ParticleTruth = np.concatenate(ParticleTruth_list, axis=0)

event_count = len(ParticleTruth)
print(f'Event Count: {event_count}')

def get_feature(dataset,event_counts):
# 数据中提取特征，打包成函数，使之既可以处理数据集，也可以处理题目


# 建立数据特征，这里的特征在 dtype 中定义
    feature_dtype = [
        ('PECount', 'i4'), # 总 PE 数，32位整数
        ('PEMean', 'f4'), # 被点亮的 PMT 数，32位整数
        ('PETimeMean', 'f4'), # 各PE到达时间的平均值
        ('PETimeStd', 'f4'), # 各PE的到达时间的标准差
    ]
# 创建空数组，以后再填入数据
    feature = np.zeros(event_counts, dtype=feature_dtype)

    ############################################### # 使用每个事件在 `PETruth` 中出现的次数，得到 PECount
    # 同时由于数据排好序，为方便后面处理每个事件，我们获取各事件索引
    event_id, event_start_index, event_pe_count = np.unique(
    dataset['EventID'],
        return_index=True,
        return_counts=True
        )
    feature['PECount'] = event_pe_count
    event_end_index = np.append(event_start_index[1:], len(PETruth))


    # 其他特征可能不方便向量化，则逐个事件处理
    # 可以用 for，也可以多进程加速
    for start_index, end_index in tqdm(zip(event_start_index, event_end_index)):
        event = dataset[start_index:end_index]
        i = event[0]['EventID']

        # 平均每个被点亮的 PMT 获得的光子数
        event_pmt_id, event_pmt_pe_count = np.unique(event['ChannelID'], return_counts=True)
        feature[i]['PEMean'] = np.mean(event_pmt_pe_count)

        # PE 到达的平均时间
        event_pe_mean_time = np.mean(event['PETime'])
        feature[i]['PETimeMean'] = event_pe_mean_time

        # PE 到达时间的标准差
        event_pe_std_time = np.std(event['PETime'])
        feature[i]['PETimeStd'] = event_pe_std_time

    return feature

feature = get_feature(PETruth,event_count)
# print(feature[:10])
# print(feature.dtype)

# 获取标签
# 获取动能 Ek 和可见能量 Evis
Ek = ParticleTruth['Ek']
Evis = ParticleTruth['Evis']

# 展示每个特征和标签的关系
"""
for name in feature.dtype.names:
    plt.scatter(
        x=feature[name],
        y=Ek,
        s=1
    )
    plt.title(f'{name} - Ek')
    plt.show()

"""

def resolution_obj(pred, dataset: lgb.Dataset):
    """Obj function for LGBM. Use `resolution` for evaluation."""
    truth = dataset.label
    grad = pred/truth - 1
    hess = 1 / truth
    return grad, hess

def resolution_loss(pred, dataset: lgb.Dataset):
    """Loss (Eval) function for LGBM. Use `resolution` for evaluation."""
    truth = dataset.label
    residual = pred - truth
    loss = np.sqrt(np.mean(residual**2 / truth))
    return 'resolution', loss, False

from sklearn.model_selection import train_test_split # 用于切分训练集和测试集
def lgb_train(feature, label):
    # 切分训练集和测试集，默认是 0.75:0.25
    train_feat, test_feat, train_label, test_label = train_test_split(
        feature, label,
        shuffle=True, random_state=42, test_size=.1
    )
        # 构建 Light GBM 格式的数据集
    train_lgb = lgb.Dataset(train_feat, train_label)
    test_lgb = lgb.Dataset(test_feat, test_label)

    # 设定参数，参数含义具体看文档
    lgb_params = dict(
        verbose = -1,   # 控制输出
        learning_rate = 0.001     # 控制学习速率；越大学习越快，但是精度差
    )
    lgb_kw = dict(
        num_boost_round = 10000,     # 最大学习轮数
        callbacks = [       # 回调函数列表
            # 早停轮数，即若干轮效果不再提升后停止优化
            lgb.early_stopping(stopping_rounds=100),
            # 输出步长
        ],
        fobj = resolution_obj,        # 目标函数，需返回梯度和 Hessian
        feval = resolution_loss      # 损失函数
    )

    # 进行训练
    model = lgb.train(
        params = lgb_params,
        train_set = train_lgb,       # 训练集
        valid_sets = [train_lgb, test_lgb],  # 测试集，也把训练集包含进去
        ** lgb_kw
     )

    return model

from numpy.lib.recfunctions import structured_to_unstructured

# LightGBM 的特征必须是二维数组，需要将结构化数组转成二维数组
feat2d = structured_to_unstructured(feature)
print('Training Ek...')
Ek_model = lgb_train(feat2d, Ek)
print('--------------')
print('Training Evis...')
Evis_model = lgb_train(feat2d, Evis)

"""
# 残差~动能
plt.scatter(
    Ek, Ek_model.predict(feat2d)-Ek, s=1
)
plt.xlabel('Ek Truth [MeV]')
plt.ylabel('Ek Predict - Ek Truth [MeV]')
plt.show()
# 残差~位置（半径）
x = ParticleTruth['x']
y = ParticleTruth['y']
z = ParticleTruth['z']
plt.scatter(
    (x**2+y**2+z**2)**2/1e6, Ek_model.predict(feat2d)-Ek, s=1
)
plt.xlabel('R^2 [m^2]')
plt.ylabel('Ek Predict - Ek Truth [MeV]')
plt.show()

"""

problem_file = './problem/problem.h5'
with h5py.File(problem_file,'r') as problem:
    PETruth_problem = problem['PETruth'][:]

# 如法炮制，读出问题数据集中的事件编号、获取特征并转化为二维数组
event_id_problem = np.unique(PETruth_problem['EventID'])
feature_problem = get_feature(PETruth_problem,10000)
feat2d_problem = structured_to_unstructured(feature_problem)

# 使用训练出的模型进行预测
Ek_predict = Ek_model.predict(feat2d_problem)
Evis_predict = Evis_model.predict(feat2d_problem)

# 将预测结果按要求拼成结构化数组
answer_dtype = np.dtype([
    ('EventID', '<i4'),
    ('Ek', '<f4'),
     ('Evis', '<f4')
])
answer_array = np.zeros(len(event_id_problem), dtype=answer_dtype)  # 先生成空数组
answer_array['EventID'] = event_id_problem          # 再将数据填入
answer_array['Ek'] = Ek_predict
answer_array['Evis'] = Evis_predict
# 写入答案数据集, 'w' 表示写入
with h5py.File('./answer/answer-lgbm.h5', 'w') as answer_data:
    # 按要求创建 Answer 数据集，并保存答案
    answer_data.create_dataset(name = 'Answer', data=answer_array)



