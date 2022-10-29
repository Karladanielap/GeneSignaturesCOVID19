import os
import sys

sys.path.insert(0, '..')
from lib import models, graph, coarsening, utils
from sklearn.model_selection import RandomizedSearchCV
import scipy.sparse
import scipy.spatial.distance
import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
import h5py
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 1, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')  # Kipf renoralization tech
flags.DEFINE_integer('coarsening_levels', 1, 'Number of coarsened graphs.')  # number of GCN layers

t_start = time.process_time()
# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')
# For PPI and PPI-singleton model change file location
# Adjacency Matrix
# test = sio.loadmat('D:/Covid19/GCN_Covid19/adj_genemania.mat')   # row/col/value of input graph

# with h5py.File('D:/Covid19/RemoveCol/input.mat','r') as f:
with h5py.File('D:/Covid19/gene-cell-graph/cell-types/GeneMANIA/GM_graph.mat', 'r') as f:
    row = f['row'][:]
    col = f['col'][:]
    value = f['val'][:]
print(row, col, value)
print(row.shape, col.shape, value.shape)
# for Correlaton model change file location

# formatting of input graph
# row = test['row'].astype(np.float32)
# col = test['col'].astype(np.float32)
# value = test['val'].astype(np.float32)
M, k = row.shape
row = np.array(row)
row = row.reshape(k)
row = row.ravel()
col = np.array(col)
col = col.reshape(k)
col = col.ravel()
value = np.array(value)
value = value.reshape(k)
value = value.ravel()
print(row, col, value)
print(row.dtype, col.dtype, value.dtype)
print(row.shape, col.shape, value.shape)

## coarsening of graph
A = scipy.sparse.coo_matrix((value, (row, col)),
                            shape=(2000, 2000))  # change size for model being used 4444 for both PPI and 3866 for
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=True)
L = [graph.laplacian(A, normalized=True, renormalized=True) for A in graphs]
# del test
del A
del row
del col
del value

# Cli = sio.loadmat('E:/GCN_Surv/survivial/TCGA_COAD/Data/Clinical0.mat')
# OLD WAY to import .mat file data (MATLAB version V7.0)
# Dat = sio.loadmat('D:/Covid19/GCN_Covid19/Concatenate_Matrix.mat')
# Lab = sio.loadmat('D:/Covid19/GCN_Covid19/Labels.mat')
# Data_L = Dat['concat_mtx'].astype(np.float32)
# Labels = Lab['Labels'].astype(np.float32)

# NEW WAY to import .mat file data (MATLAB version V7.3)
# with h5py.File('D:/Covid19/RemoveCol/Seurat_SampleXGene.mat','r') as f:
with h5py.File('D:/Covid19/gene-cell-graph/new-cell-types/M1M2Macrophage/M1M2Macrophage_GeneXCell.mat','r') as f:
# with h5py.File('D:/Covid19/GCN Process/cmp_cnn_gcn/Input Matrices/SampleXGene_removed6.mat','r') as f:
    Data_L = np.transpose(f['M1M2Macrophage_GeneXCell'][:])
# with h5py.File('D:/Covid19/GCN Process/cmp_cnn_gcn/Input Matrices/Patient6.mat','r') as f:
#     P6 = np.transpose(f['Patient6'][:])
# with h5py.File('D:/Covid19/GCN Process/Seurat_batch_normalized/3_Labeling/Labels.mat','r') as f:
# # with h5py.File('D:/Covid19/GCN Process/cmp_cnn_gcn/Labels.mat', 'r') as f:
#     Labels = np.transpose(f['Labels'][:])
#     Labels_p6 = np.transpose(f['Labels_p6'][:])



# Data_L = np.transpose(Data_L)
# print(Clinical.shape)
# print(Data_L.shape, Data_L.shape[0], Labels.shape)
# print("I Make it here")

### Remember change Loops to 15 when using partitions
Out = 3  # change label to 3 for covid
common = {}
common['dir_name'] = 'KD/'
common['num_epochs'] = 60  # original 20
common['batch_size'] = 280    # increase 2000~250 CPU hate 32's multiple, GPU like it
common['decay_steps'] = 17.7  # * common['num_epochs'] since not used use as in momentum
common['eval_frequency'] = 2 * common['num_epochs']
common['brelu'] = 'b1relu'
common['pool'] = 'apool1'  # try 'mpool1', original'apool1'

model_perf = utils.model_perf()

common['regularization'] = 0  # 0.1 ~ 0.0001
common['dropout'] = 0.7  # 0.5 ~ 0.9
common['learning_rate'] = 0.005  # original 0.005
common['decay_rate'] = 0.95
common['momentum'] = 0

# common['Clin'] = 1  # 6
common['F'] = [1]
common['K'] = [1]
common['p'] = [2]
common['M'] = [128, Out]  # try 256 or lower


### Use the 15 preset partitions
# with h5py.File('D:/Covid19/GCN Process/Seurat_batch_normalized/4_Partitions/15_Partitions.mat','r') as f:
#     training_15part = np.transpose(f['training_15_part'][:])
#     validating_15part = np.transpose(f['validating_15_part'][:])
# with h5py.File('D:/Covid19/GCN Process/Seurat_batch_normalized/4_Partitions/testing_partition.mat','r') as f:
#     testing_part = np.transpose(f['shuffled_testing_part'][:])

###
# print("15 Partition Dimensions: ")
# print(training_15part, training_15part.shape)
# print(validating_15part, validating_15part.shape)
# print("The one and only testing partition: ")
# print(testing_part, testing_part.shape)

### Create Partition inspired by GNN ##
Loops = 15
for x in range(Loops, Loops+1):
#for x in range(11, 12):
#for x in range(0, Loops):
    num_cell = np.arange(Data_L.shape[0])
    print(num_cell)
    ##############################1      2    3     5      6     7     8     9     10     11     12     13
    split_cell = np.split(num_cell, [865, 1634, 2029, 2782, 3468, 3563, 9704, 14087, 14359, 15028, 15597])
    #split_cell = np.array(split_cell)
    print(split_cell[0].shape,
          split_cell[1].shape,
          split_cell[2].shape,
          split_cell[3].shape,
          split_cell[4].shape,
          split_cell[5].shape,
          split_cell[6].shape,
          split_cell[7].shape,
          split_cell[8].shape,
          split_cell[9].shape,
          split_cell[10].shape,
          split_cell[11].shape)

    # Randomly shuffled the indices for labels and expressions
    np.random.seed(x)
    np.random.shuffle(split_cell[0])
    n1 = Data_L[split_cell[0]]
    np.random.seed(x)
    np.random.shuffle(split_cell[1])
    n2 = Data_L[split_cell[1]]
    np.random.seed(x)
    np.random.shuffle(split_cell[2])
    n3 = Data_L[split_cell[2]]
    np.random.seed(x)
    np.random.shuffle(split_cell[3])
    m1 = Data_L[split_cell[3]]
    np.random.seed(x)
    np.random.shuffle(split_cell[4])
    m2 = Data_L[split_cell[4]]
    np.random.seed(x)
    np.random.shuffle(split_cell[5])
    m3 = Data_L[split_cell[5]]
    np.random.seed(x)
    np.random.shuffle(split_cell[6])
    s1 = Data_L[split_cell[6]]
    np.random.seed(x)
    np.random.shuffle(split_cell[7])
    s2 = Data_L[split_cell[7]]
    np.random.seed(x)
    np.random.shuffle(split_cell[8])
    s3 = Data_L[split_cell[8]]
    np.random.seed(x)
    np.random.shuffle(split_cell[9])
    s4 = Data_L[split_cell[9]]
    np.random.seed(x)
    np.random.shuffle(split_cell[10])
    s5 = Data_L[split_cell[10]]
    np.random.seed(x)
    np.random.shuffle(split_cell[11])
    s6 = Data_L[split_cell[11]]
    print(n1.shape,
          n2.shape,
          n3.shape,
          m1.shape,
          m2.shape,
          m3.shape,
          s1.shape,
          s2.shape,
          s3.shape,
          s4.shape,
          s5.shape,
          s6.shape)

    n1_TRAIN_SPLIT = int(0.8 * n1.shape[0])
    n1_TEST_SPLIT = int(0.1 * n1.shape[0] + n1_TRAIN_SPLIT)
    print(n1_TRAIN_SPLIT, n1_TEST_SPLIT)
    n1_train, n1_test, n1_validate = np.split(split_cell[0], [n1_TRAIN_SPLIT, n1_TEST_SPLIT])
    print("This is N1: ", n1_train.shape, n1_test.shape, n1_validate.shape)

    n2_TRAIN_SPLIT = int(0.8 * n2.shape[0])
    n2_TEST_SPLIT = int(0.1 * n2.shape[0] + n2_TRAIN_SPLIT)
    print(n2_TRAIN_SPLIT, n2_TEST_SPLIT)
    n2_train, n2_test, n2_validate = np.split(split_cell[1], [n2_TRAIN_SPLIT, n2_TEST_SPLIT])
    print("This is N2: ", n2_train.shape, n2_test.shape, n2_validate.shape)

    n3_TRAIN_SPLIT = int(0.8 * n3.shape[0])
    n3_TEST_SPLIT = int(0.1 * n3.shape[0] + n3_TRAIN_SPLIT)
    print(n3_TRAIN_SPLIT, n3_TEST_SPLIT)
    n3_train, n3_test, n3_validate = np.split(split_cell[2], [n3_TRAIN_SPLIT, n3_TEST_SPLIT])
    print("This is N3: ", n3_train.shape, n3_test.shape, n3_validate.shape)

    m1_TRAIN_SPLIT = int(0.8 * m1.shape[0])
    m1_TEST_SPLIT = int(0.1 * m1.shape[0] + m1_TRAIN_SPLIT)
    print(m1_TRAIN_SPLIT, m1_TEST_SPLIT)
    m1_train, m1_test, m1_validate = np.split(split_cell[3], [m1_TRAIN_SPLIT, m1_TEST_SPLIT])
    print("This is M1: ", m1_train.shape, m1_test.shape, m1_validate.shape)

    m2_TRAIN_SPLIT = int(0.8 * m2.shape[0])
    m2_TEST_SPLIT = int(0.1 * m2.shape[0] + m2_TRAIN_SPLIT)
    print(m2_TRAIN_SPLIT, m2_TEST_SPLIT)
    m2_train, m2_test, m2_validate = np.split(split_cell[4], [m2_TRAIN_SPLIT, m2_TEST_SPLIT])
    print("This is M2: ", m2_train.shape, m2_test.shape, m2_validate.shape)

    m3_TRAIN_SPLIT = int(0.8 * m3.shape[0])
    m3_TEST_SPLIT = int(0.1 * m3.shape[0] + m3_TRAIN_SPLIT)
    print(m3_TRAIN_SPLIT, m3_TEST_SPLIT)
    m3_train, m3_test, m3_validate = np.split(split_cell[5], [m3_TRAIN_SPLIT, m3_TEST_SPLIT])
    print("This is M3: ", m3_train.shape, m3_test.shape, m3_validate.shape)

    s1_TRAIN_SPLIT = int(0.8 * s1.shape[0])
    s1_TEST_SPLIT = int(0.1 * s1.shape[0] + s1_TRAIN_SPLIT)
    print(s1_TRAIN_SPLIT, s1_TEST_SPLIT)
    s1_train, s1_test, s1_validate = np.split(split_cell[6], [s1_TRAIN_SPLIT, s1_TEST_SPLIT])
    print("This is S1: ", s1_train.shape, s1_test.shape, s1_validate.shape)

    s2_TRAIN_SPLIT = int(0.8 * s2.shape[0])
    s2_TEST_SPLIT = int(0.1 * s2.shape[0] + s2_TRAIN_SPLIT)
    print(s2_TRAIN_SPLIT, s2_TEST_SPLIT)
    s2_train, s2_test, s2_validate = np.split(split_cell[7], [s2_TRAIN_SPLIT, s2_TEST_SPLIT])
    print("This is S2: ", s2_train.shape, s2_test.shape, s2_validate.shape)

    s3_TRAIN_SPLIT = int(0.8 * s3.shape[0])
    s3_TEST_SPLIT = int(0.1 * s3.shape[0] + s3_TRAIN_SPLIT)
    print(s3_TRAIN_SPLIT, s3_TEST_SPLIT)
    s3_train, s3_test, s3_validate = np.split(split_cell[8], [s3_TRAIN_SPLIT, s3_TEST_SPLIT])
    print("This is S3: ", s3_train.shape, s3_test.shape, s3_validate.shape)

    s4_TRAIN_SPLIT = int(0.8 * s4.shape[0])
    s4_TEST_SPLIT = int(0.1 * s4.shape[0] + s4_TRAIN_SPLIT)
    print(s4_TRAIN_SPLIT, s4_TEST_SPLIT)
    s4_train, s4_test, s4_validate = np.split(split_cell[9], [s4_TRAIN_SPLIT, s4_TEST_SPLIT])
    print("This is S4: ", s4_train.shape, s4_test.shape, s4_validate.shape)

    s5_TRAIN_SPLIT = int(0.8 * s5.shape[0])
    s5_TEST_SPLIT = int(0.1 * s5.shape[0] + s5_TRAIN_SPLIT)
    print(s5_TRAIN_SPLIT, s5_TEST_SPLIT)
    s5_train, s5_test, s5_validate = np.split(split_cell[10], [s5_TRAIN_SPLIT, s5_TEST_SPLIT])
    print("This is S5: ", s5_train.shape, s5_test.shape, s5_validate.shape)

    s6_TRAIN_SPLIT = int(0.8 * s6.shape[0])
    s6_TEST_SPLIT = int(0.1 * s6.shape[0] + s6_TRAIN_SPLIT)
    print(s6_TRAIN_SPLIT, s6_TEST_SPLIT)
    s6_train, s6_test, s6_validate = np.split(split_cell[11], [s6_TRAIN_SPLIT, s6_TEST_SPLIT])
    print("This is S6: ", s6_train.shape, s6_test.shape, s6_validate.shape)

    Train_Indices = np.concatenate([n1_train, n2_train, n3_train, m1_train, m2_train, m3_train, s1_train, s2_train, s3_train, s4_train, s5_train, s6_train])
    np.random.seed(x)
    np.random.shuffle(Train_Indices)
    print("Training set size: ", Train_Indices.shape)

    Test_Indices = np.concatenate([n1_test, n2_test, n3_test, m1_test, m2_test, m3_test, s1_test, s2_test, s3_test, s4_test, s5_test, s6_test])
    np.random.seed(x)
    np.random.shuffle(Test_Indices)
    print("Testing set size: ", Test_Indices.shape)

    Validate_Indices = np.concatenate([n1_validate, n2_validate, n3_validate, m1_validate, m2_validate, m3_validate, s1_validate, s2_validate, s3_validate, s4_validate, s5_validate, s6_validate])
    np.random.seed(x)
    np.random.shuffle(Validate_Indices)
    print("Validating set size: ", Validate_Indices.shape)

    print('Verify with the sum of cells: ', Train_Indices.shape[0]+Test_Indices.shape[0]+Validate_Indices.shape[0])

    ##### Labels ########
    df = pd.read_excel (r'D:/Covid19/gene-cell-graph/new-cell-types/M1M2Macrophage/Celltype_M1M2Macrophage.xlsx')
    df2 = df.to_numpy()
    print(df)
    Train_Labels = df2[Train_Indices, 4].astype(np.float32)
    #print(Train_Labels, Train_Labels.shape)
    Test_Labels = df2[Test_Indices, 4].astype(np.float32)
    #print(Test_Labels, Test_Labels.shape)
    Val_Labels = df2[Validate_Indices, 4].astype(np.float32)
    #print(Val_Labels, Val_Labels.shape)

    training_idx = Train_Indices
    validating_idx = Validate_Indices
    #print("This is", x, "parition for training: ", training_idx, training_idx.shape)
    #print("This is", x, "parition for validating: ",validating_idx, validating_idx.shape)

    testing_idx = Test_Indices
    #print("This is testing partition: ", testing_idx, testing_idx.shape)

    Train_Data = coarsening.perm_data(Data_L[training_idx, :], perm)
    Val_Data = coarsening.perm_data(Data_L[validating_idx, :], perm)
    Test_Data = coarsening.perm_data(Data_L[testing_idx, :], perm)
    # Train_Labels = Labels[training_idx, :]
    # Train_Labels = Train_Labels.ravel()
    # Val_Labels = Labels[validating_idx, :]
    # Val_Labels = Val_Labels.ravel()
    # Test_Labels = Labels[testing_idx, :]
    # Test_Labels = Test_Labels.ravel()

    print(Train_Labels, Train_Labels.shape, '\n', Val_Labels, Val_Labels.shape)
    print(Test_Labels, Test_Labels.shape)
    # print(np.sum(Train_Labels[:, ]))  # Train_Labels[:, 2]
    # print(np.sum(Val_Labels[:, ]))  # Train_Labels[:, 2]
    # print(np.median(Train_Labels[:, ]))  # Train_Labels[:, 1]
    # print(np.median(Val_Labels[:, ]))  # Train_Labels[:, 1]
    print('Loop', x)

    # Below is grid search approach
    # if True:
    #     name = 'Run_{}'.format(x)
    #     params = common.copy()
    #     params['dir_name'] += name
    #     params['filter'] = 'chebyshev5'
    #     params['brelu'] = 'b1relu'
    #     print(" Im here")
    #     # grid_params = {'dropout': [0.5],
    #     #                'learning_rate': [.0005, .005]}
    #     grid_params = {'num_epochs':[30,40,50,60,70],
    #                    'batch_size':[104,208,304],
    #                    'pool':['apool1', 'mpool1'],
    #                    'dropout': [0.5, 0.6, 0.7],
    #                    'learning_rate': [.00001, .0005, .001, .005],
    #                    'M':[[64, Out],[128, Out],[256, Out]]}
    #     data = (Train_Data, Train_Labels, Val_Data, Val_Labels, Test_Data, Test_Labels)
    #     utils.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L, **x))

    if True:
        name = 'Run_{}'.format(x)
        params = common.copy()
        params['dir_name'] += name
        params['filter'] = 'chebyshev5'
        params['brelu'] = 'b1relu'
        # params['Clin'] = 5
        model_perf.test(models.cgcnn(L, **params), name, params, Train_Data, Train_Labels, Val_Data, Val_Labels, Test_Data, Test_Labels, KD=False)
        # model_perf.Surv(models.cgcnn(L, **params), name, params, Train_Data, Train_Labels, Val_Data, Val_Labels,
        # Test_Data, Test_Labels, Train_Clin, Val_Clin)


model_perf.show()
