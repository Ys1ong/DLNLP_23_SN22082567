import os
import time
import pandas as pd

from A import Decision_Tree
from B import MLP
from C import LSTM
from D import BILSTM

start_time_total = time.time()
# Run Task A
train_DT, val_DT, test_DT = Decision_Tree.run_A()

# Run Task B
train_MLP, val_MLP, test_MLP = MLP.run_B()

# Run Task C
train_LSTM, val_LSTM, test_LSTM = LSTM.run_C()

# Run Task D
train_BILSTM, val_BILSTM, test_BILSTM = BILSTM.run_D()

# Summary
print('\n\n******************************Summary******************************')

path = './Results'
if not os.path.exists(path):
    os.makedirs(path)

model_df = pd.DataFrame({
    'DT': [train_DT, val_DT, test_DT],
    'MLP': [train_MLP, val_MLP, test_MLP],
    'LSTM': [train_LSTM, val_LSTM, test_LSTM],
    'BILSTM': [train_BILSTM, val_BILSTM, test_BILSTM]},
    index = ['training', 'validation', 'accuracy']
)

print(model_df)
model_df.to_csv(os.path.join(path, 'model_results.csv'))

test_df = pd.read_csv('Datasets/test.csv')
labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
test_df['description'] = test_df['label'].map(labels_dict)

result_A = pd.read_csv(os.path.join(path, 'A/result_A.csv'))
result_B = pd.read_csv(os.path.join(path, 'B/result_B.csv'))
result_C = pd.read_csv(os.path.join(path, 'C/result_C.csv'))
result_D = pd.read_csv(os.path.join(path, 'D/result_D.csv'))

test_df['pred_DT'] = result_A['pred_DT']
test_df['pred_DT_discription'] = result_A['pred_DT_discription']

test_df['pred_MLP'] = result_B['pred_MLP']
test_df['pred_MLP_discription'] = result_B['pred_MLP_discription']

test_df['pred_LSTM'] = result_C['pred_LSTM']
test_df['pred_LSTM_discription'] = result_C['pred_LSTM_discription']

test_df['pred_BILSTM'] = result_D['pred_BILSTM']
test_df['pred_BILSTM_discription'] = result_D['pred_BILSTM_discription']

test_df.to_csv(os.path.join(path, 'test_results.csv'))

end_time_total = time.time()
print('\nTime for running all is: %.2f' % (end_time_total - start_time_total) + 's')
