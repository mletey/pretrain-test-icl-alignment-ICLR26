import numpy as np
import optax
from theory import *
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from common import *
from trainmini import train
from model.transformer import TransformerConfig
from task.regression_structured import fulltasksampler, finitetasksampler

rho = 0.01
d = int(sys.argv[1]);
alpha = 2; l = int(alpha*d);
tau = 4; n = int(tau*(d**2));
kappas = [0.2, 0.5, 1, 2, 10]
h = d+1;

myname = sys.argv[2] # grab value of $mydir to add results
kappaind = int(sys.argv[3]) # kappa index specified by array
avgind = int(sys.argv[4]) # average index specified by array
kappa = kappas[kappaind]; k = int(kappa*d);
single_index = int(sys.argv[5])

train_power = 0.9
Ctr = np.diag(np.array([(j + 1) ** -train_power for j in range(d)])); Ctr = (Ctr/np.trace(Ctr))*d
# Ctr = np.eye(d)

trainobject = finitetasksampler(d, l, n, k, rho, Ctr, single_index)
testobject_1 = fulltasksampler(d, l, n, rho, Ctr, single_index)
testobject_2 = fulltasksampler(d, l, n, rho, np.diag(spikevalue(d, 0.5, 5)), single_index)

config = TransformerConfig(pos_emb=False, n_hidden=h, n_layers=2, n_mlp_layers=1, pure_linear_self_att=False)
state, hist = train(config, data_iter=iter(trainobject), test_1_iter=iter(testobject_1), test_2_iter=iter(testobject_2), batch_size=16, loss='mse', test_every=100, train_iters=1000, optim=optax.adamw,lr=1e-4)

print('TRAINING DONE',flush=True)
file_path = f'./{myname}/pickles/train-{kappaind}-{avgind}.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)

loss_func = optax.squared_error
numsamples = 500
testobject = fulltasksampler(d, l, n, rho, Ctr, single_index)
tracker = []
for _ in range(numsamples):
    xs, labels = next(testobject); # generates data
    logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
    tracker.append(loss_func(logits, labels).mean())
tracker = np.array(tracker)

print('DONE: TESTING ON PRETRAIN')

file_path = f'./{myname}/test_equals_train_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {np.mean(tracker)}],')
    file_path = f'./{myname}/test_equals_train_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {np.std(tracker)}],')

test_powers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
# # test_powers = [0.05,0.1,0.2,0.4,0.6,0.8,1,1.2]
# test_powers = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]
power_test_m = []
power_test_s = []
for test_power in test_powers:
    Ctest = np.diag(np.array([(j + 1) ** -test_power for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
    testobject = fulltasksampler(d, l, n, rho, Ctest, single_index)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    power_test_m.append(np.mean(tracker))
    power_test_s.append(np.std(tracker))

file_path = f'./{myname}/test_powers_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {power_test_m}],')
    file_path = f'./{myname}/test_powers_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {power_test_s}],')

print('DONE: TESTING ON POWERS')

expowers = np.linspace(train_power - 0.5, train_power + 0.5, 11)
exp_test_m = []
exp_test_s = []
for expower in expowers:
    Ctest = np.diag(np.array([np.exp(-expower*(j+1)) for j in range(d)])); Ctest = (Ctest/np.trace(Ctest))*d
    testobject = fulltasksampler(d, l, n, rho, Ctest, single_index)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    exp_test_m.append(np.mean(tracker))
    exp_test_s.append(np.std(tracker))

file_path = f'./{myname}/test_exps_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {exp_test_m}],')
file_path = f'./{myname}/test_exps_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {exp_test_s}],')

print('DONE: TESTING ON EXPONENTIAL')

rankfs = [0.2,0.4,0.6,0.8,1]
rank_test_m = []
rank_test_s = []
for f in rankfs:
    Ctest = np.diag(complexity_class_covariance(d, int(d*f), True))
    testobject = fulltasksampler(d, l, n, rho, Ctest, single_index)
    tracker = []
    for _ in range(numsamples):
        xs, labels = next(testobject); # generates data
        logits = state.apply_fn({'params': state.params}, xs); # runs xs through transformer and makes predictions
        tracker.append(loss_func(logits, labels).mean())
    tracker = np.array(tracker)
    rank_test_m.append(np.mean(tracker))
    rank_test_s.append(np.std(tracker))

file_path = f'./{myname}/test_ranks_m_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {rank_test_m}],')
file_path = f'./{myname}/test_ranks_s_{avgind}.txt'
with open(file_path, 'a') as file:
    file.write(f'[{kappaind}, {rank_test_s}],')

print('DONE: TESTING ON LOWER RANKS')

