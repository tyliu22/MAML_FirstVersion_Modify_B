from __future__ import absolute_import, division, print_function
import numpy as np
from subprocess import call
import argparse
import pickle, os, timeit, time
import matplotlib.pyplot as plt
from datetime import datetime
from Utils.common import load_run_data



base_run_name = 'PermutedLabels_TasksN'

min_n_tasks = 10  # 1
max_n_tasks = 10  # 10

run_experiments = True # If false, just analyze the previously saved experiments

root_saved_dir = 'saved/'
# -------------------------------------------------------------------------------------------
# Run experiments for a grid of 'number of training tasks'
# -------------------------------------------------------------------------------------------

n_tasks_vec = np.arange(min_n_tasks, max_n_tasks+1)

sub_runs_names = [base_run_name + '/' + str(n_train_tasks) for n_train_tasks in n_tasks_vec]

if run_experiments:
    start_time = timeit.default_timer()
    for i_run, n_train_tasks in enumerate(n_tasks_vec):
        call(['python', 'DP_main_Meta_Bayes.py',
              '--run-name', sub_runs_names[i_run],
              '--data-source', 'MNIST',
              '--data-transform', 'Permute_Labels',
              '--n_train_tasks', str(n_train_tasks),
              '--limit_train_samples_in_test_tasks', '0',
              '--model-name',   'ConvNet3',
              '--complexity_type',  'PAC_Bayes_lambda', # 'NewBoundMcAllaster' 'NewBoundSeeger'
              # 'PAC_Bayes_lambda' 'PAC_Bayes_quad' 'PAC_Bayes_variational_role'
              '--n_test_tasks', '20',  # 100
              '--n_meta_train_epochs', '50',  # 150
              '--n_meta_test_epochs', '100',  # 200
              '--meta_batch_size', '16',
              '--data_prior_ratio', '0.3',  # proportion of data in data dependent prior
              '--data_prior_train_epochs', '15',  # default 20: training 20 epochs
              '--data_prior_test_epochs', '20',  # default 20: testing 20 epochs
              '--data_prior_batch_size', '128',
              '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
              ])
    stop_time = timeit.default_timer()
    # Save log text
    message = ['Run finished at ' +  datetime.now().strftime(' %Y-%m-%d %H:%M:%S'),
               'Tasks number grid: ' + str(n_tasks_vec),
                'Total runtime: ' + time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),
                '-'*30]
    log_file_path = os.path.join(root_saved_dir, base_run_name, 'log') + '.out'
    with open(log_file_path, 'a') as f:
        for string in message:
            print(string, file=f)
            print(string)

# -------------------------------------------------------------------------------------------
# Analyze the experiments
# -------------------------------------------------------------------------------------------

# test error analysis
mean_error_per_tasks_n = np.zeros(len(n_tasks_vec))
std_error_per_tasks_n = np.zeros(len(n_tasks_vec))
# test loss analysis
mean_loss_per_tasks_n = np.zeros(len(n_tasks_vec))
std_loss_per_tasks_n = np.zeros(len(n_tasks_vec))
# test bound analysis
mean_bound_per_tasks_n = np.zeros(len(n_tasks_vec))
std_bound_per_tasks_n = np.zeros(len(n_tasks_vec))

# test_err_vec
# test_err_loss
# test_err_bound
for i_run, n_train_tasks in enumerate(n_tasks_vec):
    run_result_path = os.path.join(root_saved_dir, sub_runs_names[i_run])
    prm, info_dict = load_run_data(run_result_path)
    test_err_vec = info_dict['test_err_vec']
    mean_error_per_tasks_n[i_run] = test_err_vec.mean()
    std_error_per_tasks_n[i_run] = test_err_vec.std()
    # test loss
    test_loss_vec = info_dict['test_loss_vec']
    mean_loss_per_tasks_n[i_run] = test_loss_vec.mean()
    std_loss_per_tasks_n[i_run] = test_loss_vec.std()
    # test bound
    test_bound_vec = info_dict['test_bound_vec']
    mean_bound_per_tasks_n[i_run] = test_bound_vec.mean()
    std_bound_per_tasks_n[i_run] = test_bound_vec.std()

# Saving the analysis:
with open(os.path.join(root_saved_dir, base_run_name, 'runs_analysis.pkl'), 'wb') as f:
    pickle.dump([mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec], f)

with open(os.path.join(root_saved_dir, base_run_name, 'runs_analysis.pkl'), 'wb') as f:
        pickle.dump([mean_loss_per_tasks_n, std_loss_per_tasks_n, n_tasks_vec], f)

with open(os.path.join(root_saved_dir, base_run_name, 'runs_analysis.pkl'), 'wb') as f:
    pickle.dump([mean_bound_per_tasks_n, std_bound_per_tasks_n, n_tasks_vec], f)

# # Plot the analysis:
# plt.figure()
# plt.errorbar(n_tasks_vec, 100*mean_error_per_tasks_n, yerr=100*std_error_per_tasks_n)
# plt.xticks(n_tasks_vec)
# plt.xlabel('Number of training-tasks')
# plt.ylabel('Error on new task [%]')
#
#
# plt.figure()
# plt.errorbar(n_tasks_vec, mean_loss_per_tasks_n, yerr=std_loss_per_tasks_n)
# plt.xticks(n_tasks_vec)
# plt.xlabel('Number of training-tasks')
# plt.ylabel('Loss on new task')
#
#
# plt.figure()
# plt.errorbar(n_tasks_vec, mean_bound_per_tasks_n, yerr=std_bound_per_tasks_n)
# plt.xticks(n_tasks_vec)
# plt.xlabel('Number of training-tasks')
# plt.ylabel('Bound on new task')
# plt.show()
