
from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_bayes_task_objective, run_test_Bayes
from Utils.common import grad_step, count_correct, get_loss_criterion, write_to_log
from Meta_DataDependentPrior.DP_Get_Objective_MPB import get_objective

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------

def run_prior_learning(data_loaders, prm, prior_model):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    print('DP_train_data_prior_learning: setting-up')
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    n_train_tasks = len(data_loaders)

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = prior_model

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())

    # here annotate the code
    all_params = prior_params
    # all_params = all_post_param
    all_optimizer = optim_func(all_params, **optim_args)

    # number of sample-batches in each task:
    n_batch_list = [len(data_loader['data_prior']) for data_loader in data_loaders]

    n_batches_per_task = np.max(n_batch_list)

    # -------------------------------------------------------------------------------------------
    #  Training epoch function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch, i_step = 0):

        # For each task, prepare an iterator to generate training batches:
        train_iterators = [iter(data_loaders[ii]['data_prior']) for ii in range(n_train_tasks)]

        # The task order to take batches from:
        # The meta-batch will be balanced - i.e, each task will appear roughly the same number of times
        # note: if some tasks have less data that other tasks - it may be sampled more than once in an epoch
        task_order = []
        task_ids_list = list(range(n_train_tasks))
        # create -- n_batches_per_task * n_train_tasks -- number list
        for i_batch in range(n_batches_per_task):
            random.shuffle(task_ids_list)
            task_order += task_ids_list
        # Note: this method ensures each training sample in each task is drawn in each epoch.
        # If all the tasks have the same number of sample, then each sample is drawn exactly once in an epoch.

        # random.shuffle(task_ids_list) # -- ############ -- TEMP

        # ----------- meta-batches loop (batches of tasks) -----------------------------------#
        # each meta-batch includes several tasks
        # we take a grad step with theta after each meta-batch
        # - maximum -- prm.meta_batch_size tasks -- in each meta batch
        # total -- len(task_order) / prm.meta_batch_size -- tasks
        meta_batch_starts = list(range(0, len(task_order), prm.meta_batch_size))
        # totally update -- len(meta_batch_starts) -- times
        n_meta_batches = len(meta_batch_starts)

        for i_meta_batch in range(n_meta_batches):

            # only select prm.meta_batch_size 5 tasks in each meta batch
            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            # meta-batch size may be less than prm.meta_batch_size at the last one
            # note: it is OK if some tasks appear several times in the meta-batch

            mb_data_loaders = [data_loaders[task_id] for task_id in task_ids_in_meta_batch]
            mb_iterators = [train_iterators[task_id] for task_id in task_ids_in_meta_batch]

            i_step += 1

            # Get objective based on tasks in meta-batch:
            empirical_error = get_risk(prior_model, prm, mb_data_loaders,
                                                  mb_iterators, loss_criterion, n_train_tasks)

            grad_step(empirical_error, all_optimizer, lr_schedule, prm.lr, i_epoch)
            log_interval = 20
            if i_meta_batch % log_interval == 0:
                print('number meta batch:{} \t avg_empiric_loss:{:.3f}'
                    .format(i_meta_batch, empirical_error))
            # print(i_step)
        # end  meta-batches loop
        # return i_step
    # end run_epoch()
    # Training loop:

    def get_risk(prior_model, prm, mb_data_loaders, mb_iterators, loss_criterion, n_train_tasks):
        '''  Calculate objective based on tasks in meta-batch '''
        # note: it is OK if some tasks appear several times in the meta-batch

        n_tasks_in_mb = len(mb_data_loaders)

        sum_empirical_loss = 0
        sum_intra_task_comp = 0
        correct_count = 0
        sample_count = 0

        # ----------- loop over tasks in meta-batch -----------------------------------#
        for i_task in range(n_tasks_in_mb):

            n_samples = mb_data_loaders[i_task]['n_train_samples']

            # get sample-batch data from current task to calculate the empirical loss estimate:
            batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task], mb_data_loaders[i_task]['train'])

            # The posterior model corresponding to the task in the batch:
            # post_model = mb_posteriors_models[i_task]
            prior_model.train()

            # Monte-Carlo iterations:
            n_MC = prm.n_MC
            task_empirical_loss = 0
            # task_complexity = 0
            # ----------- Monte-Carlo loop  -----------------------------------#
            for i_MC in range(n_MC):
                # get batch variables:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # Empirical Loss on current task:
                outputs = prior_model(inputs)
                curr_empirical_loss = loss_criterion(outputs, targets)

                correct_count += count_correct(outputs, targets)
                sample_count += inputs.size(0)

                task_empirical_loss += (1 / n_MC) * curr_empirical_loss
            # end Monte-Carlo loop

            sum_empirical_loss += task_empirical_loss

        # end loop over tasks in meta-batch
        avg_empirical_loss = (1 / n_tasks_in_mb) * sum_empirical_loss

        return avg_empirical_loss


    for i_epoch in range(prm.data_prior_train_epochs):
        run_train_epoch(i_epoch)
        print('Data_dependent_prior training epoch:{}'.format(i_epoch))

    return prior_model





