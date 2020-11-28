
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

def run_prior_learning(task_data, prm, prior_model):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    print('DP_test_data_prior_learning: setting-up')
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    # prior_model = prior_model

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())

    # here annotate the code
    all_params = prior_params
    # all_params = all_post_param
    all_optimizer = optim_func(all_params, **optim_args)

    data_prior_loader = task_data['data_prior']
    n_data_prior_samples = len(data_prior_loader.dataset)
    n_data_prior_batches = len(data_prior_loader)


    # -------------------------------------------------------------------------------------------
    #  Training epoch function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        prior_model.train()

        for batch_idx, batch_data in enumerate(data_prior_loader):

            correct_count = 0
            sample_count = 0

            # Monte-Carlo iterations:
            n_MC = prm.n_MC
            task_empirical_loss = 0
            for i_MC in range(n_MC):
                # get batch:
                inputs, targets = data_gen.get_batch_vars(batch_data, prm)

                # Calculate empirical loss:
                outputs = prior_model(inputs)
                curr_empirical_loss = loss_criterion(outputs, targets)

                task_empirical_loss += (1 / n_MC) * curr_empirical_loss

                correct_count += count_correct(outputs, targets)
                sample_count += inputs.size(0)

                # Total objective:

            total_objective = task_empirical_loss

            # Take gradient step with the posterior:
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)
            log_interval = 20
            if batch_idx % log_interval == 0:
                batch_acc = correct_count / sample_count
                print('number meta batch:{} \t avg_empiric_loss:{:.3f} \t batch accuracy:{:.3f}'
                      .format(batch_idx, total_objective, batch_acc))
            # return total_objective.item()

    # end run_epoch()
    # Training loop:
    for i_epoch in range(prm.data_prior_test_epochs):
        run_train_epoch(i_epoch)
        print('Data_dependent_prior testing epoch:{}'.format(i_epoch))

    return prior_model

