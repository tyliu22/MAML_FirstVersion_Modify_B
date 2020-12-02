# Meta-learning-PAC-Bayes-bound-with-data-depedent-prior

- 'PriorMetaLearning'
   to analyze the influence of different PAC-Bayes bounds with permuted labels and pixels
   - 'NewBoundMcAllaster'
   - 'NewBoundSeeger'
   - 'PAC_Bayes_lambda'
   - 'PAC_Bayes_quad' 
   - 'PAC_Bayes_variational_role'

- 'Meta_DataDependentPrior'
   to analyze the influence of data dependent prior
- 'requirements.txt'
   version of libraries
   
 ## Project Structure
```
.
|----Data_Path.py  
|----MAML/
|    |----main_MAML.py  
|    |----MAML_meta_step.py  
|    |----meta_test_MAML.py  
|    |----meta_train_MAML_finite_tasks.py  
|    |----meta_train_MAML_infinite_tasks.py  
|    |----run_MAML_PermuteLabels.py  
|    |----run_MAML_ShuffledPixels.py  
|----Meta_DataDependentPrior/ 
|    |----DP_Analyze_Prior.py  
|    |----DP_AvargeTransfer.py  
|    |----DP_Get_Objective_MPB.py  
|    |----DP_main_Meta_Bayes.py  
|    |----DP_meta_test_Bayes.py  
|    |----DP_meta_train_Bayes_finite_tasks.py  
|    |----DP_meta_train_Bayes_infinite_tasks.py  
|    |----DP_run_MPB_PermutedLabels_TasksN.py  
|    |----DP_run_MPB_ShuffledPixels_TasksN.py  
|    |----DP_show_TasksN_Plots.py  
|    |----DP_test_data_prior_learning.py  
|    |----DP_train_data_prior_learning.py  
|    |----saved/
|----ML_data_sets/
|----Models/
|    |----deterministic_models.py  
|    |----layer_inits.py  
|    |----stochastic_inits.py  
|    |----stochastic_layers.py  
|    |----stochastic_models.py  
|----PriorMetaLearning/
|    |----Analyze_Prior.py  
|    |----AvargeTransfer.py  
|    |----Get_Objective_MPB.py  
|    |----main_Meta_Bayes.py  
|    |----meta_test_Bayes.py  
|    |----meta_train_Bayes_finite_tasks.py  
|    |----meta_train_Bayes_infinite_tasks.py  
|    |----run_MPB_PermutedLabels_TasksN.py  
|    |----run_MPB_PermuteLabels.py  
|    |----run_MPB_ShuffledPixels.py  
|    |----run_MPB_ShuffledPixels_TasksN.py  
|    |----saved/
|    |----show_TasksN_Plots.py  
|----README.md  
|----requirements.txt  
|----tree.py  
|----Utils/
|    |----Bayes_utils.py  
|    |----common.py  
|    |----data_gen.py  
|    |----DP_data_gen.py  
|    |----imagenet_data.py  
|    |----omniglot.py  
|    |----Resize_ImageNet.py  
```

## Requirements
certifi==2020.6.20  
cycler==0.10.0  
future==0.18.2  
joblib==0.17.0  
kiwisolver==1.2.0  
matplotlib==3.3.1  
numpy==1.19.1  
Pillow==7.2.0  
pyparsing==2.4.7  
python-dateutil==2.8.1  
scikit-learn==0.23.2  
scipy==1.5.2  
six==1.15.0  
sklearn==0.0  
threadpoolctl==2.1.0  
torch==1.5.1  
torchvision==0.6.1  
