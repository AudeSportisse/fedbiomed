import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager
from fedbiomed.common.constants import ProcessTypes


from fedbiomed.common.training_args import TrainingArgs
from torch.optim import Optimizer

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.privacy import DPController
from fedbiomed.common.utils import get_method_spec
from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant
from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan



###########################################################
# Federated standardization                               #
###########################################################

class FedMeanStdTrainingPlan(TorchTrainingPlan):
    
    def init_dependencies(self):
        deps = ["import pandas as pd",
            "import numpy as np"]
        return deps
        
    def init_model(self,model_args):
        
        model = self.MeanStd(model_args)
        
        return model
    
    class MeanStd(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.n_features=model_args['n_features']
            
            self.mean = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.std = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.size = nn.Parameter(torch.zeros(self.n_features,dtype=torch.float64),requires_grad=False)
            self.fake = nn.Parameter(torch.randn(1),requires_grad=True)

        def forward(self, data):
            data_np = data.numpy()
            N = data.shape[0]
            
            ### Implementing with np.nanmean, np.nanstd
            self.size += torch.Tensor([N - np.count_nonzero(np.isnan(data_np[:,dim]))\
                                    for dim in range(self.n_features)])
            self.mean += torch.from_numpy(np.nanmean(data_np,0))
            self.std += torch.from_numpy(np.nanstd(data_np,0))
            
            return self.fake
    
        
    def training_data(self):
        
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        
        ### NOTE: batch_size should be == dataset size ###
        batch_size = df.shape[0]
        x_train = df.values.astype(np.float64)
        #print(x_train.dtype)
        x_mask = np.isfinite(x_train)
        xhat_0 = np.copy(x_train)
        ### NOTE: we keep nan when data is missing
        #xhat_0[np.isnan(x_train)] = 0
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        
        data_manager = DataManager(dataset=xhat_0 , target=x_mask , **train_kwargs)
        
        return data_manager
    
    def training_step(self, data, mask):
        
        return self.model().forward(data)


####################################################
##########MY CLASS FOR FEDERATED LEARNING###########
####################################################        
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, OrderedDict, Union, Iterator


from fedbiomed.common.models import TorchModel
from fedbiomed.common.optimizers.generic_optimizers import OptimizerBuilder

ModelInputType = Union[torch.Tensor, Dict, List, Tuple]

        
class GAINTrainingPlan(TorchTrainingPlan):
    
    def init_dependencies(self):
        deps = ["import torch.distributions as td",
            "import pandas as pd",
            "import numpy as np",
            "import torch",
            "from typing import Any, Dict, List, Tuple, Optional, OrderedDict, Union, Iterator",
            "from abc import ABC, abstractmethod",
            "from fedbiomed.common.constants import ErrorNumbers, TrainingPlans",
            "from fedbiomed.common.exceptions import FedbiomedTrainingPlanError",
            "from fedbiomed.common.logger import logger",
            "from fedbiomed.common.metrics import MetricTypes",
            "from fedbiomed.common.privacy import DPController",
            "from fedbiomed.common.utils import get_method_spec",
            "from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant",
            "from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan",
            "from copy import deepcopy",
            "from fedbiomed.common.training_args import TrainingArgs",
            "from torch.optim import Optimizer",
            "from fedbiomed.common.models import TorchModel",
            "from fedbiomed.common.optimizers.generic_optimizers import OptimizerBuilder"]
        return deps
        
    def init_model(self,model_args):
        
        if 'standardization' in model_args:
            self.standardization = True
            if (('fed_mean' in model_args['standardization']) and ('fed_std' in model_args['standardization'])):
                self.fed_mean = np.array(model_args['standardization']['fed_mean'])
                self.fed_std = np.array(model_args['standardization']['fed_std'])
            else:
                self.fed_mean = None
                self.fed_std = None
                
        self.n_features=model_args['n_features']
        self.n_latent=model_args['n_latent']
        self.n_hidden=model_args['n_hidden']
        self.n_samples=model_args['n_samples']
        
        self.it = 0

        self.__type = TrainingPlans.TorchTrainingPlan

        model = self.GAIN(model_args)
        
        return model

    def init_optimizer(self,optimizer_args):

        #Adam optimizer
        fed_d_lr = 1e-4 ###BY DEFAULT
        fed_g_lr = 1e-4 ###BY DEFAULT
        optimizer_D = torch.optim.Adam(params=self.model().D.parameters(), lr=fed_d_lr)
        optimizer_G = torch.optim.Adam(params=self.model().G.parameters(), lr=fed_g_lr)

        #### No Automatic self-precision (argument use_amp)

        return optimizer_D, optimizer_G  #####ATTENTION 2 OPTIMIZERS


    class GAIN(nn.Module):
            def __init__(self, model_args):
                super().__init__()

                n_features=model_args['n_features']
                n_latent=model_args['n_latent']
                n_hidden=model_args['n_hidden']
                n_samples=model_args['n_samples']

                # the encoder will output both the mean and the diagonal covariance
                G_H_Dim = 64 ###BY DEFAULT
                D_H_Dim = 64 ###BY DEFAULT
                data_dim = n_features 
                input_dim = data_dim ###They took this by default
                
                def xavier_init(size):
                    in_dim = size[0]
                    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
                    return np.random.normal(size=size, scale=xavier_stddev)
                
                class Discriminator(nn.Module):
                    def __init__(self, input_size, hidden_dim):
                        super(Discriminator, self).__init__()
                        self.d = nn.Sequential(
                            nn.Linear(input_size * 2, hidden_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(hidden_dim, hidden_dim//2),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(hidden_dim//2, hidden_dim // 2),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(hidden_dim//2, input_size),
                            nn.Sigmoid()
                        ).double()

                    def forward(self, new_x, h):
                        inputs = torch.cat(dim=1, tensors=[new_x, h])
                        D_prob = self.d(inputs)
                        return D_prob


                class Generator(nn.Module):

                    def __init__(self, input_size, hidden_dim, output_size):
                        super(Generator, self).__init__()
                        self.g = nn.Sequential(
                            nn.Linear(input_size+output_size, hidden_dim//2),
                            nn.ReLU(True).double(),
                            nn.Linear(hidden_dim//2, hidden_dim//2),
                            nn.ReLU(True),
                            nn.Linear(hidden_dim//2, output_size),
                            nn.Sigmoid()
                        ).double()

                    def forward(self, new_x, m):
                        # Mask + Data Concatenate
                        inputs = torch.cat(dim=1, tensors=(new_x, m))
                        G_prob = self.g(inputs)
                        return G_prob
                
                
                self.G = Generator(input_dim, G_H_Dim, data_dim)#.to(args.device)
                self.D = Discriminator(data_dim, D_H_Dim)#.to(args.device)

                self.G.apply(self.weights_init)
                self.D.apply(self.weights_init)
                
            def weights_init(self,m):
                classname = m.__class__.__name__
                if classname.find('Conv2d') != -1:
                    nn.init.xavier_normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0.0)
                elif classname.find('Linear') != -1:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
    
    def init_optimizer(self,optimizer_args):

        #Adam optimizer
        fed_d_lr = 1e-4 ###BY DEFAULT
        fed_g_lr = 1e-4 ###BY DEFAULT
        optimizer_D = torch.optim.Adam(params=self.model().D.parameters(), lr=fed_d_lr)
        optimizer_G = torch.optim.Adam(params=self.model().G.parameters(), lr=fed_g_lr)

        #### No Automatic self-precision (argument use_amp)

        return optimizer_D, optimizer_G  #####ATTENTION 2 OPTIMIZERS

    def _configure_model_and_optimizer(self):
        """Configures model and optimizer before training """

        # Message to format for unexpected argument definitions in special methods
        method_error = \
            ErrorNumbers.FB605.value + ": Special method `{method}` has more than one argument: {keys}. This method " \
                                       "can not have more than one argument/parameter (for {prefix} arguments) or " \
                                       "method can be defined without argument and `{alternative}` can be used for " \
                                       "accessing {prefix} arguments defined in the experiment."

        # Get model defined by user -----------------------------------------------------------------------------
        init_model_spec = get_method_spec(self.init_model)
        if not init_model_spec:
            model = self.init_model()
        elif len(init_model_spec.keys()) == 1:
            model = self.init_model(self._model_args)
        else:
            raise FedbiomedTrainingPlanError(method_error.format(prefix="model",
                                                                 method="init_model",
                                                                 keys=list(init_model_spec.keys()),
                                                                 alternative="self.model_args()"))

        # Validate and fix model
        model = self._dp_controller.validate_and_fix_model(model)
        self._model = TorchModel(model)

        # Get optimizer defined by researcher ---------------------------------------------------------------------
        init_optim_spec = get_method_spec(self.init_optimizer)
        if not init_optim_spec:
            self._optimizerD, self._optimizerG = self.init_optimizer()
            optimizer = self._optimizerD ##MultipleOptimizer(self._optimizerD, self._optimizerG)
        elif len(init_optim_spec.keys()) == 1:
            self._optimizerD, self._optimizerG = self.init_optimizer(self._optimizer_args)
            optimizer = self._optimizerD ##MultipleOptimizer(self._optimizerD, self._optimizerG)
        else:
            raise FedbiomedTrainingPlanError(method_error.format(prefix="optimizer",
                                                                 method="init_optimizer",
                                                                 keys=list(init_optim_spec.keys()),
                                                                 alternative="self.optimizer_args()"))

        # Validate optimizer
        optim_builder = OptimizerBuilder()
        #  build the optimizer wrapper
        self._optimizer = optim_builder.build(self.__type, self._model, optimizer) 


    
    def training_routine(self,
                         history_monitor: Any = None,
                         node_args: Union[dict, None] = None,
                         ) -> int:
        # FIXME: add betas parameters for ADAM solver + momentum for SGD
        # FIXME 2: remove parameters specific for validation specified in the
        # training routine
        """Training routine procedure.

        End-user should define;

        - a `training_data()` function defining how sampling / handling data in node's dataset is done. It should
            return a generator able to output tuple (batch_idx, (data, targets)) that is iterable for each batch.
        - a `training_step()` function defining how cost is computed. It should output loss values for backpropagation.

        Args:
            history_monitor: Monitor handler for real-time feed. Defined by the Node and can't be overwritten
            node_args: command line arguments for node. Can include:
                - `gpu (bool)`: propose use a GPU device if any is available. Default False.
                - `gpu_num (Union[int, None])`: if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available. Default None.
                - `gpu_only (bool)`: force use of a GPU device if any available, even if researcher
                    doesn't request for using a GPU. Default False.
        Returns:
            Total number of samples observed during the training.
        """

        #self.model().train()  # pytorch switch for training
        self._optimizer.init_training()
        # set correct type for node args
        node_args = {} if not isinstance(node_args, dict) else node_args

        # send all model to device, ensures having all the requested tensors
        self._set_device(self._use_gpu, node_args)
        self._model.send_to_device(self._device)

        # Run preprocess when everything is ready before the training
        self._preprocess()

        # # initial aggregated model parameters
        # self._init_params = deepcopy(list(self.model().parameters()))

        # DP actions
        self._optimizer, self.training_data_loader = \
            self._dp_controller.before_training(optimizer= self._optimizer, loader=self.training_data_loader)

        # set number of training loop iterations
        iterations_accountant = MiniBatchTrainingIterationsAccountant(self)

        # Training loop iterations
        for epoch in iterations_accountant.iterate_epochs():
            training_data_iter: Iterator = iter(self.training_data_loader)

            for batch_idx in iterations_accountant.iterate_batches():
                # retrieve data and target
                data, target = next(training_data_iter)

                # update accounting for number of observed samples
                batch_size = self._infer_batch_size(data)
                iterations_accountant.increment_sample_counters(batch_size)

                # handle training on accelerator devices
                data, target = self.send_to_device(data, self._device), self.send_to_device(target, self._device)

                # train this batch
                res = self._train_over_batch(data, target)

                if len(res)==4:
                    corrected_lossD, lossD,corrected_lossG, lossG  =  res
                else:
                    corrected_lossD, lossD = res

                # Reporting
                if iterations_accountant.should_log_this_batch():
                    # Retrieve reporting information: semantics differ whether num_updates or epochs were specified
                    num_samples, num_samples_max = iterations_accountant.reporting_on_num_samples()
                    num_iter, num_iter_max = iterations_accountant.reporting_on_num_iter()
                    epoch_to_report = iterations_accountant.reporting_on_epoch()

                    logger.debug('Train {}| '
                                 'Iteration {}/{} | '
                                 'Samples {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(
                                     f'Epoch: {epoch_to_report} ' if epoch_to_report is not None else '',
                                     num_iter,
                                     num_iter_max,
                                     num_samples,
                                     num_samples_max,
                                     100. * num_iter / num_iter_max,
                                     lossG.item())
                                 )

                    # Send scalar values via general/feedback topic
                    if history_monitor is not None:
                        # the researcher only sees the average value of samples observed until now
                        history_monitor.add_scalar(metric={'Loss': lossD.item()},
                                                   iteration=num_iter,
                                                   epoch=epoch_to_report,
                                                   train=True,
                                                   num_samples_trained=num_samples,
                                                   num_batches=num_iter_max,
                                                   total_samples=num_samples_max,
                                                   batch_samples=batch_size)

                # Handle dry run mode
                if self._dry_run:
                    self._model.send_to_device(self._device_init)
                    torch.cuda.empty_cache()
                    return iterations_accountant.num_samples_observed_in_total

        # release gpu usage as much as possible though:
        # - it should be done by deleting the object
        # - and some gpu memory remains used until process (cuda kernel ?) finishes

        self._model.send_to_device(self._device_init)
        torch.cuda.empty_cache()
        
        # # test (to be removed)
        # assert id(self._optimizer.model.model) == id(self._model.model)
        
        # assert id(self._optimizer.model) == id(self._model)
        
        # for (layer, val), (layer2, val2) in zip(self._model.model.state_dict().items(), self._optimizer.model.model.state_dict().items()):
        #     assert layer == layer2
        #     print(val, layer2)
        #     assert torch.isclose(val, val2).all()
            
        # for attributes, values in self._model.__dict__.items():
        #     print("ATTRIBUTES", values)
        #     assert values == getattr(self._optimizer.model, attributes)

        # for attributes, values in self._model.model.__dict__.items():
        #     print("ATTRIBUTES", values)
        #     assert values == getattr(self._optimizer.model.model, attributes) 
        return iterations_accountant.num_samples_observed_in_total

    

    def _train_over_batch(self, data: Union[torch.Tensor, Dict, List, Tuple], target: Union[torch.Tensor, Dict, List, Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model over a single batch of data.

        This function handles all the torch-specific logic concerning model training, including backward propagation,
        aggregator-specific correction terms, and optimizer stepping.

        Args:
            data: the input data to the model
            target: the training labels

        Returns:
            corrected loss: the loss value used for backward propagation, including any correction terms
            loss: the uncorrected loss for reporting
        """
        # zero-out gradients
        
        # compute loss
        loss = self.training_step(data, target)
        istuple_loss = isinstance(loss,tuple)
        if istuple_loss:# if (self.it-1) % fed_n_critic == 0:
            # raises an exception if not provided
            self._optimizerD.zero_grad()
            self._optimizerG.zero_grad()
            lossD, lossG = loss
            corrected_lossD = torch.clone(lossD)
            corrected_lossG = torch.clone(lossG)
        else: 
            self._optimizerD.zero_grad()
            lossD = loss
            corrected_lossD = torch.clone(lossD)

        # If FedProx is enabled: use regularized loss function
        if self._fedprox_mu is not None:
            if istuple_loss:
                corrected_lossD += float(self._fedprox_mu) / 2 * self.__norm_l2()
                corrected_lossG += float(self._fedprox_mu) / 2 * self.__norm_l2()
            else: 
                corrected_lossD += float(self._fedprox_mu) / 2 * self.__norm_l2()
        
        
        if istuple_loss:
            # Run the backward pass to compute parameters' gradients
            corrected_lossD.backward()
            corrected_lossG.backward()
        else:
            corrected_lossD.backward()
        
        # If Scaffold is used: apply corrections to the gradients
        if self.aggregator_name is not None and self.aggregator_name.lower() == "scaffold":
            for name, param in self._model.named_parameters():
                correction = self.correction_state.get(name)
                if correction is not None:
                    param.grad.add_(correction.to(param.grad.device))

        # Have the optimizer collect, refine and apply gradients
        if istuple_loss:
            self._optimizerD.step()
            self._optimizerG.step()    
        else:
            self._optimizerD.step()
                
        if istuple_loss:
            return corrected_lossD, lossD, corrected_lossG, lossG
        else:
            return corrected_lossD, lossD
                
            

    def gain_loss(self,data,mask):
        

        p_hint = 0.9 ###BY DEFAULT

        m = data.shape[0] #mb_size in the original code
        n = data.shape[1] #self.Dim in the original code
        Z_mb = np.random.uniform(0., 1, size=[m, n])

        X_mb = data
        M_mb = mask
        

        def sample_M(m, n, p):
            A = np.random.uniform(0., 1., size=[m, n])
            B = A > p
            C = 1. * B
            return C

        H_mb1 = sample_M(m, n, 1 - p_hint)
        
        
        H_mb = M_mb * H_mb1

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb


        def compute_D_loss(G,D,M,New_X,H):
            M = torch.tensor(np.float64(M)) 
            New_X = torch.tensor(np.float64(New_X)) 
            G_sample = G(New_X, M)
            Hat_New_X = New_X * M + G_sample * (1 - M)
            D_prob = D(Hat_New_X, H)
            D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob))
            return D_loss

        ##############
        ###TorchTrainingPlan _train_over_batch
        ##############

        def compute_G_loss(G,D,X,M,New_X,H):
            G_sample = G(New_X, M)
            Hat_New_X = New_X * M + G_sample * (1 - M)

            # Discriminator
            D_prob = D(Hat_New_X, H)

            # %% Loss
            G_loss1 = -torch.mean((1 - M) * torch.log(D_prob))

            MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

            alpha = 10 ###BY DEFAULT
            
            d_lr_decay_step = 200 ###BY DEFAULT: D learning rate decay after N step
            if (self.it + 1) % d_lr_decay_step == 0:
                alpha = alpha * 0.9

            G_loss = G_loss1 + alpha * MSE_train_loss

            # %% MSE Performance metric
            MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
            RMSE_test_loss = torch.sqrt(MSE_test_loss)

            return G_loss, alpha * MSE_train_loss, RMSE_test_loss

        D_loss_curr = compute_D_loss(self.model().G, self.model().D, M=M_mb, New_X=New_X_mb, H=H_mb)
        

        fed_n_critic = 20 # 3 ###BY DEFAULT
        if self.it % fed_n_critic == 0:
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = compute_G_loss(self.model().G, self.model().D, X=X_mb, M=M_mb,New_X=New_X_mb, H=H_mb)
            return D_loss_curr, G_loss_curr

        return D_loss_curr

    def training_data(self):
        batch_size=self._training_args.get('batch_size')
        df = pd.read_csv(self.dataset_path, sep=',', index_col=False)
        x_train = df.values.astype(np.float64)
        x_mask = np.isfinite(x_train)
        # xhat_0: missing values are replaced by zeros. 
        #This x_hat0 is what will be fed to our encoder.
        xhat_0 = np.copy(x_train)

        # Data standardization
        if self.standardization:
            xhat_0 = self.standardize_data(xhat_0)

        xhat_0[np.isnan(x_train)] = 0
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}

        data_manager = DataManager(dataset=xhat_0 , target=x_mask , **train_kwargs)

        return data_manager

    def standardize_data(self,data):
        data_norm = np.copy(data)
        if ((self.fed_mean is not None) and (self.fed_std is not None)):
            print('FEDERATED STANDARDIZATION')
            data_norm = (data_norm - self.fed_mean)/self.fed_std
        else:
            print('LOCAL STANDARDIZATION')
            data_norm = (data_norm - np.nanmean(data_norm,0))/np.nanstd(data_norm,0)
        return data_norm

    def training_step(self, data, mask):
        #self.model().G.zero_grad()
        #self.model().D.zero_grad()
        
        loss = self.gain_loss(data = data,mask = mask)
        self.it += 1
        return loss
    
