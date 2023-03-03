# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from fedbiomed.common.models import SkLearnModel

import json
import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes

from ._base_training_plan import BaseTrainingPlan


class SKLearnTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this abstract class must:
    - Specify a `_model_cls` class attribute that defines the type
      of scikit-learn model being wrapped for training.
    - Implement a `set_init_params` method that:
      - sets and assigns the model's initial trainable weights attributes.
      - populates the `_param_list` attribute with names of these attributes.
    - Implement a `_training_routine` method that performs a training round
      based on `self.train_data_loader` (which is a `NPDataLoader`).

    Attributes:
        dataset_path: The path that indicates where dataset has been stored
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    _model_cls: Type[BaseEstimator]  # wrapped model class
    _model_dep: Tuple[str, ...] = tuple()  # model-specific dependencies

    def __init__(self) -> None:
        """Initialize the SKLearnTrainingPlan."""
        super().__init__()
        self._model = SkLearnModel(self._model_cls)
        #self._model_args = {}  # type: Dict[str, Any]
        self._training_args = {}  # type: Dict[str, Any]
        #self._param_list = []  # type: List[str]
        self.__type = TrainingPlans.SkLearnTrainingPlan
        #self._is_classification = False
        self._batch_maxnum = 0
        self.dataset_path: Optional[str] = None
        self.add_dependency([
            "import inspect",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
            "from fedbiomed.common.data import DataManager",
        ])
        self.add_dependency(list(self._model_dep))

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any],
            aggregator_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
        """
        model_args.setdefault("verbose", 1)
        self._model.model_args = model_args
        self._aggregator_args = aggregator_args or {}

        self._training_args = training_args.pure_training_arguments()
        self._batch_maxnum = self._training_args.get('batch_maxnum', self._batch_maxnum)
        # Add dependencies
        self._configure_dependencies()
        # Override default model parameters based on `self._model_args`.
        params = {
            key: model_args.get(key, val)
            for key, val in self._model.get_params().items()
        }

        self._model.set_params(**params)
        # Set up additional parameters (normally created by `self._model.fit`).
        # TODO: raise error if
        self._model.set_init_params(model_args)

    # @abstractmethod
    # def set_init_params(self) -> None:
    #     """Initialize the model's trainable parameters."""

    def set_data_loaders(
            self,
            train_data_loader: Union[DataLoader, NPDataLoader, None],
            test_data_loader: Union[DataLoader, NPDataLoader, None]
    ) -> None:
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for validation routine
        """
        args = (train_data_loader, test_data_loader)
        if not all(isinstance(data, NPDataLoader) for data in args):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan expects "
                "NPDataLoader instances as training and testing data "
                f"loaders, but received {type(train_data_loader)} "
                f"and {type(test_data_loader)} respectively."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def model_args(self) -> Dict[str, Any]:
        """Retrieve model arguments.

        Returns:
            Model arguments
        """
        return self._model.model_args

    def training_args(self) -> Dict[str, Any]:
        """Retrieve training arguments.

        Returns:
            Training arguments
        """
        return self._training_args

    def model(self) -> BaseEstimator:
        """Retrieve the wrapped scikit-learn model instance.

        Returns:
            Scikit-learn model instance
        """
        return self._model.model

    def get_model_params(self) -> Dict:
        return self.after_training_params()

    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Training routine, to be called once per round.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording training metadata. Defaults to None.
            node_args: command line arguments for node.
                These arguments can specify GPU use; however, this is not
                supported for scikit-learn models and thus will be ignored.
        """
        if self._model is None:
            raise FedbiomedTrainingPlanError('model is None')

        # Run preprocesses
        self._preprocess()

        if not isinstance(self.training_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot "
                "be trained without a NPDataLoader as `training_data_loader`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Run preprocessing operations.
        self._preprocess()
        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )
        # Run the model-specific training routine.
        try:
            return self._training_routine(history_monitor)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while fitting "
                f"the model: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    @abstractmethod
    def _training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None
    ) -> None:
        """Model-specific training routine backend.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording the loss value during training.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`
        (that notably enforces preprocessing and exception catching).
        """
        return None

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional['HistoryMonitor'],
            before_train: bool
    ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        # Check that the testing data loader is of proper type.
        if not isinstance(self.testing_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot be "
                "evaluated without a NPDataLoader as `testing_data_loader`."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        # If required, make up for the lack of specifications regarding target
        # classification labels.
        if self._model._is_classification and not hasattr(self.model(), 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self.model(), 'classes_', classes)
        # If required, select the default metric (accuracy or mse).
        if metric is None:
            if self._model._is_classification:
                metric = MetricTypes.ACCURACY
            else:
                metric = MetricTypes.MEAN_SQUARE_ERROR
        # Delegate the actual evalation routine to the parent class.
        super().testing_routine(
            metric, metric_args, history_monitor, before_train
        )


    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """Return unique target labels from the training and testing datasets.

        Returns:
            Numpy array containing the unique values from the targets wrapped
            in the training and testing NPDataLoader instances.
        """
        return np.unique([t for loader in (self.training_data_loader, self.testing_data_loader) for d, t in loader])

    def save(
            self,
            filename: str,
            params: Union[None, List[float], Dict[str, np.ndarray]] = None
    ) -> None:
        """Save the wrapped model and its trainable parameters.

        This method is designed for parameter communication. It
        uses the joblib.dump function, which in turn uses pickle
        to serialize the model. Note that unpickling objects can
        lead to arbitrary code execution; hence use with care.

        Args:
            filename: Path to the output file.
            params: Model parameters to enforce and save.
                This may either be a {name: array} parameters dict, or a
                nested dict that stores such a parameters dict under the
                'model_params' key (in the context of the Round class).

        Notes:
            Save can be called from Job or Round.
            * From [`Round`][fedbiomed.node.round.Round] it is called with params (as a complex dict).
            * From [`Job`][fedbiomed.researcher.job.Job] it is called with no params in constructor, and
                with params in update_parameters.
        """

        if params is None:
            params = {"model_params": self.after_training_params()}
        elif "model_params" not in params:
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605}: params should contain `model_params`"
            )

        # Save the wrapped model (using joblib, hence pickle).
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False, indent=4)

    def load(
            self,
            filename: str,
            update_model: bool = True
    ) -> Union[BaseEstimator, Dict[str, Dict[str, np.ndarray]]]:
        """Load a scikit-learn model dump, overwriting the wrapped model.

        This method uses the joblib.load function, which in turn uses
        pickle to deserialize the model. Note that unpickling objects
        can lead to arbitrary code execution; hence use with care.

        This function updates the `_model` private attribute with the
        loaded instance, and returns either that same model or a dict
        wrapping its trainable parameters.

        Args:
            filename: The path to the pickle file to load.
            update_model: Whether to return the model's parameters wrapped as a dict rather than the
                model instance.

        !!! info 'Notes'
            Load can be called from a Job or Round:
            * From [`Round`][fedbiomed.node.round.Round] it is called to return the model.
            * From [`Job`][fedbiomed.researcher.job.Job] it is called with to return its parameters dict.

        Returns:
            Dictionary with the loaded parameters.
        """
        # Deserialize the dump, type-check the instance and assign it.
        with open(filename, "r") as file:
            content = file.read()
            params = json.loads(content)
            file.close()

        if update_model:
            model_params = params["model_params"]
            self._model.set_weights(model_params)

        return params

    def type(self) -> TrainingPlans:
        """Getter for training plan type """
        return self.__type

    def after_training_params(
            self,
            vector: bool = False
    ) -> Union[List[float], Dict[str, np.ndarray]]:
        """Return the wrapped model's trainable parameters' current values.

        This method returns a dict containing parameters that need
        to be reported back and aggregated in a federated learning
        setting.

        Args:


        Returns:
            params: the trained parameters to aggregate.
            vector: Returns the vectorized parameters ff the vector argument is `True`
        """

        model_params = self._model.get_weights()

        if vector:
            params = []
            for key, param in model_params.items():
                params.extend(param.flatten().astype(float).tolist())
        else:
            # Convert to list
            params = {key: param.astype(float).tolist() for key, param in model_params.items()}

        return params

    def convert_vector_to_parameters(self, vec: List[float]):
        """Converts given float vector to numpy typed params

        Args:
            vec: List of flatten model parameters
        """

        vector = np.array(vec)

        params = {key: getattr(self._model, key) for key in self._param_list}
        pointer = 0

        for key, param in params.items():
            num_param = param.size
            params[key] = vector[pointer: pointer + num_param].reshape(param.shape)

            pointer += num_param

        return params
