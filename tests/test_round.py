import builtins
import copy
import inspect
import logging
import os
import tempfile
import unittest
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, create_autospec, patch
from fedbiomed.common.optimizers.generic_optimizers import DeclearnOptimizer
from fedbiomed.common.serializer import Serializer


#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from testsupport.fake_training_plan import FakeModel
from testsupport.fake_message import FakeMessages
from testsupport.fake_uuid import FakeUuid
from testsupport.testing_data_loading_block import ModifyGetItemDP, LoadingBlockTypesForTesting
from testsupport import fake_training_plan

import torch
from declearn.optimizer.modules import YogiModule, ScaffoldClientModule
from declearn.optimizer.regularizers import RidgeRegularizer

from fedbiomed.common.constants import DatasetTypes, TrainingPlans
from fedbiomed.common.data import DataManager, DataLoadingPlanMixin, DataLoadingPlan
from fedbiomed.common.exceptions import FedbiomedOptimizerError, FedbiomedRoundError
from fedbiomed.common.logger import logger
from fedbiomed.common.models import TorchModel
from fedbiomed.common.optimizers import BaseOptimizer, Optimizer
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.node.environ import environ
from fedbiomed.node.round import Round


# Needed to access length of dataset from Round class
class FakeLoader:
    dataset = [1, 2, 3, 4, 5]


class TestRound(NodeTestCase):

    # values and attributes for dummy classes
    URL_MSG = 'http://url/where/my/file?is=True'

    @classmethod
    def setUpClass(cls):
        """Sets up values in the test once """

        # Sets mock environ for the test -------------------
        super().setUpClass()
        # --------------------------------------------------

        # we define here common side effect functions
        def node_msg_side_effect(msg: Dict[str, Any]) -> Dict[str, Any]:
            fake_node_msg = FakeMessages(msg)
            return fake_node_msg

        cls.node_msg_side_effect = node_msg_side_effect

    @patch('fedbiomed.common.repository.Repository.__init__')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.__init__')
    def setUp(self,
              tp_security_manager_patch,
              repository_patch):
        tp_security_manager_patch.return_value = None
        repository_patch.return_value = None

        # instantiate logger (we will see if exceptions are logged)
        # we are setting the logger level to "ERROR" to output
        # logs messages
        logger.setLevel("ERROR")
        # instanciate Round class
        self.r1 = Round(training_plan_url='http://somewhere/where/my/model?is_stored=True',
                        training_plan_class='MyTrainingPlan',
                        params_url='https://url/to/model/params?ok=True',
                        training_kwargs={},
                        training=True
                        )
        params = {'path': 'my/dataset/path',
                  'dataset_id': 'id_1234'}
        self.r1.dataset = params
        self.r1.job_id = '1234'
        self.r1.researcher_id = '1234'
        dummy_monitor = MagicMock()
        self.r1.history_monitor = dummy_monitor

        self.r2 = Round(training_plan_url='http://a/b/c/model',
                        training_plan_class='another_training_plan',
                        params_url='https://to/my/model/params',
                        training_kwargs={},
                        training=True
                        )
        self.r2.dataset = params
        self.r2.history_monitor = dummy_monitor


    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_outgoing_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_01_run_model_training_normal_case(self,
                                                     uuid_patch,
                                                     repository_download_patch,
                                                     tp_security_manager_patch,
                                                     import_module_patch,
                                                     serialize_load_patch,
                                                     repository_upload_patch,
                                                     node_msg_patch,
                                                     mock_split_test_train_data,
                                                     ):
        """tests correct execution and message parameters.
        Besides  tests the training time.
         """
        # Tests details:
        # - Test 1: normal case scenario where no model_kwargs has been passed during model instantiation
        # - Test 2: normal case scenario where model_kwargs has been passed when during model instantiation

        FakeModel.SLEEPING_TIME = 1

        # initalisation of side effect function

        def repository_side_effect(training_plan_url: str, model_name: str):
            return 200, 'my_python_model'

        class FakeModule:
            MyTrainingPlan = FakeModel
            another_training_plan = FakeModel

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_test_train_data.return_value = (FakeLoader, FakeLoader)

        # test 1: case where argument `model_kwargs` = None
        # action!
        msg_test1 = self.r1.run_model_training()

        # check results
        self.assertTrue(msg_test1.get('success', False))
        serialize_load_patch.assert_called_once()
        self.assertEqual(msg_test1.get('params_url', False), TestRound.URL_MSG)
        self.assertEqual(msg_test1.get('command', False), 'train')

        # This test is not relevant since it just tests SLEEPING_TIME added in FakeModel
        # and it fails in macosx-m1
        # timing test - does not always work with self.assertAlmostEqual
        # self.assertGreaterEqual(
        #     msg_test1.get('timing', {'rtime_training': 0}).get('rtime_training'),
        #     FakeModel.SLEEPING_TIME
        # )
        # self.assertLess(
        #     msg_test1.get('timing', {'rtime_training': 0}).get('rtime_training'),
        #     FakeModel.SLEEPING_TIME * 1.1
        # )

        # test 2: redo test 1 but with the case where `model_kwargs` != None
        FakeModel.SLEEPING_TIME = 0
        self.r2.model_kwargs = {'param1': 1234,
                                'param2': [1, 2, 3, 4],
                                'param3': None}
        serialize_load_patch.reset_mock()
        msg_test2 = self.r2.run_model_training()

        # check values in message (output of `run_model_training`)
        self.assertTrue(msg_test2.get('success', False))
        serialize_load_patch.assert_called_once()
        self.assertEqual(TestRound.URL_MSG, msg_test2.get('params_url', False))
        self.assertEqual('train', msg_test2.get('command', False))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.common.serializer.Serializer.dump')
    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_02_run_model_training_correct_model_calls(self,
                                                             uuid_patch,
                                                             repository_download_patch,
                                                             tp_security_manager_patch,
                                                             import_module_patch,
                                                             serialize_load_patch,
                                                             serialize_dump_patch,
                                                             repository_upload_patch,
                                                             node_msg_patch,
                                                             mock_split_train_and_test_data):
        """tests if all methods of `model` have been called after instanciating
        (in run_model_training)"""
        # `run_model_training`, when no issues are found
        # methods tested:
        #  - model.load
        #  - model.save
        #  - model.training_routine
        #  - model.after_training_params
        #  - model.set_dataset_path

        FakeModel.SLEEPING_TIME = 0
        MODEL_NAME = "my_model"
        MODEL_PARAMS = {"coef": [1, 2, 3, 4]}

        class FakeModule:
            MyTrainingPlan = FakeModel

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, MODEL_NAME)
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (FakeLoader, FakeLoader)

        self.r1.training_kwargs = {}
        self.r1.dataset = {'path': 'my/dataset/path',
                           'dataset_id': 'id_1234'}

        # arguments of `save` method
        _model_filename = os.path.join(environ["TMP_DIR"], "node_params_1234.mpk")
        _model_results = {
            'researcher_id': self.r1.researcher_id,
            'job_id': self.r1.job_id,
            'model_weights': MODEL_PARAMS,
            'node_id': environ['NODE_ID'],
            'optimizer_args': {},
            'encrypted': False,
            'optim_aux_var': {},
        }

        # define context managers for each model method
        # we are mocking every methods of our dummy model FakeModel,
        # and we will check if there are called when running
        # `run_model_training`
        with (
                patch.object(FakeModel, 'set_dataset_path') as mock_set_dataset,
                patch.object(FakeModel, 'training_routine') as mock_training_routine,
                patch.object(FakeModel, 'after_training_params', return_value=MODEL_PARAMS) as mock_after_training_params,  # noqa
        ):
            msg = self.r1.run_model_training()
            self.assertTrue(msg.get("success"))

            # Check that the model weights were loaded.
            serialize_load_patch.assert_called_once()

            # Check set train and test data split function is called
            # Set dataset is called in set_train_and_test_data
            # mock_set_dataset.assert_called_once_with(self.r1.dataset.get('path'))
            mock_split_train_and_test_data.assert_called_once()

            # Since set training data return None, training_routine should be called as None
            mock_training_routine.assert_called_once_with( history_monitor=self.r1.history_monitor,
                                                           node_args=None)

            # Check that the model weights were saved.
            mock_after_training_params.assert_called_once()
            serialize_dump_patch.assert_called_once_with(_model_results, _model_filename)

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_03_test_run_model_training_with_real_model(self,
                                                              uuid_patch,
                                                              repository_download_patch,
                                                              serialize_load_patch,
                                                              tp_security_manager_patch,
                                                              repository_upload_patch,
                                                              node_msg_patch,
                                                              mock_split_train_and_test_data):
        """tests normal case scenario with a real model file"""
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = (True, True)

        # create dummy_model
        dummy_training_plan_test = "\n".join([
            "from testsupport.fake_training_plan import FakeModel",
            "class MyTrainingPlan(FakeModel):",
            "    dataset = [1, 2, 3, 4]",
            "    def set_data_loaders(self, *args, **kwargs):",
            "       self.testing_data_loader = MyTrainingPlan",
            "       self.training_data_loader = MyTrainingPlan",
        ])

        module_file_path = os.path.join(environ['TMP_DIR'],
                                        'training_plan_' + str(FakeUuid.VALUE) + '.py')

        # creating file for toring dummy training plan
        with open(module_file_path, "w", encoding="utf-8") as file:
            file.write(dummy_training_plan_test)

        # action
        msg_test = self.r1.run_model_training()
        # checks
        serialize_load_patch.assert_called_once_with('my_python_model')
        self.assertTrue(msg_test.get('success', False))
        self.assertEqual(TestRound.URL_MSG, msg_test.get('params_url', False))
        self.assertEqual('train', msg_test.get('command', False))

        # remove model file
        os.remove(module_file_path)

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_rounds_04_run_model_training_bad_http_status(self,
                                                          uuid_patch,
                                                          repository_download_patch,
                                                          tp_security_manager_patch,
                                                          repository_upload_patch,
                                                          node_msg_patch,
                                                          mock_split_train_and_test_data):
        """tests failures and exceptions during the download file process
        (in run_model_training)"""
        # Tests details:
        # Test 1: tests case where downloading model file fails
        # Test 2: tests case where downloading model paraeters fails
        FakeModel.SLEEPING_TIME = 0

        # initalisation of side effects functions

        def download_repo_answers_gene() -> int:
            """Generates different values of connections:
            First one is HTTP code 200, second one is HTTP code 404
            Raises: StopIteration, if called more than twice
            """
            for i in [200, 404]:
                yield i

        def repository_side_effect_test_1(training_plan_url: str, model_name: str):
            """Returns HTTP 404 error, mimicking an error happened during
            download process"""
            return 404, 'my_python_model'

        download_repo_answers_iter = iter(download_repo_answers_gene())
        # initialisation of patchers

        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect_test_1
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

        # test 1: case where first call to `Repository.download` generates HTTP
        # status 404 (when downloading model_file)
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_1 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_1.get('msg'))

        self.assertFalse(msg_test_1.get('success', True))

        # test 2: case where second call to `Repository.download` generates HTTP
        # status 404 (when downloading params_file)
        # overwriting side effect function for second test:
        def repository_side_effect_2(training_plan_url: str, model_name: str):
            """Returns different values when called
            First call: returns (200, 'my_python_model') mimicking a first download
                that happened without encoutering any issues
            Second call: returns (404, 'my_python_model') mimicking a second download
                that failed
            Third Call (or more): raises StopIteration (due to generator)

            """
            val = next(download_repo_answers_iter)
            return val, 'my_python_model'

        repository_download_patch.side_effect = repository_side_effect_2

        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_2 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_2.get('msg'))
        self.assertFalse(msg_test_2.get('success', True))

        # test 3: check if unknown exception is raised and caught during the download
        # files process

        def repository_side_effect_3(training_plan_url: str, model_name: str):
            raise Exception('mimicking an error during download files process')

        repository_download_patch.side_effect = repository_side_effect_3
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})

        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test_3 = self.r1.run_model_training()

        # checks
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_3.get('msg'))
        self.assertFalse(msg_test_3.get('success', True))

    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_05_run_model_training_model_not_approved(self,
                                                            uuid_patch,
                                                            repository_download_patch,
                                                            tp_security_manager_patch):
        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (False, {'name': "model_name"})
        # action
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            msg_test = self.r1.run_model_training()
            # checks
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test.get('msg'))

        self.assertFalse(msg_test.get('success', True))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_06_run_model_training_import_error(self,
                                                      uuid_patch,
                                                      repository_download_patch,
                                                      tp_security_manager_patch,
                                                      repository_upload_patch,
                                                      node_msg_patch,
                                                      mock_split_train_and_test_data):
        """tests case where the import/loading of the model have failed"""

        FakeModel.SLEEPING_TIME = 0

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

        # test 1: tests raise of exception during model import
        def exec_side_effect(*args, **kwargs):
            """Overriding the behaviour of `exec` builitin function,
            and raises an Exception"""
            raise Exception("mimicking an exception happening when loading file")

        # patching builtin objects & looking for generated logs
        # NB: this is the only way I have found to use
        # both patched bulitins functions and assertLogs
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value = None),
              patch.object(builtins, 'eval') as eval_patch):
            eval_patch.side_effect = exec_side_effect
            msg_test_1 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_1.get('msg'))

        self.assertFalse(msg_test_1.get('success', True))

        # test 2: tests raise of Exception during loading parameters
        # into model instance

        # Here we creating a new class inheriting from the FakeModel,
        # but overriding `load` through classes inheritance
        # when `load` is called, an Exception will be raised
        #
        class FakeModelRaiseExceptionWhenLoading(FakeModel):
            def load(self, **kwargs):
                """Mimicks an exception happening in the `load`
                method

                Raises:
                    Exception:
                """
                raise Exception('mimicking an error happening during model training')

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value=FakeModelRaiseExceptionWhenLoading)
              ):

            msg_test_2 = self.r1.run_model_training()

        # checks:
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training`
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_2.get('msg'))
        self.assertFalse(msg_test_2.get('success', True))

        # test 3: tests raise of Exception during model training
        # into model instance

        # Here we are creating a new class inheriting from the FakeModel,
        # but overriding `training_routine` through classes inheritance
        # when `training_routine` is called, an Exception will be raised
        #
        class FakeModelRaiseExceptionInTraining(FakeModel):
            def training_routine(self, **kwargs):
                """Mimicks an exception happening in the `training_routine`
                method

                Raises:
                    Exception:
                """
                raise Exception('mimicking an error happening during model training')

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value= FakeModelRaiseExceptionInTraining)):
            msg_test_3 = self.r1.run_model_training()

        # checks :
        # check if error message generated and logged is the same as the one
        # collected in the output of `run_model_training``
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test_3.get('msg'))

        self.assertFalse(msg_test_3.get('success', True))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_07_run_model_training_upload_file_fails(self,
                                                           uuid_patch,
                                                           repository_download_patch,
                                                           tp_security_manager_patch,
                                                           repository_upload_patch,
                                                           node_msg_patch,
                                                           mock_split_train_and_test_data):

        """tests case where uploading model parameters file fails"""
        FakeModel.SLEEPING_TIME = 0

        # declaration of side effect functions

        def upload_side_effect(*args, **kwargs):
            """Raises an exception when calling this function"""
            raise Exception("mimicking an error happening during upload")

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, 'my_python_model')
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        repository_upload_patch.side_effect = upload_side_effect
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

        # action
        with (self.assertLogs('fedbiomed', logging.ERROR) as captured,
              patch.object(builtins, 'exec', return_value=None),
              patch.object(builtins, 'eval', return_value=FakeModel)
              ):
            msg_test = self.r1.run_model_training()

        # checks if message logged is the message returned as a reply
        self.assertEqual(
            captured.records[-1].getMessage(),
            msg_test.get('msg'))

        self.assertFalse(msg_test.get('success', True))

    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('builtins.eval')
    @patch('builtins.exec')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_08_run_model_training_bad_training_argument(self,
                                                               uuid_patch,
                                                               repository_download_patch,
                                                               tp_security_manager_patch,
                                                               builtin_exec_patch,
                                                               builtin_eval_patch,
                                                               repository_upload_patch,
                                                               node_msg_patch,
                                                               mock_split_train_and_test_data):
        """tests case where training plan contains node_side arguments"""
        FakeModel.SLEEPING_TIME = 1

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, "my_model")
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        builtin_exec_patch.return_value = None
        builtin_eval_patch.return_value = FakeModel
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_train_and_test_data.return_value = None

    @patch('inspect.signature')
    def test_round_09_data_loading_plan(self,
                                        patch_inspect_signature,
                                        ):
        """Test that Round correctly handles a DataLoadingPlan during training"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super().__init__()

            def __getitem__(self, item):
                return self.apply_dlb('orig-value', LoadingBlockTypesForTesting.MODIFY_GETITEM)

            @staticmethod
            def get_dataset_type() -> DatasetTypes:
                return DatasetTypes.TEST

        patch_inspect_signature.return_value = inspect.Signature(parameters={})

        my_dataset = MyDataset()
        data_loader_mock = MagicMock()
        data_loader_mock.dataset = my_dataset

        data_manager_mock = MagicMock(spec=DataManager)
        data_manager_mock.split = MagicMock()
        data_manager_mock.split.return_value = (data_loader_mock, None)
        data_manager_mock.dataset = my_dataset

        r3 = Round(training_kwargs={})
        r3.initialize_validate_training_arguments()
        r3.training_plan = MagicMock()
        r3.training_plan.training_data.return_value = data_manager_mock

        training_data_loader, _ = r3._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'orig-value')

        dlp = DataLoadingPlan({LoadingBlockTypesForTesting.MODIFY_GETITEM: ModifyGetItemDP()})
        r4 = Round(training_kwargs={},
                   dlp_and_loading_block_metadata=dlp.serialize()
                   )
        r4.initialize_validate_training_arguments()
        r4.training_plan = MagicMock()
        r4.training_plan.training_data.return_value = data_manager_mock

        training_data_loader, _ = r4._split_train_and_test_data(test_ratio=0.)
        dataset = training_data_loader.dataset
        self.assertEqual(dataset[0], 'modified-value')

    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_10_download_aggregator_args(
        self, uuid_patch, repository_download_patch, serializer_load_patch,
    ):
        uuid_patch.return_value = FakeUuid()

        repository_download_patch.side_effect = ((200, "my_model_var"+ str(i)) for i in range(3, 5))
        serializer_load_patch.side_effect = (i for i in range(3, 5))
        success, _ = self.r1.download_aggregator_args()
        self.assertEqual(success, True)
        # if attribute `aggregator_args` is None, then do nothing
        repository_download_patch.assert_not_called()

        aggregator_args = {'var1': 1,
                            'var2': [1, 2, 3, 4],
                            'var3': {'url': 'http://to/var/3',},
                            'var4': {'url': 'http://to/var/4'}}
        self.r1.aggregator_args = copy.deepcopy(aggregator_args)

        success, error_msg = self.r1.download_aggregator_args()
        self.assertEqual(success, True)
        self.assertEqual(error_msg, '')

        for var in ('var1', 'var2'):
            self.assertEqual(self.r1.aggregator_args[var], aggregator_args[var])

        for var in ('var3', 'var4'):
            serializer_load_patch.assert_any_call(f"my_model_{var}")
            self.assertEqual(self.r1.aggregator_args[var], int(var[-1]))

    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    def test_round_11_download_file(self, uuid_patch, repository_download_patch):
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.return_value = (200, "my_model")
        file_path = 'path/to/my/downloaded/files'
        success, param_path, msg = self.r1.download_file('http://some/url/to/some/files', file_path)
        self.assertEqual(success, True)
        self.assertEqual(param_path, 'my_model')
        self.assertEqual(msg, '')

    @patch("fedbiomed.node.round.BPrimeManager.get")
    @patch("fedbiomed.node.round.SKManager.get")
    def test_round_12_configure_secagg(self,
                                       servkey_get,
                                       biprime_get
                                       ):
        """Tests round secure aggregation configuration"""

        servkey_get.return_value = {"context": {}}
        biprime_get.return_value = {"context": {}}

        environ["SECURE_AGGREGATION"] = True

        result = self.r1._configure_secagg(
            secagg_random=1.5,
            secagg_biprime_id='123',
            secagg_servkey_id='123'
        )
        self.assertTrue(result)

        result = self.r1._configure_secagg(
            secagg_random=None,
            secagg_biprime_id=None,
            secagg_servkey_id=None
        )
        self.assertFalse(result)

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id="1234",
                secagg_servkey_id=None)

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id="1234",
                secagg_servkey_id="1223")

        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id=None,
                secagg_servkey_id="1223")

        with self.assertRaises(FedbiomedRoundError):
            servkey_get.return_value = None
            biprime_get.return_value = {"context": {}}
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )

        with self.assertRaises(FedbiomedRoundError):
            servkey_get.return_value = {"context": {}}
            biprime_get.return_value = None
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )

        # If node forces using secagg
        environ["SECURE_AGGREGATION"] = True
        environ["FORCE_SECURE_AGGREGATION"] = True
        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=None,
                secagg_biprime_id=None,
                secagg_servkey_id=None
            )

        # If secagg is not activated
        environ["SECURE_AGGREGATION"] = False
        environ["FORCE_SECURE_AGGREGATION"] = False
        with self.assertRaises(FedbiomedRoundError):
            self.r1._configure_secagg(
                secagg_random=1.5,
                secagg_biprime_id='123',
                secagg_servkey_id='123'
            )



    @patch('fedbiomed.node.round.Round._split_train_and_test_data')
    @patch('fedbiomed.common.message.NodeMessages.format_incoming_message')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.common.serializer.Serializer.load')
    @patch('importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.download_file')
    @patch('uuid.uuid4')
    @patch("fedbiomed.node.round.BPrimeManager.get")
    @patch("fedbiomed.node.round.SKManager.get")
    def test_round_13_run_model_training_secagg(self,
                                                servkey_get,
                                                biprime_get,
                                                uuid_patch,
                                                repository_download_patch,
                                                tp_security_manager_patch,
                                                import_module_patch,
                                                serialize_load_patch,
                                                repository_upload_patch,
                                                node_msg_patch,
                                                mock_split_test_train_data):
        """tests correct execution and message parameters.
        Besides  tests the training time.
         """
        # Tests details:
        # - Test 1: normal case scenario where no model_kwargs has been passed during model instantiation
        # - Test 2: normal case scenario where model_kwargs has been passed when during model instantiation

        FakeModel.SLEEPING_TIME = 1

        # initalisation of side effect function

        def repository_side_effect(training_plan_url: str, model_name: str):
            return 200, 'my_python_model'

        class M(FakeModel):
            def after_training_params(self, flatten):
                return [0.1,0.2,0.3,0.4,0.5]

        class FakeModule:
            MyTrainingPlan = M
            another_training_plan = M

        # initialisation of patchers
        uuid_patch.return_value = FakeUuid()
        repository_download_patch.side_effect = repository_side_effect
        tp_security_manager_patch.return_value = (True, {'name': "model_name"})
        import_module_patch.return_value = FakeModule
        repository_upload_patch.return_value = {'file': TestRound.URL_MSG}
        node_msg_patch.side_effect = TestRound.node_msg_side_effect
        mock_split_test_train_data.return_value = (FakeLoader, FakeLoader)


        # Secagg configuration
        servkey_get.return_value = {"parties": ["r-1", "n-1", "n-2"],  "context" : {"server_key": 123445}}
        biprime_get.return_value = {"parties": ["r-1", "n-1", "n-2"], "context" : {"biprime": 123445}}
        environ["SECURE_AGGREGATION"] = True
        environ["FORCE_SECURE_AGGREGATION"] = True

        msg_test1 = self.r1.run_model_training(secagg_arguments={
            'secagg_random': 1.12,
            'secagg_servkey_id': '1234',
            'secagg_biprime_id': '1234',
        })

        # Back to normal
        environ["SECURE_AGGREGATION"] = False
        environ["FORCE_SECURE_AGGREGATION"] = False

    @patch("uuid.uuid4")
    @patch('fedbiomed.node.round.importlib.import_module')
    @patch('fedbiomed.node.training_plan_security_manager.TrainingPlanSecurityManager.check_training_plan_status')
    @patch('fedbiomed.common.repository.Repository.upload_file')
    @patch('fedbiomed.common.repository.Repository.download_file')
    def test_round_14_run_model_training_optimizer_aux_var_error(self,
                                                                 patch_repo_download_file,
                                                                 patch_repo_upload_file,
                                                                 tp_security_manager_patch,
                                                                 importlib_importmodule_patch,
                                                                 patch_uuid):
        aux_var = {'scaffold': {'delta': 'some incorrect value for scaffold'}}
        def create_file(file_path: str, aux_var: Dict):
            Serializer.dump(aux_var, file_path)
        repo_uploaded = []
        def repo_upload_side_effect(file:str):
            repo_uploaded.append(file)
            return {'file': file}
        
        # patches
        patch_repo_upload_file.side_effect = repo_upload_side_effect
        patch_uuid.return_value = FakeUuid()
        tp_security_manager_patch.return_value = (True, {'name': 'my_node_id'})
        importlib_importmodule_patch.return_value = fake_training_plan

        with tempfile.TemporaryDirectory() as tmp_directory:
            fake_path_towards_tp = os.path.join('path', 'to', 'my', 'tp', 'file.mpk')
            aux_var_tmp_path = os.path.join(tmp_directory, 'aux_var.mpk')
            model_param_tmp_path = os.path.join(tmp_directory, 'model_params.mpk')
            patch_repo_download_file.side_effect = (
                    (200, fake_path_towards_tp),
                    (200, model_param_tmp_path),
                    (200, aux_var_tmp_path)
                    )
            create_file(model_param_tmp_path, {'model_weights': [1,2,3,4,5]},)
            create_file(aux_var_tmp_path, aux_var)

            # creating Round
            rnd = Round(
                training_kwargs={'optimizer_args': {'lr': .1234}}
            )
            rnd.researcher_id = 'researcher_id_1234'
            rnd.job_id = 'job_id_1234'
            rnd.training_plan_class = "DeclearnAuxVarModel"
            rnd.dataset = {'dataset_id': 'dataset_id_1234',
                           'path': os.path.join('path', 'to', 'my', 'dataset')}
            rnd.aux_var_urls = ['http://url/to/my/file']

            # configure optimizer
            lr = .1234
            optim_node = Optimizer(lr=lr, modules=[ScaffoldClientModule(), YogiModule()],
                                   regularizers=[RidgeRegularizer()])


            dec_node_optim = DeclearnOptimizer(TorchModel(torch.nn.Linear(3, 1)), optim_node)

            fake_training_plan.DeclearnAuxVarModel.OPTIM = dec_node_optim
            fake_training_plan.DeclearnAuxVarModel.TYPE = TrainingPlans.TorchTrainingPlan
            
            # action
            rnd_reply = rnd.run_model_training()

            self.assertIn("TrainingPlan Optimizer failed to ingest the provided auxiliary variables",
                          rnd_reply['msg'])

    @patch("uuid.uuid4")
    @patch("fedbiomed.common.serializer.Serializer.load")
    @patch("fedbiomed.common.repository.Repository.download_file")
    def test_round_15_download_optimizer_aux_var(
        self,
        patch_repo_download_file,
        patch_serializer_load,
        patch_uuid,
    ):
        """Test that 'download_optimizer_aux_var' works properly."""
        # Set up a Round with two aux var urls, and patch downloading tools.
        rnd = Round(aux_var_urls=["fake_url_1", "fake_url_2"])
        patch_uuid.side_effect = ("uuid_1", "uuid_2")

        def fake_repository_download(url: str, path: str) -> Tuple[int, str]:
            return 200, f"{url}-{path}"

        def fake_serializer_load(path: str) -> Dict[str, Dict[str, Any]]:
            return {path: {"key": "val"}}

        patch_repo_download_file.side_effect = fake_repository_download
        patch_serializer_load.side_effect = fake_serializer_load
        # Run the method.
        success, error_msg = rnd.download_optimizer_aux_var()
        # Verify its outputs and side effect.
        self.assertTrue(success)
        self.assertEqual(error_msg, "")
        aux_var = getattr(rnd, "_optim_aux_var")
        expected = {
            "fake_url_1-aux_var_uuid_1.mpk": {"key": "val"},
            "fake_url_2-aux_var_uuid_2.mpk": {"key": "val"},
        }
        self.assertDictEqual(aux_var, expected)

    @patch("uuid.uuid4")
    @patch("fedbiomed.common.repository.Repository.download_file")
    def test_round_16_download_optimizer_aux_var_download_error(
        self,
        patch_repo_download_file,
        patch_uuid,
    ):
        """Test that 'download_optimizer_aux_var' fails properly on 404 error."""
        # Set up a Round with an aux var url, and failing downloader.
        fake_url = "fake_url"
        rnd = Round(aux_var_urls=[fake_url])
        patch_uuid.return_value = "uuid"
        patch_repo_download_file.return_value = (404, "fake_path")
        # Run the method.
        success, error_msg = rnd.download_optimizer_aux_var()
        # Verify its outputs and side effect.
        self.assertFalse(success)
        self.assertTrue(fake_url in error_msg)
        aux_var = getattr(rnd, "_optim_aux_var")
        self.assertDictEqual(aux_var, {})
        # Verify that the download instructions matched expectations.
        patch_repo_download_file.assert_called_once_with(
            fake_url, "aux_var_uuid.mpk"
        )

    @patch("uuid.uuid4")
    @patch("fedbiomed.common.serializer.Serializer.load")
    @patch("fedbiomed.common.repository.Repository.download_file")
    def test_round_17_download_optimizer_aux_var_serializer_error(
        self,
        patch_repo_download_file,
        patch_serializer_load,
        patch_uuid,
    ):
        """Test that 'download_optimizer_aux_var' fails properly on Serializer error."""
        # Set up a Round with an aux var url, and failing de-serializer.
        fake_url = "fake_url"
        fake_path = "fake_path.mpk"
        fake_err = "fake serializer error message"
        rnd = Round(aux_var_urls=[fake_url])
        patch_uuid.return_value = "uuid"
        patch_repo_download_file.return_value = (200, fake_path)
        patch_serializer_load.side_effect = TypeError(fake_err)
        # Run the method.
        success, error_msg = rnd.download_optimizer_aux_var()
        # Verify its outputs and side effect.
        self.assertFalse(success)
        self.assertTrue(fake_err in error_msg)
        aux_var = getattr(rnd, "_optim_aux_var")
        self.assertDictEqual(aux_var, {})
        # Verify that the download instructions matched expectations.
        patch_repo_download_file.assert_called_once_with(
            fake_url, "aux_var_uuid.mpk"
        )
        patch_serializer_load.assert_called_once_with(fake_path)

    def test_round_18_process_optim_aux_var(self):
        """Test that 'process_optim_aux_var' works properly."""
        rnd = Round()
        # Set up a mock BaseOptimizer with an attached Optimizer.
        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        # Attach the former to the Round's mock TrainingPlan.
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Attach fake auxiliary variables (as though pre-downloaded).
        fake_aux_var = {"module": {"key": "val"}}
        setattr(rnd, "_optim_aux_var", fake_aux_var)
        # Call the tested method and verify its outputs and effects.
        msg = rnd.process_optim_aux_var()
        self.assertEqual(msg, "")
        mock_optim.set_aux.assert_called_once_with(fake_aux_var)

    def test_round_19_process_optim_aux_var_without_aux_var(self):
        """Test that 'process_optim_aux_var' exits properly without aux vars."""
        # Set up a Round with a mock Optimizer attached, but no aux vars.
        rnd = Round()
        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it exits without effects.
        msg = rnd.process_optim_aux_var()
        self.assertEqual(msg, "")
        mock_optim.set_aux.assert_not_called()

    def test_round_20_process_optim_aux_var_without_base_optimizer(self):
        """Test that 'process_optim_aux_var' documents missing BaseOptimizer."""
        # Set up a Round with fake aux_vars, but no BaseOptimizer.
        rnd = Round()
        setattr(rnd, "_optim_aux_var", {"module": {"key": "val"}})
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = None
        # Call the tested method, verifying that it returns an error.
        msg = rnd.process_optim_aux_var()
        self.assertTrue("TrainingPlan does not hold a BaseOptimizer" in msg)
        self.assertIsInstance(msg, str)

    def test_round_21_process_optim_aux_var_without_optimizer(self):
        """Test that 'process_optim_aux_var' documents missing Optimizer."""
        # Set up a Round with aux vars, but a non-Optimizer optimizer.
        rnd = Round()
        setattr(rnd, "_optim_aux_var", {"module": {"key": "val"}})
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = MagicMock()  # not a declearn-based Optimizer
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it returns an error.
        msg = rnd.process_optim_aux_var()
        self.assertTrue("does not manage a compatible Optimizer" in msg)
        self.assertIsInstance(msg, str)

    def test_round_22_process_optim_aux_var_with_optimizer_error(self):
        """Test that 'process_optim_aux_var' documents 'Optimizer.set_aux' error."""
        # Set up a Round with fake pre-downloaded aux vars.
        rnd = Round()
        fake_aux_var = {"module": {"key": "val"}}
        setattr(rnd, "_optim_aux_var", fake_aux_var)
        # Set up a mock BaseOptimizer with an attached failing Optimizer.
        mock_optim = create_autospec(Optimizer, instance=True)
        fake_error = "fake FedbiomedOptimizerError on 'set_aux' call"
        mock_optim.set_aux.side_effect = FedbiomedOptimizerError(fake_error)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = mock_optim
        # Attach the former to the Round's mock TrainingPlan.
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method, verifying that it returns an error.
        msg = rnd.process_optim_aux_var()
        self.assertTrue(fake_error in msg)
        mock_optim.set_aux.assert_called_once_with(fake_aux_var)

    def test_round_23_collect_optim_aux_var(self):
        """Test that 'collect_optim_aux_var' works properly with an Optimizer."""
        # Set up a Round with an attached mock Optimizer.
        rnd = Round()
        mock_optim = create_autospec(Optimizer, instance=True)
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        # why not using DeclearnOptimizer?
        mock_b_opt.optimizer = mock_optim
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method and verify its outputs.
        aux_var = rnd.collect_optim_aux_var()
        self.assertEqual(aux_var, mock_optim.get_aux.return_value)
        mock_optim.get_aux.assert_called_once()

    def test_round_24_collect_optim_aux_var_without_optimizer(self):
        """Test that 'collect_optim_aux_var' works properly without an Optimizer."""
        # Set up a Round with a non-Optimizer optimizer.
        rnd = Round()
        mock_b_opt = create_autospec(BaseOptimizer, instance=True)
        mock_b_opt.optimizer = MagicMock()  # non-declearn-based object
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = mock_b_opt
        # Call the tested method and verify its outputs.
        aux_var = rnd.collect_optim_aux_var()
        self.assertEqual(aux_var, {})

    def test_round_25_collect_optim_aux_var_without_base_optimizer(self):
        """Test that 'collect_optim_aux_var' fails without a BaseOptimizer."""
        # Set up a Round without a BaseOptimizer.
        rnd = Round()
        rnd.training_plan = create_autospec(BaseTrainingPlan, instance=True)
        rnd.training_plan.optimizer.return_value = None
        # Verify that aux-var collection raises.
        self.assertRaises(FedbiomedRoundError, rnd.collect_optim_aux_var)

    # add a test with : shared and node specific auxiliary avraibales

    def test_round_26_split_train_and_test_data_raises_exceptions(self):
        """Test that _split_train_and_test_data raises correct exceptions"""
        mock_training_plan = MagicMock()
        def foo_takes_an_argument(x):
            return x
        mock_training_plan.training_data = foo_takes_an_argument
        mock_training_plan.type.return_value = 'tp type'
        r = Round()
        r.training_plan = mock_training_plan
        with self.assertRaises(FedbiomedRoundError):
            r._split_train_and_test_data()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
