# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


import time

from typing import List, Union, Optional, Dict

import numpy as np
from gmpy2 import mpz

from fedbiomed.common.exceptions import FedbiomedSecaggCrypterError
from fedbiomed.common.constants import ErrorNumbers, VEParameters
from fedbiomed.common.logger import logger
from fedbiomed.common.secagg._prf import PRF

from ._jls import JoyeLibert, \
    EncryptedNumber, \
    ServerKey, \
    UserKey, \
    FDH, \
    PublicParam, \
    quantize, \
    reverse_quantize


class FlamingoCrypter:
    """
    TODO: Add docstring
    """
    prf: Optional[PRF] = None
    def __init__(self) -> None:
        super().__init__()
        self.my_node_id = None
        self.pairwise_secrets = {}
        self.pairwise_seeds = {}
        self.vector_dtype = 'uint32'

    @staticmethod
    def _init_prf(num_params: int) -> None:
        """
        TODO: Add docstring
        """
        FlamingoCrypter.prf = PRF(vectorsize=num_params, elementsize=16)


    def setup_pairwise_secrets(self, my_node_id:int, nodes_ids: List[str]) -> None:
        """
        TODO: Add docstring
        """
        self.my_node_id = str(my_node_id)
        # this is just an hardcoded example
        for node in nodes_ids:
            if node != my_node_id:
                #create 32 bytes secret of zeros
                self.pairwise_secrets[node] = b'\x02' * 32


    def encrypt(
            self,
            current_round: int,
            params: List[float],
            clipping_range: Union[int, None] = None,
            weight: int = None,
    ):
        """
        TODO: Add docstring
        """
        start = time.process_time()
        params = self._apply_weighing(params, weight)
        params = quantize(weights=params,
                          clipping_range=clipping_range)
        params = np.array(params, dtype=self.vector_dtype)
        vec = np.zeros(len(params), dtype=self.vector_dtype)
        for node_id, secret in self.pairwise_secrets.items():
            # generate seed for pairwise encryption
            pairwise_seed = FlamingoCrypter.prf.eval_key(key=secret, round=current_round)
            # expand seed to a random vector
            pairwise_vector = FlamingoCrypter.prf.eval_vector(seed=pairwise_seed)
            pairwise_vector = np.frombuffer(pairwise_vector, dtype=self.vector_dtype)
            if node_id < self.my_node_id:
                vec += pairwise_vector
            else:
                vec -= pairwise_vector

        time_elapsed = time.process_time() - start
        logger.debug(f"Encryption of the parameters took {time_elapsed} seconds.")
        encrypted_params = vec + params
        return encrypted_params

    def aggregate(
            self,
            num_nodes: int,
            params: Dict[str, np.ndarray],
            total_sample_size: int,
            clipping_range: Union[int, None] = None
    ) -> List[float]:
        """
        TODO: Add docstring
        """
        start = time.process_time()
        params = [p for _, p in params.items()]
        sum_of_weights = np.zeros(len(params[0]), dtype=self.vector_dtype)
        for param in params:
            param = np.array(param, dtype=self.vector_dtype)
            sum_of_weights += param
        # TODO implement weighted averaging here or in `self._jls.aggregate`
        sum_of_weights = sum_of_weights.tolist()
        # Reverse quantize and division (averaging)
        aggregated_params: List[float] = reverse_quantize(
            self._apply_average(sum_of_weights, num_nodes, total_sample_size),
            clipping_range=clipping_range
        )

        time_elapsed = time.process_time() - start
        logger.debug(f"Aggregation is completed in {round(time_elapsed, ndigits=2)} seconds.")

        return aggregated_params

    @staticmethod
    def _apply_average(
            params: List[int],
            num_nodes: int,
            total_sample_size: int
    ) -> List:
        """Takes the average of summed quantized parameters.

        Args:
            params: List of aggregated/summed parameters
            num_nodes: Number of nodes participated in the training
            total_sample_size: Num of total samples used for federated training

        Returns:
            Averaged parameters
        """

        return [param / num_nodes for param in params]

    @staticmethod
    def _apply_weighing(
            params: List[float],
            weight: int,
    ) -> List[float]:
        """Takes the average of summed parameters.

        Args:
            params: A list containing list of parameters
            weight: The weight factor to apply

        Returns:
            Weighed parameters
        """

        # TODO: Currently weighing is not activated due to CLIPPING_RANGE problem.
        #  Implement weighing.
        return [param * 1 for param in params]

    @staticmethod
    def _convert_to_encrypted_number(
            params: List[List[int]],
            public_param: PublicParam
    ) -> List[List[EncryptedNumber]]:
        """Converts encrypted integers to `EncryptedNumber`

        Args:
            params: A list containing list of encrypted integers for each node
            public_param: Public parameter used while encrypting the model parameters
        Returns:
            list of `EncryptedNumber` objects
        """

        encrypted_number = []
        for parameters in params:
            encrypted_number.append([EncryptedNumber(public_param, mpz(param)) for param in parameters])

        return encrypted_number
