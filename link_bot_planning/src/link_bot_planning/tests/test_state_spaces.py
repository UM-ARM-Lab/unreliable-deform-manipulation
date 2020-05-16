from unittest import TestCase

import numpy as np
from moonshine.tests.testing_utils import assert_dicts_close_np
import ompl.base as ob

from link_bot_planning.state_spaces import compound_to_numpy, compound_from_numpy


class Test(TestCase):
    def test_simple_compound_state(self):
        description = {
            'a': {
                "idx": 0,
                "n_state": 2,
                "weight": 0,
            }
        }
        expected_numpy_state = {
            'a': np.array([1.0, 2.0])
        }

        state_space = ob.CompoundStateSpace()
        subspace = ob.RealVectorStateSpace(2)
        state_space.addSubspace(subspace, weight=1)
        scope_ompl_compound_state = ob.CompoundState(state_space)
        ompl_compound_state = scope_ompl_compound_state()
        compound_from_numpy(description, expected_numpy_state, ompl_compound_state)
        numpy_state = compound_to_numpy(description, ompl_compound_state)
        assert_dicts_close_np(numpy_state, expected_numpy_state)

    def test_complex_compound_state(self):
        description = {
            'a': {
                "idx": 0,
                "n_state": 2,
                "weight": 0,
            },
            'b': {
                "idx": 1,
                "n_state": 1,
                "weight": 0,
            },
            'c': {
                "idx": 2,
                "n_state": 4,
                "weight": 0,
            },
        }
        expected_numpy_state = {
            'a': np.array([1.0, 2.0]),
            'b': np.array([3.0]),
            'c': np.array([4.0, 5.0, 6.0, 7.0]),
        }

        state_space = ob.CompoundStateSpace()
        a_subspace = ob.RealVectorStateSpace(2)
        b_subspace = ob.RealVectorStateSpace(1)
        c_subspace = ob.RealVectorStateSpace(4)
        state_space.addSubspace(a_subspace, weight=1)
        state_space.addSubspace(b_subspace, weight=1)
        state_space.addSubspace(c_subspace, weight=1)
        scope_ompl_compound_state = ob.CompoundState(state_space)
        ompl_compound_state = scope_ompl_compound_state()
        compound_from_numpy(description, expected_numpy_state, ompl_compound_state)
        numpy_state = compound_to_numpy(description, ompl_compound_state)
        assert_dicts_close_np(numpy_state, expected_numpy_state)
