# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    rejection_random_sample_ears_pytorch,
    rejection_random_sample_pytorch,
)

PLACEHOLDER_TOKEN_ID = -1


def mock_pin_memory(original_func):

    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestEARSRejectionSampling(TestBase):
    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_ears_increases_acceptance_with_high_uncertainty(self):
        """When model is uncertain (uniform target probs), EARS tolerance
        should accept tokens that standard sampling would reject."""
        batch_size = 1
        max_spec_len = 2
        vocab_size = 4

        output_standard = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)
        output_ears = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])

        # Draft probs: token 0 has prob 0.4, token 1 has prob 0.3
        draft_probs = torch.zeros(2, vocab_size)
        draft_probs[0, 0] = 0.4
        draft_probs[1, 1] = 0.3

        # Target probs: high uncertainty (close to uniform)
        # uncertainty = 1 - 0.3 = 0.7
        target_probs = torch.tensor(
            [
                [0.3, 0.25, 0.25, 0.2],
                [0.2, 0.3, 0.25, 0.25],
            ]
        )

        bonus_token_ids = torch.tensor([[2]])
        recovered_token_ids = torch.tensor([3, 3])

        # For token 0: target/draft = 0.3/0.4 = 0.75
        # uniform = 0.8 -> standard rejects (0.75 < 0.8)
        # EARS tolerance = 0.1 * 0.7 = 0.07
        # EARS accepts (0.75 >= 0.8 - 0.07 = 0.73)
        uniform_probs = torch.tensor([0.8, 0.5])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_standard,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
        )

        rejection_random_sample_ears_pytorch(
            output_ears,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            base_tolerance=0.1,
        )

        # Standard should reject token 0 -> use recovered token
        assert output_standard[0, 0].item() == 3

        # EARS should accept token 0 due to tolerance
        assert output_ears[0, 0].item() == 0

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_ears_zero_tolerance_matches_standard(self):
        """With base_tolerance=0, EARS should produce identical results
        to standard rejection sampling."""
        batch_size = 1
        max_spec_len = 2
        vocab_size = 4

        output_standard = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)
        output_ears = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = torch.zeros(2, vocab_size)
        draft_probs[0, 0] = 0.5
        draft_probs[1, 1] = 0.4

        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.5, 0.2, 0.2],
            ]
        )

        bonus_token_ids = torch.tensor([[2]])
        recovered_token_ids = torch.tensor([3, 3])
        uniform_probs = torch.tensor([0.5, 0.5])
        is_greedy = torch.tensor([False])

        rejection_random_sample_pytorch(
            output_standard,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
        )

        rejection_random_sample_ears_pytorch(
            output_ears,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            base_tolerance=0.0,
        )

        assert torch.equal(output_standard, output_ears)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_ears_no_effect_on_high_confidence(self):
        """When model is very confident (low uncertainty), EARS tolerance
        should have minimal effect and still reject bad tokens."""
        batch_size = 1
        max_spec_len = 1
        vocab_size = 4

        output_ears = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([0])
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, 0] = 0.5

        # Very confident: max prob is 0.95, uncertainty = 0.05
        target_probs = torch.tensor([[0.05, 0.95, 0.0, 0.0]])

        bonus_token_ids = torch.tensor([[2]])
        recovered_token_ids = torch.tensor([3])
        # ratio = 0.05/0.5 = 0.1, uniform = 0.8
        # tolerance = 0.1 * 0.05 = 0.005
        # 0.1 >= 0.8 - 0.005 = 0.795? No -> still rejected
        uniform_probs = torch.tensor([0.8])
        is_greedy = torch.tensor([False])

        rejection_random_sample_ears_pytorch(
            output_ears,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            base_tolerance=0.1,
        )

        # Should still reject because uncertainty is too low
        assert output_ears[0, 0].item() == 3
