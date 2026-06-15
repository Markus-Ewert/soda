import numpy as np
import pytest

from soda.mechanism.gsp import GSPAuction


@pytest.fixture
def mechanism():
    bidder = ["1"] * 4
    o_space = {"1": [0, 1]}
    a_space = {"1": [0, 1]}
    param_prior = {"distribution": "uniform"}
    param_util = {
        "tie_breaking": "lose",
        "click_probs": [1.0, 0.8, 0.6],
        "reserve_price": 0.4,
    }
    return GSPAuction(bidder, o_space, a_space, param_prior, param_util)


def test_bid_below_reserve_is_ineligible(mechanism):
    bids = np.array(
        [
            [0.39, 0.40, 0.50],
            [0.30, 0.30, 0.70],
            [0.20, 0.20, 0.60],
            [0.10, 0.10, 0.40],
        ]
    )

    allocation = mechanism.get_allocation(bids, idx=0)

    assert np.allclose(allocation, [0.0, 1.0, 0.6])


def test_qualifying_winner_pays_at_least_reserve(mechanism):
    bids = np.array(
        [
            [0.50, 0.80],
            [0.30, 0.70],
            [0.20, 0.60],
            [0.10, 0.50],
        ]
    )
    allocation = mechanism.get_allocation(bids, idx=0)

    payment = mechanism.get_payment(bids, allocation, idx=0)

    assert np.allclose(payment, [0.4, 0.7])
