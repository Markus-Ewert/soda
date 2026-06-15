import numpy as np
import pytest

from soda.game import Game
from soda.learner.gradient import Gradient
from soda.mechanism.gsp import GSPAuction
from soda.strategy import Strategy


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


def test_own_gradient_matches_utility_tensor(mechanism):
    assert mechanism.own_gradient
    mechanism.own_gradient = False
    game = Game(mechanism, n=5, m=6)
    game.get_utility()
    strategies = {"1": Strategy("1", game)}
    strategies["1"].initialize("random")

    generic_gradient = Gradient()
    generic_gradient.prepare(game, strategies)
    expected = generic_gradient.compute(game, strategies, "1")
    actual = mechanism.compute_gradient(game, strategies, "1")

    assert np.allclose(actual, expected, atol=5e-7)
