from typing import Dict, List

import numpy as np

from soda.mechanism.mechanism import Mechanism


class GSPAuction(Mechanism):
    """Generalized Second-Price (GSP) Keyword Auction

    n bidders compete for n-1 slots ranked by click-through rate (CTR).
    Bidder at rank k (0-indexed, descending bid order) wins slot k with
    allocation click_probs[k] and pays max(reserve_price, bid of rank k+1).

    Parameter Utility (param_util):
        click_probs     List[float]: CTR per slot, length n_bidder - 1 (required)
        reserve_price   float: minimum payment for winners. Defaults to 0.
        tie_breaking    str: "lose" (default) or "random"
    """

    def __init__(
        self,
        bidder: List[str],
        o_space: Dict[str, List],
        a_space: Dict[str, List],
        param_prior: Dict,
        param_util: Dict,
    ):
        super().__init__(bidder, o_space, a_space, param_prior, param_util)
        self.name = "gsp_auction"
        self.check_param()

    def utility(
        self, obs_profile: np.ndarray, bids_profile: np.ndarray, index_bidder: int
    ) -> np.ndarray:
        self.test_input_utility(obs_profile, bids_profile, index_bidder)
        valuation = self.get_valuation(obs_profile, index_bidder)
        allocation = self.get_allocation(bids_profile, index_bidder)
        payment = self.get_payment(bids_profile, allocation, index_bidder)
        return allocation * (valuation - payment)

    def get_allocation(self, bids: np.ndarray, idx: int) -> np.ndarray:
        """Allocation = CTR of the slot won; 0 if no slot is won.

        Args:
            bids: shape (n_bidder, n_bid_combos)
            idx: index of agent

        Returns:
            np.ndarray: shape (n_bid_combos,)
        """
        other_bids = np.delete(bids, idx, axis=0)  # (n_bidder-1, n_bid_combos)
        rank = (other_bids > bids[idx]).sum(axis=0)  # 0-indexed rank of bidder idx
        click_arr = np.array(self.click_probs)
        rank_clipped = np.minimum(rank, self.n_slots - 1)
        return np.where(rank < self.n_slots, click_arr[rank_clipped], 0.0)

    def get_payment(
        self, bids: np.ndarray, allocation: np.ndarray, idx: int
    ) -> np.ndarray:
        """Payment = bid of next-ranked bidder, clipped to reserve_price from below.

        Bidder at rank k pays the k-th highest bid among the other bidders
        (equivalently, the (k+1)-th highest bid overall). Zero if not winning.

        Args:
            bids: shape (n_bidder, n_bid_combos)
            allocation: shape (n_bid_combos,), used to zero out losers
            idx: index of agent

        Returns:
            np.ndarray: shape (n_bid_combos,)
        """
        other_bids = np.delete(bids, idx, axis=0)  # (n_bidder-1, n_bid_combos)
        rank = (other_bids > bids[idx]).sum(axis=0)
        # sort other bids descending; row k = k-th highest among others
        other_bids_sorted = np.sort(other_bids, axis=0)[::-1]
        n_bid_combos = bids.shape[1]
        rank_clipped = np.clip(rank, 0, self.n_bidder - 2)
        payment = other_bids_sorted[rank_clipped, np.arange(n_bid_combos)]
        payment = np.maximum(payment, self.reserve_price)
        return payment * (allocation > 0).astype(float)

    def check_param(self):
        if "tie_breaking" not in self.param_util:
            self.tie_breaking = "lose"
        else:
            self.tie_breaking = self.param_util["tie_breaking"]

        self.n_slots = self.n_bidder - 1

        if "click_probs" not in self.param_util:
            raise ValueError("click_probs must be specified in param_util")
        self.click_probs = list(self.param_util["click_probs"])
        if len(self.click_probs) != self.n_slots:
            raise ValueError(
                f"click_probs must have length n_bidder - 1 = {self.n_slots}, "
                f"got {len(self.click_probs)}"
            )

        if "reserve_price" not in self.param_util:
            self.reserve_price = 0.0
        else:
            self.reserve_price = float(self.param_util["reserve_price"])
