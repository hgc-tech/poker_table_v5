"""MCCFR blueprint bots with depth-limited search and opponent modeling.

The heavy lifting is in C++ (native/): Linear MCCFR training, the card
abstraction, and the per-decision search. This package builds and loads that
code and connects it to the table.
"""
from .bot import MCCFRBot, OpponentStats, SearchSettings, build_request, mccfr_bots, seat_order
from .config import CfrConfig, default_config
