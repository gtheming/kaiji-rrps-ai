from __future__ import annotations
import random
import pandas as pd
from gym_core.player import PlayerID, PlayerTable
from gym_core.cards import Card
from gym_core.challenge_table import ChallengeTable
from gym_core.matchup_table import MatchupTable




# preference gen ============================
# all players passed in through PlayerTable MUST be eligible to play (cards remaining)


def _rank_opponents(table: PlayerTable, pid: PlayerID) -> list[PlayerID, PlayerID, PlayerID]:
   candidates = [oid for oid in table if oid != pid]
   return random.sample(candidates, 3)


def _agent_rank_opponents(table: PlayerTable, pid: PlayerID) -> list[PlayerID, PlayerID, PlayerID]:
   candidates = [oid for oid in table if oid != pid]
   return random.sample(candidates, 3)


def _select_move(pid: PlayerID, table: PlayerTable) -> Card:
   available = [card for card in Card if table[pid][card.value] > 0]
   return random.choice(available)


def _agent_select_move(pid: PlayerID, table: PlayerTable) -> Card:
   available = [card for card in Card if table[pid][card.value] > 0]
   return random.choice(available)


# table making ============================


def build_challenge_table(table: PlayerTable) -> ChallengeTable:
   rows: list[dict] = []


   for pid in table:
       card = _select_move(pid, table) if pid is not 0 else _agent_select_move(pid, table)
       targets = _rank_opponents(pid, table) if pid is not 0 else _agent_rank_opponents(table, pid)
       for target_id in targets:
           rows.append({
               "player_id": pid,
               "card": str(card),
               "target_id": target_id,
           })


   df = pd.DataFrame(rows, columns=["player_id", "card", "target_id"])
   return ChallengeTable.validate(df)

