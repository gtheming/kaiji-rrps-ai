from typing import TypedDict
from gym_core.matchup_table import MatchupDict
from gym_core.challenge_table import ChallengeTable

class Info(TypedDict):
    matchup_table: MatchupDict
    challenge_table: ChallengeTable
    
