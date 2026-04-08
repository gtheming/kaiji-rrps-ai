from __future__ import annotations
import random
import numpy
from environment.move import Move
from abc import ABC, abstractmethod

'''
    Abstract representation of a player in a tournament
    
    Each player has a fixed move budget and number of stars
'''
class Player(ABC):
    rng = numpy.random.default_rng()
    #===================== Universal =====================
    '''
    Instantiation
    Args:
        player_id   (int) the id of each player
        stars       (int) the number of stars the player starts with
        budget      (int) the number of each move the players start with
    '''
    def __init__(self, player_id: int, stars: int = 3, budget: int = 4, position: tuple[int, int] = (0, 0)):
        self.id       = player_id
        self.stars    = stars
        self.budget   = {Move.ROCK: budget, Move.PAPER: budget, Move.SCISSORS: budget}
        self.position = position

    '''
    A list containing the moves available

    Return:
        list[Move]
    '''
    def available_moves(self) -> list[Move]:
        return [m for m, n in self.budget.items() if n > 0]

    '''
    Determines if the player is still alive

    Return:
        bool
    '''
    def is_alive(self) -> bool:
        return self.stars > 0 and len(self.available_moves()) > 0

    '''
    Manages the move budget after an action
    Args:
        move (Move) the action taken by player
    '''
    def use_move(self, move: Move):
        self.budget[move] -= 1

    '''
    Steals the other player's star
    Args:
        other   (Player) opponent star is being stolen from
    '''
    def steal_life(self, other: "Player"):
       self.stars += 1
       other.lose_life()

    '''
    Deducts a star
    '''
    def lose_life(self):
        self.stars -= 1

    #===================== Abstract Methods =====================

    '''
    '''
    @abstractmethod
    def select_move(self, op: "Player" | None = None) -> Move: ...

    
    @abstractmethod
    def select_opponent(self, ops: list["Player"]) -> tuple[bool, list["Player"], "Player" | None]: ...
        
    
    @abstractmethod
    def accept_opponent(self, opponent: "Player") -> bool: ...

'''
Player with completely randomized behavior

* Opponent selection:   selects a random opponent from list of viable opponents

* Move selection:       Selects a random move from remaining moveset

* Accept opponent:      80% chance to accept a given opponent
'''
class RandomPlayer(Player):
    def select_move(self, op: Player) -> Move:
        '''
        Selects a move from the set of available moves

        Return
            Move
        '''
        return random.choice(self.available_moves())

    def select_opponent(self, ops: list[Player]) -> tuple[bool, list, Player | None]:
        '''
        Selects a random opponent from the provided list to 'battle'
        If accepted, returns if accepted and list without them, and opponent selected
        Cannot select self as opponent
        Args:
            ops (list[Players]) list of opponents

        Return
            tuple[bool, list[Player], Player]
        '''
        candidates = [p for p in ops if p is not self]
        if not candidates:
            return False, ops, None
        op = random.choice(candidates)
        if op.accept_opponent(self):
            remaining = [p for p in ops if p is not op and p is not self]
            return True, remaining, op
        return False, ops, None

    def accept_opponent(self, challenger: Player) -> bool:
        '''
        Decides wether to accept a challenge with probability 0.8 yes

        Return
            bool
        '''
        return bool(self.rng.choice([True, False], p=[0.8, 0.2]))
    

class AgentPlayer(Player):
    def select_move(self, op: Player | None) -> Move:
        # Agent moves are driven externally via env.step(action)
        # This fallback should never be called during normal training
        raise NotImplementedError("AgentPlayer moves are controlled by the environment")
    
    def select_opponent(self, ops: list[Player]) -> tuple[bool, list, Player | None]:
        # Agent challenges are also handled externally in step()
        raise NotImplementedError("AgentPlayer challenges are controlled by the environment")

    def accept_opponent(self, challenger: Player) -> bool:
        # Agent always accepts — the environment handles this interaction
        return True