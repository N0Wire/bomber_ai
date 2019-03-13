from settings import e

#dictionary for rewards
rewards_normal = {
	e.MOVED_LEFT : 1,
	e.MOVED_RIGHT : 1,
	e.MOVED_UP : 1,
	e.MOVED_DOWN : 1,
	e.WAITED : -2,
	e.INTERRUPTED : 0,
	e.INVALID_ACTION : -10,
	e.BOMB_DROPPED : 3,
	e.BOMB_EXPLODED : 0,
	e.CRATE_DESTROYED : 1,
	e.COIN_FOUND : 1,
	e.COIN_COLLECTED : 2,
	e.KILLED_OPPONENT : 10,
	e.KILLED_SELF : -100,
	e.GOT_KILLED : -100,
	e.OPPONENT_ELIMINATED : 0,
	e.SURVIVED_ROUND : 10
}

rewards_clipped = {
	e.MOVED_LEFT : 1,
	e.MOVED_RIGHT : 1,
	e.MOVED_UP : 1,
	e.MOVED_DOWN : 1,
	e.WAITED : -1,
	e.INTERRUPTED : 0,
	e.INVALID_ACTION : -1,
	e.BOMB_DROPPED : 1,
	e.BOMB_EXPLODED : 0,
	e.CRATE_DESTROYED : 1,
	e.COIN_FOUND : 1,
	e.COIN_COLLECTED : 1,
	e.KILLED_OPPONENT : 1,
	e.KILLED_SELF : -1,
	e.GOT_KILLED : -1,
	e.OPPONENT_ELIMINATED : 0,
	e.SURVIVED_ROUND : 1
}
