from games.chesss import Chess
from move_mapper import uci_to_index
from stock_fish import stock_fish

c = Chess()
finished = False
game_count = 0
winner_white = 0
winner_black = 0
remis = 0

while game_count < 100:
    while not finished:
        stock_fish.set_fen_position(c.board.fen())
        best_move = stock_fish.get_best_move()
        action = uci_to_index[best_move]
        observation, reward, done = c.step(action)
        finished = done

    c.reset()
    finished = False
    game_count += 1

