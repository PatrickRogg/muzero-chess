from stockfish import Stockfish

stock_fish = Stockfish('stockfish/stockfish_13_linux_x64_bmi2', parameters={"Threads": 2, "Minimum Thinking Time": 500})
stock_fish.set_elo_rating(3000)
