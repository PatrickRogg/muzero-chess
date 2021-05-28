uci_moves = []
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for n1 in range(1, 9):
    for l1 in letters:
        for n2 in range(1, 9):
            for l2 in letters:
                uci = l1 + str(n1) + l2 + str(n2)
                uci_moves.append(uci)

promotions_white = ['a7a8', 'b7b8', 'c7c8', 'd7d8', 'e7e8', 'f7f8', 'g7g8', 'h7h8',
                    'a7b8', 'b7a8', 'b7c8', 'c7b8', 'c7d8', 'd7c8', 'd7e8', 'e7d8', 'e7f8', 'f7e8', 'f7g8', 'g7f8',
                    'g7h8', 'h7g8']
promotions_black = ['a2a1', 'b2b1', 'c2c1', 'd2d1', 'e2e1', 'f2f1', 'g2g1', 'h2h1',
                    'a2b1', 'b2a1', 'b2c1', 'c2b1', 'c2d1', 'd2c1', 'd2e1', 'e2d1', 'e2f1', 'f2e1', 'f2g1', 'g2f1',
                    'g2h1', 'h2g1']
promotions = promotions_white + promotions_black

for piece in ['r', 'n', 'b', 'q']:
    for promotion in promotions:
        uci_moves.append(promotion + piece)

uci_to_index = {}

for i, uci in enumerate(uci_moves):
    uci_to_index[uci] = i
