import math


def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("--" * 5)


def is_winner(board, player):
    for row in board:
        if all([s == player for s in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False


def get_empty_cells(board):
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "_":
                empty_cells.append((i, j))
    return empty_cells


def is_draw(board):
    return len(get_empty_cells(board)) == 0


def minimax(board, depth, is_maximizing, alpha, beta, current_player):
    if is_winner(board, "X"):
        return -1
    if is_winner(board, "O"):
        return 1
    if is_draw(board):
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for (x, y) in get_empty_cells(board):
            board[x][y] = "O"
            eval = minimax(board, depth + 1, False, alpha, beta, "O")
            board[x][y] = "_"
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for (x, y) in get_empty_cells(board):
            board[x][y] = "X"
            eval = minimax(board, depth + 1, True, alpha, beta, "X")
            board[x][y] = "_"
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def best_move(board):
    best_eval = -math.inf
    move = None
    for (x, y) in get_empty_cells(board):
        board[x][y] = "O"
        eval = minimax(board, 0, False, -math.inf, math.inf, "O")
        board[x][y] = "_"
        if eval > best_eval:
            best_eval = eval
            move = (x, y)
    return move


def is_valid_move(board, x, y):
    return board[x][y] == "_"


# Kezdő állapot
board = [["_"] * 3 for _ in range(3)]
current_player = "X"

while True:
    print_board(board)
    print("")
    if current_player == "X":
        while True:
            x, y = map(int, input("Enter your move (row col): ").split())
            if is_valid_move(board, x, y):
                board[x][y] = "X"
                break
            else:
                print("Invalid move. Please try again.")
    else:
        move = best_move(board)
        if move:
            board[move[0]][move[1]] = "O"
            if is_winner(board, current_player):
                print_board(board)  # Nyerő lépés kirajzolása
                print(f"{current_player} wins!")
                break
        else:
            print("Draw!")
            break

    if current_player == "X" and is_winner(board, current_player):
        print(f"{current_player} wins!")
        break
    if is_draw(board):
        print("Draw!")
        break

    current_player = "O" if current_player == "X" else "X"

