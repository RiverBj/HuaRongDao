"""华容道游戏主类"""

import os
import sys

import pygame

from src.constants import *
from src.piece import Piece


class HuarongdaoGame:
    """华容道游戏类"""

    def __init__(self):
        pygame.font.init()  # 初始化字体模块
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("华容道")
        self.clock = pygame.time.Clock()
        self.pieces = []
        self.selected_piece = None
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.setup_pieces()
        self.update_board()
        print("初始棋盘状态:")
        self.print_board()

    def print_board(self):
        """打印棋盘状态用于调试"""
        for row in self.board:
            print([p.name if p else "." for p in row])
        print()

    def setup_pieces(self):
        """设置初始棋子布局"""
        # 清空棋子列表
        self.pieces = []

        # 曹操 (2x2) - 位于顶部中央
        cao_cao = Piece(1, 0, 2, 2, RED, "曹操")
        self.pieces.append(cao_cao)

        # 左侧上方垂直将 (1x2)
        general1 = Piece(0, 0, 1, 2, BLUE, "将")
        self.pieces.append(general1)

        # 右侧上方垂直将 (1x2)
        general2 = Piece(3, 0, 1, 2, BLUE, "将")
        self.pieces.append(general2)

        # 左侧下方水平将 (2x1)
        general3 = Piece(0, 2, 1, 2, BLUE, "将")
        self.pieces.append(general3)

        # 右侧下方水平将 (2x1)
        general4 = Piece(3, 2, 1, 2, BLUE, "将")
        self.pieces.append(general4)

        # 中间下方垂直将 (1x2)
        vertical_general = Piece(1, 2, 2, 1, GREEN, "将")
        self.pieces.append(vertical_general)

        # 四个小兵
        soldier1 = Piece(1, 3, 1, 1, YELLOW, "兵")
        self.pieces.append(soldier1)

        soldier2 = Piece(2, 3, 1, 1, YELLOW, "兵")
        self.pieces.append(soldier2)

        soldier3 = Piece(0, 4, 1, 1, YELLOW, "兵")
        self.pieces.append(soldier3)

        soldier4 = Piece(3, 4, 1, 1, YELLOW, "兵")
        self.pieces.append(soldier4)

    def update_board(self):
        """更新棋盘状态"""
        # 清空棋盘
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

        # 更新每个棋子的位置
        for piece in self.pieces:
            for i in range(piece.y, piece.y + piece.height):
                for j in range(piece.x, piece.x + piece.width):
                    # 检查是否有重叠
                    if self.board[i][j] is not None:
                        print(f"警告: 棋子在 ({j},{i}) 位置重叠!")
                    self.board[i][j] = piece

    def get_piece_at(self, x, y):
        """获取指定位置的棋子"""
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE

        if 0 <= grid_x < BOARD_WIDTH and 0 <= grid_y < BOARD_HEIGHT:
            return self.board[grid_y][grid_x]
        return None

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    piece = self.get_piece_at(event.pos[0], event.pos[1])
                    if piece:
                        self.selected_piece = piece
                        print(f"选中棋子: {piece.name} 位置: ({piece.x}, {piece.y})")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif self.selected_piece:
                    moved = False
                    if event.key == pygame.K_UP:
                        moved = self.selected_piece.move(0, -1, self.board)
                        if moved:
                            print(f"向上移动: {self.selected_piece.name}")
                    elif event.key == pygame.K_DOWN:
                        moved = self.selected_piece.move(0, 1, self.board)
                        if moved:
                            print(f"向下移动: {self.selected_piece.name}")
                    elif event.key == pygame.K_LEFT:
                        moved = self.selected_piece.move(-1, 0, self.board)
                        if moved:
                            print(f"向左移动: {self.selected_piece.name}")
                    elif event.key == pygame.K_RIGHT:
                        moved = self.selected_piece.move(1, 0, self.board)
                        if moved:
                            print(f"向右移动: {self.selected_piece.name}")

                    if moved:
                        self.update_board()
                        self.print_board()
        return True

    def draw(self):
        """绘制游戏画面"""
        self.screen.fill(WHITE)

        # 绘制网格线
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (SCREEN_WIDTH, y))

        # 绘制棋子
        for piece in self.pieces:
            piece.draw(self.screen)

        # 显示选中提示
        if self.selected_piece:
            pygame.draw.rect(self.screen, (255, 100, 100), self.selected_piece.rect, 3)

        pygame.display.flip()

    def run(self):
        """运行游戏主循环"""
        running = True
        while running:
            running = self.handle_events()

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()
