"""华容道棋子类"""

import pygame

from src.constants import BLACK, BOARD_HEIGHT, BOARD_WIDTH, CELL_SIZE, WHITE


class Piece:
    """表示华容道中的一个棋子"""

    def __init__(self, x, y, width, height, color, name):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name
        self.id = 0  # 添加ID属性
        self.rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, width * CELL_SIZE, height * CELL_SIZE)

    def can_move(self, dx, dy, board):
        """检查棋子是否可以移动"""
        new_x = self.x + dx
        new_y = self.y + dy

        # 检查边界
        if new_x < 0 or new_y < 0 or new_x + self.width > BOARD_WIDTH or new_y + self.height > BOARD_HEIGHT:
            return False

        # 检查与其他棋子的碰撞
        for i in range(new_y, new_y + self.height):
            for j in range(new_x, new_x + self.width):
                if board[i][j] is not None and board[i][j] != self:
                    return False

        return True

    def move(self, dx, dy, board):
        """尝试移动棋子"""
        if self.can_move(dx, dy, board):
            self.x += dx
            self.y += dy
            self.rect.x = self.x * CELL_SIZE
            self.rect.y = self.y * CELL_SIZE
            return True
        return False

    def draw(self, screen):
        """绘制棋子"""
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        # 绘制棋子名称，使用支持中文的字体
        # 尝试使用黑体显示中文
        font = pygame.font.SysFont("simhei", 24)

        text = font.render(self.name, True, WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
