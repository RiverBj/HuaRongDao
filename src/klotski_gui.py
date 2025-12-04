import pygame


class KlotskiGUI:
    def __init__(self, env):
        self.env = env
        self.cell_size = 90
        self.width = env.cols * self.cell_size
        self.height = env.rows * self.cell_size + 200

        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("华容道 AI 学习")

        # 字体
        self.font = pygame.font.SysFont("SimHei", size=15)
        self.title_font = pygame.font.SysFont("SimHei", 15, bold=True)

    def draw(self, episode=None, step=None, epsilon=None, reward=None, mode="训练"):
        """绘制界面"""
        # 背景
        self.screen.fill((240, 240, 240))

        # 标题
        title = self.title_font.render("华容道 AI 学习系统", True, (0, 0, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))

        # 信息
        info_y = 60
        if episode is not None:
            text = self.font.render(f"轮次: {episode}", True, (0, 100, 200))
            self.screen.blit(text, (20, info_y))

        if step is not None:
            text = self.font.render(f"步数: {step}", True, (0, 100, 200))
            self.screen.blit(text, (150, info_y))

        if epsilon is not None:
            text = self.font.render(f"探索率: {epsilon:.3f}", True, (200, 100, 0))
            self.screen.blit(text, (280, info_y))

        if reward is not None:
            color = (0, 150, 0) if reward > 0 else (200, 0, 0) if reward < 0 else (100, 100, 100)
            text = self.font.render(f"奖励: {reward:+.2f}", True, color)
            self.screen.blit(text, (450, info_y))

        # 模式
        text = self.font.render(f"模式: {mode}", True, (150, 0, 150))
        self.screen.blit(text, (self.width - 150, info_y))

        # 绘制棋盘
        board_y = 100
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                val = self.env.board[r, c]
                x = c * self.cell_size
                y = board_y + r * self.cell_size

                # 棋子颜色
                if val == 0:
                    color = (220, 220, 220)  # 空格
                else:
                    color = self.env.pieces[val]["color"]

                # 绘制棋子
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 2)

                # 棋子名称
                if val != 0:
                    name = self.env.pieces[val]["name"]
                    text = self.font.render(name, True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        # 目标区域（红色边框）
        target_rect = pygame.Rect(
            1 * self.cell_size, board_y + 3 * self.cell_size, 2 * self.cell_size, 2 * self.cell_size
        )
        pygame.draw.rect(self.screen, (255, 0, 0), target_rect, 4)

        # 说明
        instructions = ["目标：将曹操（红色）移动到红色方框", "按ESC退出，空格键暂停"]
        for i, text in enumerate(instructions):
            inst = self.font.render(text, True, (50, 50, 50))
            self.screen.blit(inst, (20, self.height - 50 + i * 25))

        pygame.display.flip()

    def close(self):
        pygame.quit()
