import time

import pygame
import torch

from .agent import DQNAgent
from .klotski_env import KlotskiEnv
from .klotski_gui import KlotskiGUI


def train_model(episodes=2000, render_freq=50):
    """训练AI模型"""
    print("开始训练华容道AI...")
    print("=" * 60)

    # 初始化
    env = KlotskiEnv()
    state_size = env.rows * env.cols
    max_actions = len(env.pieces) * 4  # 每个棋子4个方向

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    agent = DQNAgent(state_size, max_actions, device)
    gui = KlotskiGUI(env)

    # 训练统计
    stats = {"rewards": [], "steps": [], "solved": [], "losses": []}

    print(f"{'轮次':>6} | {'步数':>5} | {'奖励':>8} | {'探索率':>7} | {'状态':>8}")
    print("-" * 60)

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < 300:  # 最多300步
            step += 1

            # 获取合法动作
            legal_actions = env.get_legal_moves()
            if not legal_actions:
                break

            # 选择动作
            action = agent.select_action(state, legal_actions)
            if action is None:
                break

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 学习
            loss = agent.learn()
            if loss > 0:
                stats["losses"].append(loss)

            state = next_state

            # 渲染（每隔一定轮次）
            if render_freq > 0 and episode % render_freq == 0 and step % 3 == 0:
                gui.draw(episode, step, agent.epsilon, total_reward, "训练中")

                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gui.close()
                        return agent, stats
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            gui.close()
                            return agent, stats
                        elif event.key == pygame.K_SPACE:
                            # 暂停
                            paused = True
                            while paused:
                                for e in pygame.event.get():
                                    if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                        paused = False
                                pygame.time.delay(100)

                pygame.time.delay(30)

        # 记录统计
        stats["rewards"].append(total_reward)
        stats["steps"].append(step)
        stats["solved"].append(1 if done else 0)

        # 显示进度
        status = "成功！" if done else "继续..."
        if episode % 100 == 0 or done:
            print(f"{episode:6d} | {step:5d} | {total_reward:8.2f} | {agent.epsilon:7.3f} | {status:>8}")

        # 保存最佳模型
        if done and total_reward > 30:
            torch.save(agent.policy_net.state_dict(), "klotski_best_model.pth")
            print(f"💾 保存最佳模型 (奖励: {total_reward:.1f})")

    # 保存最终模型
    torch.save(
        {"policy_net": agent.policy_net.state_dict(), "epsilon": agent.epsilon, "stats": stats},
        "klotski_final_model.pth",
    )

    print("=" * 60)
    print("训练完成！模型已保存")

    gui.close()
    return agent, stats


# ==================== 5. 演示函数 ====================
def demonstrate(model_path="klotski_best_model.pth"):
    """演示训练好的AI"""
    print("\n开始演示AI解决方案...")

    # 初始化
    env = KlotskiEnv()
    state_size = env.rows * env.cols
    max_actions = len(env.pieces) * 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_size, max_actions, device)

    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
            agent.policy_net.load_state_dict(checkpoint["policy_net"])
        else:
            agent.policy_net.load_state_dict(checkpoint)

        agent.epsilon = 0.01  # 演示时使用低探索率
        print("✅ 模型加载成功")
    except:
        print(f"❌ 无法加载模型 {model_path}")
        print("请先运行训练模式")
        return

    gui = KlotskiGUI(env)

    # 演示
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    print("按ESC退出，空格键暂停")
    print("-" * 40)

    moves_history = []

    while not done and step < 150:
        step += 1

        # 获取合法动作
        legal_actions = env.get_legal_moves()
        if not legal_actions:
            print("无合法动作")
            break

        # AI选择动作（无探索）
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(state, legal_actions)
        agent.epsilon = original_epsilon

        if action is None:
            break

        # 解码动作信息
        piece_id = action // 4
        dir_idx = action % 4
        piece_name = env.pieces[piece_id]["name"]
        dir_name = env.direction_names[dir_idx]

        # 执行动作
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        # 记录移动
        moves_history.append(f"{step:2d}. {piece_name:>2} → {dir_name}")

        # 显示移动
        print(f"第{step:2d}步: 移动 {piece_name} 向{dir_name} (奖励: {reward:+.2f})")

        # 渲染
        gui.draw(None, step, 0.0, total_reward, "AI演示")

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    gui.close()
                    return
                elif event.key == pygame.K_SPACE:
                    print("暂停...按空格键继续")
                    paused = True
                    while paused:
                        for e in pygame.event.get():
                            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                paused = False
                        pygame.time.delay(100)

        pygame.time.delay(600)  # 慢速演示

    # 显示结果
    print("-" * 40)
    if done:
        print("\n🎉 AI成功解出华容道！")
        print(f"   总步数: {step}")
        print(f"   总奖励: {total_reward:.2f}")

        # 显示移动序列
        print("\n移动序列:")
        for i in range(0, len(moves_history), 5):
            print("  " + " | ".join(moves_history[i : i + 5]))
    else:
        print("\n⚠️ AI未能在150步内解出")

    gui.close()
    time.sleep(2)


# ==================== 6. 手动游戏 ====================
def play_manually():
    """手动玩游戏模式 - 修复按键控制版本"""
    env = KlotskiEnv()  # 你的华容道环境
    visualizer = KlotskiGUI(env)  # 或 KlotskiGUI
    clock = pygame.time.Clock()

    # 游戏状态变量
    selected_piece_id = None  # 当前选中的棋子ID
    selected_pos = [0, 0]  # 选中的位置（用于视觉反馈）
    steps = 0
    running = True

    print("手动游戏模式已启动")
    print("控制方式:")
    print("  方向键(↑↓←→): 移动选择框")
    print("  Enter/空格: 选择/取消选择棋子")
    print("  WASD: 移动已选中的棋子")
    print("  R: 重新开始游戏")
    print("  ESC: 退出游戏")

    while running:
        # 处理所有事件
        r, c = selected_pos[0], selected_pos[1]
        x = c * visualizer.cell_size
        y = 100 + r * visualizer.cell_size
        highlight = pygame.Rect(x + 2, y + 2, visualizer.cell_size - 4, visualizer.cell_size - 4)
        pygame.draw.rect(visualizer.screen, (255, 255, 100), highlight, 3)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # ESC键退出
                if event.key == pygame.K_ESCAPE:
                    running = False

                # R键重置游戏
                elif event.key == pygame.K_r:
                    env.reset()
                    steps = 0
                    selected_piece_id = None
                    selected_pos = [0, 0]
                    print("游戏已重置")

                # 方向键移动选择框
                elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    # 初始化选择位置（如果还没有选择的话）
                    if selected_pos is None:
                        selected_pos = [0, 0]  # [行, 列]

                    # 根据按键更新选择位置
                    if event.key == pygame.K_UP and selected_pos[0] > 0:
                        selected_pos[0] -= 1
                    elif event.key == pygame.K_DOWN and selected_pos[0] < env.rows - 1:
                        selected_pos[0] += 1
                    elif event.key == pygame.K_LEFT and selected_pos[1] > 0:
                        selected_pos[1] -= 1
                    elif event.key == pygame.K_RIGHT and selected_pos[1] < env.cols - 1:
                        selected_pos[1] += 1

                    # 更新当前选中的棋子ID
                    r, c = selected_pos
                    selected_piece_id = env.board[r][c] if env.board[r][c] != 0 else None

                # Enter或空格键确认/取消选择
                elif event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                    if selected_pos is not None:
                        r, c = selected_pos
                        current_id = env.board[r][c] if env.board[r][c] != 0 else None

                        if selected_piece_id is None and current_id is not None:
                            # 选择棋子
                            selected_piece_id = current_id
                            piece_name = env.pieces[selected_piece_id]["name"]
                            print(f"已选择棋子: {piece_name}")
                        elif selected_piece_id == current_id:
                            # 取消选择（再次选择同一棋子）
                            selected_piece_id = None
                            print("已取消选择棋子")
                        else:
                            # 选择另一个棋子
                            selected_piece_id = current_id
                            piece_name = env.pieces[selected_piece_id]["name"] if selected_piece_id else "无"
                            print(f"已选择棋子: {piece_name}")

                # WASD移动已选中的棋子
                elif selected_piece_id is not None and event.key in [
                    pygame.K_w,
                    pygame.K_s,
                    pygame.K_a,
                    pygame.K_d,
                ]:
                    # 将按键映射为移动方向
                    direction_map = {
                        pygame.K_w: (-1, 0),  # 上
                        pygame.K_s: (1, 0),  # 下
                        pygame.K_a: (0, -1),  # 左
                        pygame.K_d: (0, 1),  # 右
                    }

                    direction = direction_map[event.key]
                    piece_name = env.pieces[selected_piece_id]["name"]
                    dir_names = {(-1, 0): "上", (1, 0): "下", (0, -1): "左", (0, 1): "右"}
                    dir_name = dir_names[direction]

                    # 尝试移动棋子
                    if env.can_move(selected_piece_id, direction):
                        env.move_piece(selected_piece_id, direction)
                        steps += 1
                        print(f"第{steps}步: 移动 {piece_name} 向{dir_name}")

                        # 检查是否获胜
                        if env.is_solved():
                            print(f"🎉 恭喜！你用了 {steps} 步完成了华容道！")
                            # 这里可以添加胜利的视觉反馈

                    else:
                        print(f"无法移动 {piece_name} 向{dir_name} (位置被阻挡或超出边界)")

        # 更新显示（包括选择框高亮）
        visualizer.draw(step=steps, mode="手动模式")

        # 如果有选中的位置，绘制选择框
        if selected_pos is not None:
            r, c = selected_pos
            # 这里需要调用visualizer的绘制选择框方法
            # 或者修改draw方法以接收selected_pos参数
            # 例如：visualizer.highlight_cell(r, c)

        # 控制帧率
        clock.tick(30)

    print("游戏结束")
    visualizer.close()
