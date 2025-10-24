# individual.py

import pygame
import setting
import numpy as np
from .entity import Entity
from neural_input_config import get_sensor_inputs, get_input_size, draw_sensor_inputs
        
class Individual(Entity):
    """
    种群个体类，继承自实体类，表示游戏中的一个遗传算法个体，使用神经网络进行AI决策，支持平面移动。

    参数:
        size (int): 种群个体的尺寸（正方形）。
        pos_x (float): 初始水平位置。
        pos_y (float): 初始垂直位置。
        individual_id (int): 种群个体编号，默认为0。

        neural_network (np.ndarray): 神经网络权重矩阵，默认为随机初始化。
        input_size (int): 神经网络输入层大小，默认为14（从配置文件获取）。
        hidden1_size (int): 第一隐藏层大小，默认为128。
        hidden2_size (int): 第二隐藏层大小，默认为64。
        hidden3_size (int): 第三隐藏层大小，默认为32。
        hidden4_size (int): 第四隐藏层大小，默认为16。
        output_size (int): 输出层大小，默认为2。
    """

    def __init__(self, radius=None, width=None, height=None, pos_x=None, pos_y=None, individual_id=0,
                 neural_network=None, input_size=None, 
                 hidden1_size=setting.NEURAL_NETWORK_HIDDEN1_SIZE, 
                 hidden2_size=setting.NEURAL_NETWORK_HIDDEN2_SIZE,
                 hidden3_size=setting.NEURAL_NETWORK_HIDDEN3_SIZE, 
                 hidden4_size=setting.NEURAL_NETWORK_HIDDEN4_SIZE,
                 output_size=setting.NEURAL_NETWORK_OUTPUT_SIZE, 
                 frame_counter_ref=None):
        
        # 使用配置文件中的输入大小
        input_size = get_input_size()
        if input_size is None:
            raise ValueError("个体神经网络输入层大小必须从配置文件中正确获取")
        
        # 调用父类构造函数，设置基础属性
        # shape参数可选值: 'circle'(圆形), 'rectangle'(矩形)
        super().__init__(radius=radius, width=width, height=height, pos_x=pos_x, pos_y=pos_y, 
                        entity_id=individual_id, 
                        shape='rectangle', color=setting.BLACK)
        
        # 种群个体特有属性
        self.fitness = 0
        self.goals_reached = 0  # 达到的目标数量
        self.reached_goal_ids = []  # 已触碰的目标编号列表，用于多目标训练
        self.goal_reach_times = []  # 记录到达每个目标点的时间（从游戏开始到到达目标的时间）

        # 初始化适应度分量
        self.distance_fitness = 0.0
        self.time_efficiency_fitness = 0.0
        self.survival_time_fitness = 0.0  # 存活时间适应度
        self.flight_count_fitness = 0.0   # 飞行次数适应度
        # 预计算屏幕对角线长度的平方（用于平方距离计算）
        self.max_screen_distance_squared = setting.SCREEN_WIDTH**2 + setting.SCREEN_HEIGHT**2
        
        # 帧计数器引用
        self.frame_counter_ref = frame_counter_ref  # 引用主游戏中的帧计数器
        self.last_reach_frame = 0  # 记录最后到达目标的时间，用于计算时间差
        
        # 神经网络配置参数
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.hidden4_size = hidden4_size
        self.output_size = output_size
        
        # 手动控制属性
        self.manual_left = False
        self.manual_right = False
        self.manual_up = False
        self.manual_down = False
        
        # 轨迹相关属性
        self.trajectory_points = []  # 记录个体走过的点
        self.max_trajectory_points = 200  # 最大记录点数，避免内存占用过大
        
        # 飞行次数相关属性
        self.flight_count = 0  # 向上飞行按键次数
        self.last_move_up_state = False  # 上一次的向上移动状态，用于检测按键变化
        
        # 飞行能量相关属性
        self.max_energy = setting.INDIVIDUAL_MAX_FLIGHT_ENERGY  # 最大能量值
        self.current_energy = self.max_energy  # 当前能量值
        self.is_flying = False  # 是否正在飞行
        self.energy_recovery_timer = 0  # 能量恢复计时器（帧数）
        self.is_on_ground_or_obstacle = False  # 是否在地面或障碍物上表面
        
        # 计算神经网络权重总数
        total_weights = (input_size * hidden1_size + hidden1_size + 
                        hidden1_size * hidden2_size + hidden2_size + 
                        hidden2_size * hidden3_size + hidden3_size + 
                        hidden3_size * hidden4_size + hidden4_size + 
                        hidden4_size * output_size + output_size)
        
        # 神经网络决策系统
        # 默认结构：输入层 → 128节点隐藏层 → 64节点隐藏层 → 32节点隐藏层 → 16节点隐藏层 → 2输出层
        if neural_network is None:
            # 手动控制模式下允许神经网络权重为空
            self.neural_network = None
            # raise ValueError("个体神经网络权重矩阵不能为空")
        else:
            self.neural_network = neural_network
            # 验证权重数组大小是否匹配
            if len(neural_network) != total_weights:
                raise ValueError(f"权重数组大小不匹配，期望{total_weights}，实际{len(neural_network)}")

    # 功能: 获取传感器输入值
    def get_sensor_inputs(self, individual, goals, obstacles):
        """
        获取传感器输入值 - 使用neural_input_config中的统一函数
        
        参数:
            individual: 种群个体对象
            goals: 目标列表
            obstacles: 障碍物列表
        返回:
            np.array: 归一化的输入值数组
        """
        # 使用统一传感器输入函数
        inputs = get_sensor_inputs(individual, goals, obstacles)
        
        # 确保输入大小正确
        if len(inputs) != self.input_size:
            raise ValueError(f"传感器输入大小不匹配，期望{self.input_size}，实际{len(inputs)}")
        
        return np.array(inputs)

    # 功能: 检查与目标的碰撞
    def check_goal_collision(self, goals):
        """
        检查与目标的碰撞，只检查编号最小的未触碰目标
        
        参数:
            goals: 目标列表
            
        返回:
            Goal or None: 碰撞的目标对象，如果没有碰撞返回None
        """

        # 检查是否已死亡，若已死亡则不检查碰撞
        if not self.active:
            return None
            
        # 查找编号最小的未触碰目标
        min_unreached_goal = None
        for goal in goals:
            # 跳过已触碰的目标（通过目标编号判断）
            if goal.entity_id in self.reached_goal_ids:
                continue
                
            # 查找编号最小的未触碰目标
            if min_unreached_goal is None or goal.entity_id < min_unreached_goal.entity_id:
                min_unreached_goal = goal
                
        # 如果存在未触碰的目标，检查是否与之发生碰撞
        if min_unreached_goal:
            if self.check_collision(min_unreached_goal):
                return min_unreached_goal
        
        return None

    # 功能: 处理目标碰撞事件
    def on_goal_collision(self, goal):
        """
        当个体碰撞到目标时的处理逻辑，包括记录目标ID、计算适应度等
        
        参数:
            goal: 碰撞到的目标对象
        """
        # 检查目标是否已经被触碰过，避免重复计算
        if goal.entity_id in self.reached_goal_ids:
            return
            
        # 增加已到达目标数
        self.goals_reached += 1

        # 记录已到达目标的ID
        self.reached_goal_ids.append(goal.entity_id)

        # 记录到达目标的时间（帧数）
        if self.frame_counter_ref is not None:
            current_frame = self.frame_counter_ref()
            
            # 计算到达当前目标的时间差
            if not self.goal_reach_times:
                # 如果是第一个目标，记录从开始到第一个目标的时间
                reach_time_diff = current_frame
            else:
                # 如果是后续目标，记录从上个目标到达后到当前目标的时间差
                reach_time_diff = current_frame - self.last_reach_frame
            
            # 记录时间差
            self.goal_reach_times.append(reach_time_diff)
            # 更新最后到达时间
            self.last_reach_frame = current_frame
        else:
            # 如果没有帧计数器引用
            raise ValueError("没有帧计数器引用，无法记录到达帧数")

    def calculate_fitness(self, goals, obstacles):
        """
        计算个体适应度，包括距离适应度、目标到达适应度、障碍物避免适应度等
        
        参数:
            goals (list): 目标点列表  
            obstacles (list): 障碍物列表
        返回:
            float: 个体适应度值
        """
        # 计算距离适应度
        self.calculate_distance_fitness(goals)
        
        # 计算时间效率适应度
        self.calculate_time_efficiency_fitness()
        
        # 计算存活时间适应度
        self.calculate_survival_time_fitness()
        
        # # 计算飞行次数适应度
        # self.calculate_flight_count_fitness()
        
        # # 计算目标到达适应度
        # self.calculate_reach_fitness(goals)
        
        # # 计算障碍物避免适应度
        # self.calculate_avoidance_fitness(obstacles)

        # 总适应度 = 距离奖励适应度 * 15 + 时间奖励适应度 * 30 + 存活时间适应度 * 10 + 飞行次数适应度 * 1
        # 只有当个体到达至少一个目标时，才计算存活时间和飞行次数适应度
        if self.goals_reached > 0:
            self.fitness = self.distance_fitness * 20 + self.time_efficiency_fitness * 50 + self.survival_time_fitness * 10 + self.flight_count_fitness * 1
        else:
            # 如果个体没有到达任何目标，只计算距离适应度
            self.fitness = self.distance_fitness * 20
        return self.fitness

    # 功能: 计算个体距离适应度：用于促进个体靠近并接触目标点
    def calculate_distance_fitness(self, goals):
        """
        计算个体距离适应度，包括距离适应度
        
        参数:
            goals (list): 目标点列表  
        返回:
            float: 距离适应度值
        """
        
        # 距离适应度计算
        # 查找编号最小的未触碰目标
        min_unreached_goal = None
        for goal in goals:
            # 跳过已触碰的目标（通过目标编号判断）
            if goal.entity_id in self.reached_goal_ids:
                continue
                
            # 查找编号最小的未触碰目标
            if min_unreached_goal is None or goal.entity_id < min_unreached_goal.entity_id:
                min_unreached_goal = goal

        # 计算到最近未触碰目标点的距离
        if min_unreached_goal:
            dx = min_unreached_goal.pos_x - self.pos_x
            dy = min_unreached_goal.pos_y - self.pos_y
            # 使用平方距离代替精确欧几里得距离，提高计算效率
            squared_distance = dx*dx + dy*dy
            
            # 距离越近，得分越高（使用平方距离进行比较）
            # 只使用当前帧的距离计算适应度，不记录历史最佳值
            self.distance_fitness = max(0, 1 - squared_distance/self.max_screen_distance_squared)
        else:
            # 如果所有目标都已到达，给予满分奖励
            self.distance_fitness = 1.0

    # 功能: 计算时间效率适应度
    def calculate_time_efficiency_fitness(self):
        """
        计算时间效率适应度，基于个体到达目标的时间效率
        每个差值时间/最大时间累加
        
        返回:
            float: 时间效率适应度值
        """
        # 如果个体已经到达了目标，给予时间效率奖励
        if self.goal_reach_times:
            # 最大合理时间，时间越短得分越高
            max_reasonable_time = setting.ROUND_DURATION_FRAMES
            
            # 计算每个差值时间/最大时间的累加值
            time_efficiency_sum = 0.0
            for reach_time in self.goal_reach_times:
                # 每个差值时间/最大时间，时间越短得分越高
                time_efficiency = max(0, 1 - reach_time / max_reasonable_time)
                time_efficiency_sum += time_efficiency
            
            # 时间效率适应度：每个差值时间/最大时间累加
            self.time_efficiency_fitness = time_efficiency_sum
        else:
            # 如果没有到达目标，时间效率适应度为0
            self.time_efficiency_fitness = 0.0

    # 功能: 计算存活时间适应度
    def calculate_survival_time_fitness(self):
        """
        计算存活时间适应度，基于个体存活的时间长度
        存活得越久，得分越高
        
        返回:
            float: 存活时间适应度值
        """
        # 检查是否有帧计数器引用
        if self.frame_counter_ref is not None:
            # 获取当前帧数
            current_frame = self.frame_counter_ref()
            
            # 最大合理时间
            max_reasonable_time = setting.ROUND_DURATION_FRAMES
            
            # 计算存活时间适应度：每帧增长不得低于0.2，适应度增长量为max(0.2，current_frame / max_reasonable_time)
            # 存活得越久，得分越高（线性增长，无增长极限）
            survival_fitness = max(0.2, current_frame / max_reasonable_time)
            
            # 更新存活时间适应度
            self.survival_time_fitness = survival_fitness
        else:
            # 如果没有帧计数器引用，存活时间适应度为0
            self.survival_time_fitness = 0.0

    # 功能: 计算飞行次数适应度
    def calculate_flight_count_fitness(self):
        """
        计算飞行次数适应度，使用向上飞行按键的次数越少得分越高
        奖励为100/飞行次数，飞行次数为0时给予最大奖励100
        
        返回:
            float: 飞行次数适应度值
        """
        if self.flight_count == 0:
            # 如果完全没有使用飞行，给予最大奖励
            self.flight_count_fitness = 1.0
        else:
            # 使用1/飞行次数作为奖励，飞行次数越少得分越高
            self.flight_count_fitness = 1.0 / self.flight_count

    # 功能: 获取神经网络详细信息
    def get_neural_network_info(self):
        """
        获取神经网络详细信息
        
        返回:
            dict: 包含网络结构、权重统计等信息的字典
        """
        if self.neural_network is None:
            return {
                'input_size': self.input_size,
                'hidden1_size': self.hidden1_size,
                'hidden2_size': self.hidden2_size,
                'hidden3_size': self.hidden3_size,
                'hidden4_size': self.hidden4_size,
                'output_size': self.output_size,
                'total_weights': 0,
                'weight_stats': {
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0
                },
                'architecture': f"{self.input_size}→{self.hidden1_size}→{self.hidden2_size}→{self.hidden3_size}→{self.hidden4_size}→{self.output_size}",
                'status': '未初始化'
            }
            
        return {
            'input_size': self.input_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'hidden3_size': self.hidden3_size,
            'hidden4_size': self.hidden4_size,
            'output_size': self.output_size,
            'total_weights': len(self.neural_network),
            'weight_stats': {
                'mean': float(np.mean(self.neural_network)),
                'std': float(np.std(self.neural_network)),
                'min': float(np.min(self.neural_network)),
                'max': float(np.max(self.neural_network))
            },
            'architecture': f"{self.input_size}→{self.hidden1_size}→{self.hidden2_size}→{self.hidden3_size}→{self.hidden4_size}→{self.output_size}",
            'status': '已初始化'
        }
    
    # 功能: 设置神经网络权重
    def set_neural_network_weights(self, neural_network):
        """
        设置神经网络权重
        
        参数:
            neural_network: 神经网络权重数组，如果为None则清除神经网络权重
        """
        if neural_network is None:
            # 手动控制模式下允许神经网络权重为空
            self.neural_network = None
            return
            
        # 计算神经网络权重总数
        total_weights = (self.input_size * self.hidden1_size + self.hidden1_size + 
                        self.hidden1_size * self.hidden2_size + self.hidden2_size + 
                        self.hidden2_size * self.hidden3_size + self.hidden3_size + 
                        self.hidden3_size * self.hidden4_size + self.hidden4_size + 
                        self.hidden4_size * self.output_size + self.output_size)
        
        # 验证权重数组大小是否匹配
        if len(neural_network) != total_weights:
            raise ValueError(f"权重数组大小不匹配，期望{total_weights}，实际{len(neural_network)}")
            
        self.neural_network = neural_network
    
    # 功能: 使用神经网络进行决策
    def think(self, individual, goals, obstacles):
        """
        使用神经网络进行决策
        
        参数:
            goals: 目标列表
            individuals: 种群个体列表
            obstacles: 障碍物列表
        返回:
            tuple: (move_up, move_down, move_left, move_right) 四个动作的布尔值（上、下、左、右）
        """
        if not self.active:
            return False, False, False, False
            
        # 如果没有神经网络，返回默认值（不移动）
        if self.neural_network is None:
            # 手动控制模式下允许神经网络为空，返回默认不移动
            return False, False, False, False
            
        # 获取传感器输入
        inputs = self.get_sensor_inputs(individual, goals, obstacles)

        try:
            # 计算权重分割点
            w1_size = self.input_size * self.hidden1_size
            b1_size = self.hidden1_size
            w2_size = self.hidden1_size * self.hidden2_size
            b2_size = self.hidden2_size
            w3_size = self.hidden2_size * self.hidden3_size
            b3_size = self.hidden3_size
            w4_size = self.hidden3_size * self.hidden4_size
            b4_size = self.hidden4_size
            w5_size = self.hidden4_size * self.output_size
            b5_size = self.output_size
            
            # 分割权重数组
            start = 0
            w1 = self.neural_network[start:start+w1_size].reshape(self.input_size, self.hidden1_size)
            start += w1_size
            b1 = self.neural_network[start:start+b1_size]
            start += b1_size
            w2 = self.neural_network[start:start+w2_size].reshape(self.hidden1_size, self.hidden2_size)
            start += w2_size
            b2 = self.neural_network[start:start+b2_size]
            start += b2_size
            w3 = self.neural_network[start:start+w3_size].reshape(self.hidden2_size, self.hidden3_size)
            start += w3_size
            b3 = self.neural_network[start:start+b3_size]
            start += b3_size
            w4 = self.neural_network[start:start+w4_size].reshape(self.hidden3_size, self.hidden4_size)
            start += w4_size
            b4 = self.neural_network[start:start+b4_size]
            start += b4_size
            w5 = self.neural_network[start:start+w5_size].reshape(self.hidden4_size, self.output_size)
            start += w5_size
            b5 = self.neural_network[start:start+b5_size]
            
            # 五层神经网络前向传播
            # 第一层：输入层 → 隐藏层1
            hidden1 = np.dot(inputs, w1) + b1
            hidden1 = np.maximum(hidden1, 0)  # ReLU激活
            
            # 第二层：隐藏层1 → 隐藏层2
            hidden2 = np.dot(hidden1, w2) + b2
            hidden2 = np.maximum(hidden2, 0)  # ReLU激活
            
            # 第三层：隐藏层2 → 隐藏层3
            hidden3 = np.dot(hidden2, w3) + b3
            hidden3 = np.maximum(hidden3, 0)  # ReLU激活
            
            # 第四层：隐藏层3 → 隐藏层4
            hidden4 = np.dot(hidden3, w4) + b4
            hidden4 = np.maximum(hidden4, 0)  # ReLU激活
            
            # 第五层：隐藏层4 → 输出层
            output = np.dot(hidden4, w5) + b5
            
            # 输出值, 范围在-1到1之间（通过tanh激活函数）
            output = np.tanh(output)
            
            # 转换为动作
            if self.output_size >= 2:
                # 第一个输出控制上下移动：正值向上，负值向下（暂时禁用向下）
                vertical_output = output[0]  # 输出已经在[-1,1]范围内
                move_up = vertical_output > 0
                move_down = False  # 暂时禁用向下移动
                
                # 第二个输出控制左右移动：使用三分区阈值法
                # [-1, -0.33): 左移
                # [-0.33, 0.33]: 不操作
                # (0.33, 1]: 右移
                horizontal_output = output[1]  # 输出已经在[-1,1]范围内
                
                if horizontal_output < -0.33:
                    move_left = True
                    move_right = False
                elif horizontal_output > 0.33:
                    move_left = False
                    move_right = True
                else:
                    move_left = False
                    move_right = False
            else:
                raise ValueError("输出层大小必须至少为2，才能控制上下左右移动")
            
            return move_up, move_down, move_left, move_right
            
        except Exception as e:
            print(f"神经网络计算错误: {e}")
            return False, False, False, False

    # 功能: 处理障碍物的碰撞
    def on_obstacle_collisions(self, obstacles):
        """
        处理与障碍物的碰撞
        
        参数:
            obstacles: 障碍物列表
            
        返回:
            dict: 碰撞信息，包括是否站在障碍物上、是否被阻挡、是否被杀死等
        """
        
        # 初始化碰撞信息
        collision_info = {
            'on_obstacle': False,  # 是否站在障碍物上
            'blocked_left': False,  # 是否左侧被阻挡
            'blocked_right': False,  # 是否右侧被阻挡
            'blocked_top': False,  # 是否上方被阻挡
            'blocked_bottom': False,  # 是否下方被阻挡
            'killed': False  # 是否被杀死
        }
        
        # 处理与每个障碍物的碰撞
        for obstacle in obstacles:
            if not obstacle.active:
                continue
                
            # 获取障碍物的碰撞信息
            obstacle_collision = obstacle.check_collision(self.rect)
            
            # 如果发生碰撞
            if obstacle_collision['collided']:
                # 如果是致命障碍物，杀死个体
                if obstacle.kill_individual:
                    self.active = False
                    collision_info['killed'] = True
                    return collision_info
                
                # 根据碰撞方向处理
                if obstacle_collision['top']:
                    # 个体从上方碰撞障碍物（站在障碍物上）
                    # 将个体位置调整到障碍物上方
                    self.pos_y = obstacle.rect.top - self.height // 2
                    # 停止垂直下落
                    if self.vel_y > 0:
                        self.vel_y = 0
                    collision_info['on_obstacle'] = True
                    
                elif obstacle_collision['bottom']:
                    # 个体从下方碰撞障碍物（碰到障碍物底部）
                    # 将个体位置调整到障碍物下方
                    self.pos_y = obstacle.rect.bottom + self.height // 2
                    # 停止垂直上升
                    if self.vel_y < 0:
                        self.vel_y = 0
                    collision_info['blocked_top'] = True
                    
                elif obstacle_collision['left']:
                    # 个体从左侧碰撞障碍物
                    # 将个体位置调整到障碍物左侧
                    self.pos_x = obstacle.rect.left - self.width // 2
                    # 停止水平移动
                    if self.vel_x > 0:
                        self.vel_x = 0
                    collision_info['blocked_left'] = True
                    
                elif obstacle_collision['right']:
                    # 个体从右侧碰撞障碍物
                    # 将个体位置调整到障碍物右侧
                    self.pos_x = obstacle.rect.right + self.width // 2
                    # 停止水平移动
                    if self.vel_x < 0:
                        self.vel_x = 0
                    collision_info['blocked_right'] = True
        
        # 在所有障碍物碰撞处理完成后统一更新矩形位置
        self.rect.x = self.pos_x - self.width // 2
        self.rect.y = self.pos_y - self.height // 2
        
        return collision_info

    # 功能: 更新种群个体状态
    def update(self, individual, goals, obstacles, manual_control=False):
        """
        更新种群个体状态，包括平面移动、AI决策逻辑、目标碰撞检测和障碍物碰撞检测
        
        参数:
            goals: 目标列表
            individuals: 种群个体列表
            obstacles: 障碍物列表
            manual_control: 是否由手动控制，默认为False（AI控制）
        """
        if not self.active:
            return

        # # 检查是否已经到达所有目标点（目标始终激活，检查是否已标记所有目标）
        # if self.has_reached_all_goals(goals):
        #     # 停止运动，不再更新速度和位置
        #     self.vel_x = 0
        #     self.vel_y = 0
        #     self.active = False
        #     return

        # 根据控制模式决定移动逻辑
        if manual_control:
            # 手动控制模式：使用手动输入控制移动
            move_left = self.manual_left
            move_right = self.manual_right
            move_up = self.manual_up
            # move_down = self.manual_down
        else:
            # AI控制模式：使用神经网络决策
            move_up, move_down, move_left, move_right = self.think(individual, goals, obstacles)
        
        # 记录飞行次数：检测向上移动按键的按下事件
        if move_up and not self.last_move_up_state:
            # 检测到向上移动按键从松开变为按下，增加飞行次数
            self.flight_count += 1
        self.last_move_up_state = move_up  # 更新上一次状态
        
        # 检查是否在地面或障碍物上表面
        self._check_ground_or_obstacle_contact(obstacles)
        
        # 飞行能量管理
        self._manage_flight_energy(move_up)
        
        # 水平移动
        if move_left:
            self.vel_x = -setting.INDIVIDUAL_HORIZONTAL_SPEED
        elif move_right:
            self.vel_x = setting.INDIVIDUAL_HORIZONTAL_SPEED
        else:
            self.vel_x = 0  # 不操控时直接停止
        
        # 垂直移动 - 考虑能量限制
        if move_up and self.current_energy > 0:
            # 有能量时允许飞行
            self.vel_y = -setting.INDIVIDUAL_VERTICAL_SPEED
            self.is_flying = True
        else:
            # 没有能量或未按下飞行键时应用重力
            self.vel_y += setting.GRAVITY_ACCELERATION
            self.is_flying = False
        # elif move_down:
            # self.vel_y = setting.INDIVIDUAL_VERTICAL_SPEED
        
        # 调用父类的update方法更新位置
        super().update()
        
        # # 记录当前位置到轨迹
        # self.trajectory_points.append((int(self.pos_x), int(self.pos_y)))
        # 限制轨迹点数量，避免内存占用过大
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
        
        # 检查障碍物碰撞
        if obstacles:
            self.on_obstacle_collisions(obstacles)
        
        # 检查目标碰撞
        collided_goal = self.check_goal_collision(goals)
        if collided_goal:
            self.on_goal_collision(collided_goal)

    # 功能: 检查是否到达所有目标点
    def has_reached_all_goals(self, goals):
        """
        检查个体是否已经到达所有目标点（目标始终激活，检查是否已标记所有目标）
        
        参数:
            goals: 目标列表
            
        返回:
            bool: 如果已到达所有目标点返回True，否则返回False
        """
        # 如果没有目标，返回False
        if len(goals) == 0:
            return False
            
        # 检查个体是否已经标记了所有当前存在的目标
        # 需要确保个体已到达的目标ID集合包含所有当前目标ID
        current_goal_ids = {goal.entity_id for goal in goals}
        reached_goal_ids_set = set(self.reached_goal_ids)
        
        # 只有当个体已到达的目标ID集合包含所有当前目标ID时，才返回True
        has_all_goals = reached_goal_ids_set.issuperset(current_goal_ids)
        
        # 如果到达所有目标，不需要重新计算适应度，因为我们已经在on_goal_collision中累加了
        # 只需要返回True表示已完成所有目标
        return has_all_goals

    # 功能: 绘制个体实体
    def draw(self, screen, goals=None): 
        """
        在屏幕上绘制个体实体及其编号
        
        参数:
            screen (pygame.Surface): 要绘制到的目标表面
            goals (list): 目标列表，用于判断是否已到达所有目标
        """
        if not self.visible:
            return
            
        # 只绘制活着的个体，死亡的个体不绘制
        if self.active:
            # 检查是否已到达所有目标点，如果是则绘制为黄色，否则为黑色
            # 只有在到达所有目标后才显示为黄色
            if goals is not None:
                self.color = setting.YELLOW if self.has_reached_all_goals(goals) else setting.BLACK
            else:
                # 如果没有提供goals参数，则保持原来的颜色
                pass
            
            # 调用父类的draw方法来绘制形状（根据self.shape参数）
            super().draw(screen)

            # 绘制个体编号（只在个体活着时绘制）
            id_text = self.font.render(str(self.entity_id), True, setting.WHITE)
            text_rect = id_text.get_rect(center=(self.pos_x, self.pos_y))
            screen.blit(id_text, text_rect)
            
            # 绘制个体轨迹（只在个体活着时绘制）
            if len(self.trajectory_points) > 1:
                # 使用橙色绘制轨迹线条
                trajectory_color = (255, 165, 0)  # 橙色
                # 绘制连接所有轨迹点的线条
                pygame.draw.lines(screen, trajectory_color, False, self.trajectory_points, 2)
            
            # 绘制能量条（只在个体活着时绘制）
            self._draw_energy_bar(screen)
    
    # 功能: 绘制传感器输入
    def draw_sensor_inputs(self, screen, goals, obstacles=None):
        """
        绘制传感器输入效果，包括射线检测和相对位置向量线
        
        参数:
            screen: pygame屏幕对象
            goals: 目标列表
            obstacles: 障碍物列表（可选）
        """
        # 调用draw_sensor_inputs函数，并启用绘制
        draw_sensor_inputs(screen, self, goals, obstacles, draw_enabled=True)

    def reset_individual(self, x, y):
        """重置个体的全部信息"""
        self.set_position(x, y)
        self.vel_x = 0
        self.vel_y = 0
        self.active = True
        
        # 重置适应度相关属性
        self.fitness = 0
        self.distance_fitness = 0.0
        self.time_efficiency_fitness = 0.0
        self.flight_count_fitness = 0.0
        
        # 重置目标达成记录
        self.goals_reached = 0
        self.reached_goal_ids = []
        self.goal_reach_times = []
        
        # 重置手动控制状态
        self.manual_left = False
        self.manual_right = False
        self.manual_up = False
        self.manual_down = False
        
        # 重置飞行次数相关属性
        self.flight_count = 0
        self.last_move_up_state = False
        
        # 重置飞行能量相关属性
        self.current_energy = self.max_energy
        self.is_flying = False
        self.energy_recovery_timer = 0
        self.is_on_ground_or_obstacle = False
        
        # 清空轨迹记录
        self.trajectory_points = []

    def _check_boundaries(self):
        # 暂时弃用，改用边界碰撞处理
        # """检查个体是否超出边界，如果超出边界则杀死个体"""
        # # 检查是否超出任何边界
        # if (self.pos_x < self.width // 2 or 
        #     self.pos_x > setting.SCREEN_WIDTH - self.width // 2 or
        #     self.pos_y < self.height // 2 or 
        #     self.pos_y > setting.SCREEN_HEIGHT - self.height // 2):
        #     # 超出边界，杀死个体
        #     self.active = False
        #     return

        """检查个体是否超出边界，直接调用父类的边界碰撞处理"""
        # 调用父类的边界检查方法
        super()._check_boundaries()

    def _check_ground_or_obstacle_contact(self, obstacles):
        """
        检查个体是否在地面或障碍物上表面
        
        参数:
            obstacles: 障碍物列表
        """
        # 检查是否在地面（屏幕底部）
        if self.pos_y >= setting.SCREEN_HEIGHT - self.height // 2:
            self.is_on_ground_or_obstacle = True
            return
        
        # 检查是否在障碍物上表面
        self.is_on_ground_or_obstacle = False
        if obstacles:
            for obstacle in obstacles:
                # 检查个体是否正好在障碍物上表面（接触）
                # 个体底部应该与障碍物顶部接触
                individual_bottom = self.pos_y + self.height // 2
                obstacle_top = obstacle.pos_y - obstacle.height // 2
                
                # 允许一定的接触容差（2像素）
                contact_tolerance = 2
                
                # 检查垂直接触和水平重叠
                if (abs(individual_bottom - obstacle_top) <= contact_tolerance and
                    abs(self.pos_x - obstacle.pos_x) <= (self.width // 2 + obstacle.width // 2)):
                    self.is_on_ground_or_obstacle = True
                    break

    def _manage_flight_energy(self, move_up):
        """
        管理飞行能量系统
        
        参数:
            move_up: 是否正在向上移动
        """
        # 能量消耗逻辑
        if move_up and self.current_energy > 0:
            # 每次飞行扣除指定百分比的能量
            energy_cost = self.max_energy * setting.FLIGHT_ENERGY_COST_PERCENT
            self.current_energy = max(0, self.current_energy - energy_cost)
        
        # 能量恢复逻辑
        if self.current_energy <= 0:
            # 能量耗尽，需要在地面或障碍物上表面停留5帧后才能恢复
            if self.is_on_ground_or_obstacle:
                self.energy_recovery_timer += 1
                if self.energy_recovery_timer >= 5:
                    # 停留5帧后开始恢复，每帧恢复指定百分比
                    self.current_energy = min(self.max_energy, self.current_energy + self.max_energy * setting.FLIGHT_ENERGY_RECOVERY_PERCENT)
            else:
                # 不在恢复位置，重置计时器
                self.energy_recovery_timer = 0
        else:
            # 有能量时，只有在地面或障碍物上表面且不飞行时才能恢复能量
            if self.is_on_ground_or_obstacle and not move_up:
                # 每帧恢复指定百分比
                self.current_energy = min(self.max_energy, self.current_energy + self.max_energy * setting.FLIGHT_ENERGY_RECOVERY_PERCENT)
            # 重置计时器
            self.energy_recovery_timer = 0

    def _draw_energy_bar(self, screen):
        """
        在个体上方绘制能量条
        
        参数:
            screen: pygame屏幕对象
        """
        if not self.active:
            return
            
        # 能量条尺寸和位置
        bar_width = 30
        bar_height = 4
        bar_x = self.pos_x - bar_width // 2
        bar_y = self.pos_y - self.height // 2 - 10  # 在个体上方
        
        # 绘制背景条（灰色）
        pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        
        # 计算当前能量百分比
        energy_percentage = self.current_energy / self.max_energy
        
        # 根据能量百分比选择颜色
        if energy_percentage > 0.7:
            color = (0, 255, 0)  # 绿色（高能量）
        elif energy_percentage > 0.3:
            color = (255, 255, 0)  # 黄色（中等能量）
        else:
            color = (255, 0, 0)  # 红色（低能量）
        
        # 绘制当前能量条
        current_width = int(bar_width * energy_percentage)
        if current_width > 0:
            pygame.draw.rect(screen, color, (bar_x, bar_y, current_width, bar_height))

