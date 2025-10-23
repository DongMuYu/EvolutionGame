"""
动态元素管理模块
专门负责课程学习中的动态元素（障碍物、目标）的更新和管理
"""

import random
import setting
from entities.obstacle import Obstacle
from entities.goal import Goal


class Level2ElementsManager:
    """第二级难度元素管理器，负责跑酷障碍物和动态目标的生成、移动和销毁"""
    
    def __init__(self):
        # 跑酷障碍物系统设置
        self.parkour_obstacles = []  # 存储当前活动的跑酷障碍物
        self.parkour_spawn_timer = 0  # 障碍物生成计时器
        self.parkour_spawn_interval = 40  # 初始生成间隔（帧数）
        
        # 障碍物和目标移动方向设置（每一局随机决定）
        self.obstacle_direction = random.choice([-1, 1])  # -1: 从右往左, 1: 从左往右

        # 障碍物生成位置交替控制
        self.last_obstacle_position = 'bottom'  # 上一次障碍物生成位置
        
        # 目标ID计数器
        self.goal_id_counter = 0  # 目标ID累加计数器
    
    def reset_for_new_round(self):
        """每局开始时重置跑酷障碍物系统"""
        # 清空跑酷障碍物列表
        self.parkour_obstacles.clear()
        
        # 重置计时器和间隔
        self.parkour_spawn_timer = 0
        self.parkour_spawn_interval = 40
        
        # 重新随机障碍物与目标的移动方向
        self.obstacle_direction = random.choice([-1, 1])
        
        # 重置障碍物生成位置交替控制
        self.last_obstacle_position = 'bottom'
        
        # 重置目标ID计数器
        self.goal_id_counter = 0
    
    def update_obstacles_level_2(self, obstacles):
        """更新第二级难度障碍物系统：生成、移动和销毁跑酷障碍物
        
        参数:
            obstacles: 障碍物列表
        """
        # 更新生成计时器
        self.parkour_spawn_timer += 1
        
        # 检查是否需要生成新的跑酷障碍物
        if self.parkour_spawn_timer >= self.parkour_spawn_interval:
            self.spawn_parkour_obstacle(obstacles)
            self.parkour_spawn_timer = 0
            # 随机生成下一个障碍物的时间间隔（70-110帧） 
            self.parkour_spawn_interval = random.randint(70, 110)
        
        # 更新所有跑酷障碍物的位置
        obstacles_to_remove = []
        for obstacle in self.parkour_obstacles:
            # 根据随机方向移动障碍物
            obstacle.rect.x += 4 * self.obstacle_direction  # 移动速度，根据方向调整（从8降低到4）
            obstacle.pos_x = obstacle.rect.centerx
            
            # 检查障碍物是否移出屏幕
            if self.obstacle_direction == -1:  # 从右往左移动
                # 检查障碍物是否移出屏幕左侧
                if obstacle.rect.right < 0:
                    obstacles_to_remove.append(obstacle)
            else:  # 从左往右移动
                # 检查障碍物是否移出屏幕右侧
                if obstacle.rect.left > setting.SCREEN_WIDTH:
                    obstacles_to_remove.append(obstacle)
        
        # 移除移出屏幕的障碍物
        for obstacle in obstacles_to_remove:
            if obstacle in self.parkour_obstacles:
                self.parkour_obstacles.remove(obstacle)
            if obstacle in obstacles:
                obstacles.remove(obstacle)
    
    def spawn_parkour_obstacle(self, obstacles):
        """生成一个新的跑酷障碍物，交替生成上下表面障碍物
        
        参数:
            obstacles: 障碍物列表
        """
        # 计算通道高度和最大障碍物高度
        platform_thickness = 250  # 平台厚度
        channel_height = setting.SCREEN_HEIGHT - 2 * platform_thickness  # 通道高度
        max_obstacle_height = int(channel_height * 3 / 5)  # 最大障碍物高度（通道的3/5）
        
        # 随机障碍物高度（30-最大障碍物高度像素）
        obstacle_height = random.randint(30, max_obstacle_height)
        
        # 随机障碍物宽度（20-60像素）
        obstacle_width = random.randint(20, 60)
        
        # 交替生成上下表面障碍物
        if self.last_obstacle_position == 'bottom':
            # 这次生成上表面障碍物
            top_platform_y = platform_thickness  # 上平台y坐标
            obstacle_y = top_platform_y  # 贴着上平台下表面
            self.last_obstacle_position = 'top'
            position_desc = "上表面"
        else:
            # 这次生成下表面障碍物
            bottom_platform_y = setting.SCREEN_HEIGHT - platform_thickness  # 下平台y坐标
            obstacle_y = bottom_platform_y - obstacle_height  # 贴着下平台上表面
            self.last_obstacle_position = 'bottom'
            position_desc = "下表面"
        
        # 根据随机方向决定生成位置
        if self.obstacle_direction == -1:  # 从右往左
            obstacle_x = setting.SCREEN_WIDTH
            direction_desc = "从右往左"
        else:  # 从左往右
            obstacle_x = 0 - obstacle_width
            direction_desc = "从左往右"
        
        # 创建红色障碍物（会杀死个体）
        parkour_obstacle = Obstacle(
            x=obstacle_x,
            y=obstacle_y,
            width=obstacle_width,
            height=obstacle_height,
            kill_individual=True,
            color=setting.RED
        )
        
        # 添加到障碍物列表和跑酷障碍物列表
        obstacles.append(parkour_obstacle)
        self.parkour_obstacles.append(parkour_obstacle)
        
        print(f"生成跑酷障碍物：{position_desc}，{direction_desc}，位置({obstacle_x}, {obstacle_y})，尺寸({obstacle_width}x{obstacle_height})")
    
    def update_goals_level_2(self, goals):
        """更新第二级难度目标系统：在屏幕中只存在一个目标，如果当前有目标则删除旧目标生成新目标
        
        参数:
            goals: 目标列表
        """
        # 通道参数
        platform_thickness = 250  # 平台厚度
        channel_height = setting.SCREEN_HEIGHT - 2 * platform_thickness  # 通道高度
        channel_top = platform_thickness  # 通道顶部y坐标
        channel_bottom = setting.SCREEN_HEIGHT - platform_thickness  # 通道底部y坐标
        
        # 检查是否需要生成新目标（基于障碍物生成间隔）
        if self.parkour_spawn_timer == 0:  # 当障碍物刚生成时
            # 随机决定是否生成目标（不一定要每一个间隔都生成）
            if random.random() < 0.35:  # 35%的概率生成目标
                # 如果当前有目标存在，先删除旧目标
                if goals:
                    old_goal = goals[0]
                    goals.clear()
                    print(f"删除旧目标：ID:{old_goal.entity_id}")
                
                # Y坐标在通道内随机生成
                max_y = channel_bottom - setting.GOAL_SIZE  # 最大Y坐标（避免超出边界）
                min_y = channel_top + setting.GOAL_SIZE  # 最小Y坐标（避免超出边界）
                pos_y = random.uniform(min_y, max_y)
                
                # X坐标在屏幕内随机生成（避免太靠近边界）
                min_x = setting.GOAL_SIZE  # 最小X坐标（避免超出左边界）
                max_x = setting.SCREEN_WIDTH - setting.GOAL_SIZE  # 最大X坐标（避免超出右边界）
                pos_x = random.uniform(min_x, max_x)
                direction_desc = "随机位置"
                
                # 创建目标，分配编号（ID累加）
                goal_id = self.goal_id_counter
                goal = Goal(pos_x, pos_y, goal_id=goal_id)
                goals.append(goal)
                
                # 累加目标ID计数器
                self.goal_id_counter += 1
                
                print(f"生成动态目标：{direction_desc}，位置({pos_x:.1f}, {pos_y:.1f})，ID:{goal_id}")
        