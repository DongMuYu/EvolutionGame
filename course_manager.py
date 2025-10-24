import random
from entities.goal import Goal
from entities.individual import Individual
from entities.obstacle import Obstacle
from course_elements.level1_elements import Level1ElementsManager
from course_elements.level2_elements import Level2ElementsManager
from course_elements.level3_elements import Level3ElementsManager
import setting

class CourseManager:
    """
    课程管理类，负责训练器的课程学习设置
    提供便于扩展的难度级别框架，每个级别可自定义目标和障碍物初始化
    """
    
    def __init__(self):
        # 课程难度设置
        self.current_difficulty = 2  # 当前难度级别
        self.max_difficulty = 5      # 最大难度级别
        
        # 难度专用元素管理器
        self.level_managers = {
            1: Level1ElementsManager(),  # 难度一：静态元素管理器
            2: Level2ElementsManager(),  # 难度二：跑酷障碍物系统
            3: Level3ElementsManager(),  # 难度三：高级动态元素管理器（待实现）
            4: None,  # 难度四：待实现
            5: None   # 难度五：待实现
        }

    def initialize_individuals(self, population_manager, frame_counter_ref=None, difficulty_level=None):
        """初始化种群个体，根据难度级别调整个体设置
        
        参数:
            population_manager: 种群管理器实例
            frame_counter_ref: 帧计数器引用函数
            difficulty_level: 难度级别，如果为None则使用当前难度
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        # 根据难度级别调用对应的初始化方法
        if difficulty_level == 1:
            # 第一级难度：原始随机生成方法
            individuals = self._initialize_individuals_level_1(population_manager, frame_counter_ref)
        elif difficulty_level == 2:
            # 第二级难度：障碍跑酷
            individuals = self._initialize_individuals_level_2(population_manager, frame_counter_ref)
        elif difficulty_level == 3:
            # 第三级难度：待补充 - 请在此处添加您的实现
            individuals = self._initialize_individuals_level_3(population_manager, frame_counter_ref)
        elif difficulty_level == 4:
            # 第四级难度：待补充 - 请在此处添加您的实现
            individuals = self._initialize_individuals_level_4(population_manager, frame_counter_ref)
        else:
            # 第五级难度：待补充 - 请在此处添加您的实现
            individuals = self._initialize_individuals_level_5(population_manager, frame_counter_ref)
        
        return individuals

    def _initialize_individuals_level_1(self, population_manager, frame_counter_ref=None):
        """第一级难度个体初始化：使用population_manager的初始化函数"""
        # 屏幕中心坐标
        center_x = setting.SCREEN_WIDTH // 2
        center_y = setting.SCREEN_HEIGHT // 2
        
        return population_manager.initialize_individuals(center_x, center_y, frame_counter_ref)

    def _initialize_individuals_level_2(self, population_manager, frame_counter_ref=None):
        """第二级难度个体初始化：待补充"""
        
        # 屏幕中心坐标
        center_x = setting.SCREEN_WIDTH // 2
        center_y = setting.SCREEN_HEIGHT // 2

        return population_manager.initialize_individuals(center_x, center_y, frame_counter_ref)

    def _initialize_individuals_level_3(self, population_manager, frame_counter_ref=None):
        """第三级难度个体初始化：待补充"""
        # 请在此处添加第三级难度的个体初始化逻辑
        return self._initialize_individuals_level_1(population_manager, frame_counter_ref)  # 暂时使用第一级方法

    def _initialize_individuals_level_4(self, population_manager, frame_counter_ref=None):
        """第四级难度个体初始化：待补充"""
        # 请在此处添加第四级难度的个体初始化逻辑
        return self._initialize_individuals_level_1(population_manager, frame_counter_ref)  # 暂时使用第一级方法

    def _initialize_individuals_level_5(self, population_manager, frame_counter_ref=None):
        """第五级难度个体初始化：待补充"""
        # 请在此处添加第五级难度的个体初始化逻辑
        return self._initialize_individuals_level_1(population_manager, frame_counter_ref)  # 暂时使用第一级方法
        
    def initialize_goals(self, goal_count=None, difficulty_level=None):
        """初始化目标，根据难度级别调整目标设置
        
        参数:
            goal_count: 目标数量，如果为None则使用setting.GOAL_COUNT
            difficulty_level: 难度级别，如果为None则使用当前难度
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        # 根据难度级别调用对应的初始化方法
        if difficulty_level == 1:
            # 第一级难度：原始随机生成方法
            goals = self._initialize_goals_level_1(goal_count)
        elif difficulty_level == 2:
            # 第二级难度：障碍跑酷，在上下平台夹出的管道中随机生成目标
            goals = self._initialize_goals_level_2(goal_count)
        elif difficulty_level == 3:
            # 第三级难度：待补充 - 请在此处添加您的实现
            goals = self._initialize_goals_level_3(goal_count)
        elif difficulty_level == 4:
            # 第四级难度：待补充 - 请在此处添加您的实现
            goals = self._initialize_goals_level_4(goal_count)
        else:
            # 第五级难度：待补充 - 请在此处添加您的实现
            goals = self._initialize_goals_level_5(goal_count)
        
        return goals

    def _initialize_goals_level_1(self, goal_count=None):
        """第一级难度目标初始化：原始随机生成方法"""
        if goal_count is None:
            goal_count = setting.GOAL_COUNT
            
        goals = []
        for i in range(goal_count):
            # 随机选择屏幕左侧或右侧
            side = random.choice(['left', 'right'])
            
            if side == 'left':
                # 左侧区域：屏幕宽度的1/5范围内
                pos_x = random.uniform(setting.GOAL_SIZE, setting.SCREEN_WIDTH * 0.2)
            else:
                # 右侧区域：屏幕宽度的4/5到右侧边缘
                pos_x = random.uniform(setting.SCREEN_WIDTH * 0.8, setting.SCREEN_WIDTH - setting.GOAL_SIZE)
            
            # Y坐标在屏幕范围内随机
            pos_y = random.uniform(setting.GOAL_SIZE, setting.SCREEN_HEIGHT - setting.GOAL_SIZE)
            
            # 创建目标，分配编号
            goal = Goal(pos_x, pos_y, goal_id=i)
            goals.append(goal)
        
        return goals

    def _initialize_goals_level_2(self, goal_count=None):
        """第二级难度目标初始化：通道中的任意位置生成一个目标"""
        # 强制只生成一个目标
        goal_count = 1
            
        goals = []
        
        # 通道参数
        platform_thickness = 250  # 平台厚度
        channel_top = platform_thickness  # 通道顶部y坐标
        channel_bottom = setting.SCREEN_HEIGHT - platform_thickness  # 通道底部y坐标
        
        # 生成目标，通道中的任意位置
        for i in range(goal_count):
            # Y坐标在通道范围内随机生成
            pos_y = random.uniform(channel_top + setting.GOAL_SIZE, channel_bottom - setting.GOAL_SIZE)
            
            # X坐标在屏幕中间区域随机生成
            pos_x = random.uniform(setting.SCREEN_WIDTH * 0.3, setting.SCREEN_WIDTH * 0.7)
            
            # 创建目标，分配编号
            goal = Goal(pos_x, pos_y, goal_id=i)
            goals.append(goal)
        
        return goals

    def _initialize_goals_level_3(self, goal_count=None):
        """第三级难度目标初始化：待补充"""
        if goal_count is None:
            goal_count = setting.GOAL_COUNT
            
        # 请在此处添加第三级难度的目标初始化逻辑
        return self._initialize_goals_level_1(goal_count)  # 暂时使用第一级方法

    def _initialize_goals_level_4(self, goal_count=None):
        """第四级难度目标初始化：待补充"""
        if goal_count is None:
            goal_count = setting.GOAL_COUNT
            
        # 请在此处添加第四级难度的目标初始化逻辑
        return self._initialize_goals_level_1(goal_count)  # 暂时使用第一级方法

    def _initialize_goals_level_5(self, goal_count=None):
        """第五级难度目标初始化：待补充"""
        if goal_count is None:
            goal_count = setting.GOAL_COUNT
            
        # 请在此处添加第五级难度的目标初始化逻辑
        return self._initialize_goals_level_1(goal_count)  # 暂时使用第一级方法

    def initialize_obstacles(self, difficulty_level=None):
        """初始化障碍物，根据难度级别调整障碍物数量和属性
        
        参数:
            difficulty_level: 难度级别，如果为None则使用当前难度
            corner_position: 个体角落位置，如果为None则使用内部存储的位置
            target_corner: 目标角落位置，如果为None则使用内部存储的位置
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        obstacles = []
        
        # 根据难度级别调用对应的初始化方法
        if difficulty_level == 1:
            # 第一级难度：原始随机生成方法
            obstacles = self._initialize_obstacles_level_1()
        elif difficulty_level == 2:
            # 第二级难度：障碍跑酷
            obstacles = self._initialize_obstacles_level_2()
        elif difficulty_level == 3:
            # 第三级难度：待补充 - 请在此处添加您的实现
            obstacles = self._initialize_obstacles_level_3()
        elif difficulty_level == 4:
            # 第四级难度：待补充 - 请在此处添加您的实现
            obstacles = self._initialize_obstacles_level_4()
        else:
            # 第五级难度：待补充 - 请在此处添加您的实现
            obstacles = self._initialize_obstacles_level_5()
        
        return obstacles

    def _initialize_obstacles_level_1(self):
        """第一级难度障碍物初始化"""
        obstacles = []
        
        return obstacles

    def _initialize_obstacles_level_2(self):
        """第二级难度障碍物初始化：上边界和下边界平面障碍物 + 跑酷障碍物系统 + 管道两端危险障碍物"""
        obstacles = []
        
        # 障碍物厚度
        obstacle_thickness = 250
        
        # 创建上边界平面障碍物
        top_obstacle = Obstacle(
            x=0, 
            y=0, 
            width=setting.SCREEN_WIDTH, 
            height=obstacle_thickness,
            kill_individual=False
        )
        obstacles.append(top_obstacle)
        
        # 创建下边界平面障碍物
        bottom_obstacle = Obstacle(
            x=0, 
            y=setting.SCREEN_HEIGHT - obstacle_thickness, 
            width=setting.SCREEN_WIDTH, 
            height=obstacle_thickness,
            kill_individual=False
        )
        obstacles.append(bottom_obstacle)
        
        # 在管道两端添加薄薄的危险障碍物（2个格子厚度）
        danger_thickness = 2  # 危险障碍物厚度
        channel_height = setting.SCREEN_HEIGHT - 2 * obstacle_thickness  # 管道高度
        
        # 左端危险障碍物（从管道顶部到底部）
        left_danger_obstacle = Obstacle(
            x=0,
            y=obstacle_thickness,  # 从上平台底部开始
            width=danger_thickness,
            height=channel_height,
            kill_individual=True,
            color=setting.RED
        )
        obstacles.append(left_danger_obstacle)
        
        # 右端危险障碍物（从管道顶部到底部）
        right_danger_obstacle = Obstacle(
            x=setting.SCREEN_WIDTH - danger_thickness,
            y=obstacle_thickness,  # 从上平台底部开始
            width=danger_thickness,
            height=channel_height,
            kill_individual=True,
            color=setting.RED
        )
        obstacles.append(right_danger_obstacle)
        
        # 重置跑酷障碍物系统设置（每一局重新随机）
        if self.level_managers[2]:
            self.level_managers[2].reset_for_new_round()
        
        return obstacles

    def _initialize_obstacles_level_3(self):
        """第三级难度障碍物初始化：待补充"""
        # 请在此处添加第三级难度的障碍物初始化逻辑
        return []

    def _initialize_obstacles_level_4(self):
        """第四级难度障碍物初始化：待补充"""
        # 请在此处添加第四级难度的障碍物初始化逻辑
        return []

    def _initialize_obstacles_level_5(self):
        """第五级难度障碍物初始化：待补充"""
        # 请在此处添加第五级难度的障碍物初始化逻辑
        return []

    def increase_difficulty(self):
        """增加课程难度"""
        if self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            print(f"难度级别提升到: {self.current_difficulty}")
        else:
            print("已达到最高难度级别")
    
    def decrease_difficulty(self):
        """降低课程难度"""
        if self.current_difficulty > 1:
            self.current_difficulty -= 1
            print(f"难度级别降低到: {self.current_difficulty}")
        else:
            print("已是最低难度级别")
    
    def set_difficulty(self, level):
        """设置特定难度级别
        
        参数:
            level: 难度级别 (1-5)
        """
        if 1 <= level <= self.max_difficulty:
            self.current_difficulty = level
            print(f"设置难度级别为: {self.current_difficulty}")
        else:
            print(f"无效的难度级别，请选择1-{self.max_difficulty}")

    def get_difficulty_info(self, difficulty_level=None):
        """获取难度级别的详细信息"""
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        if difficulty_level == 1:
            return "第一级：中心初始位置，原始随机目标布局，随机障碍物（1-3个）"
        elif difficulty_level == 2:
            return "第二级：移动墙跑酷模式，目标随机生成在两个移动障碍物之间的通道中，高度随意"
        elif difficulty_level == 3:
            return "第三级：待补充 - 请实现_initialize_individuals_level_3、_initialize_goals_level_3和_initialize_obstacles_level_3方法"
        elif difficulty_level == 4:
            return "第四级：待补充 - 请实现_initialize_individuals_level_4、_initialize_goals_level_4和_initialize_obstacles_level_4方法"
        else:
            return "第五级：待补充 - 请实现_initialize_individuals_level_5、_initialize_goals_level_5和_initialize_obstacles_level_5方法"

    def reset_course(self):
        """重置课程设置"""
        self.current_difficulty = 1
        print("课程设置已重置为默认难度")

    def get_course_elements(self, difficulty_level=None):
        """获取指定难度级别的课程元素（个体、目标、障碍物）
        
        参数:
            difficulty_level: 难度级别，如果为None则使用当前难度
            
        返回:
            tuple: (individuals, goals, obstacles) - 个体列表、目标列表、障碍物列表
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        # 创建临时的PopulationManager实例用于初始化个体
        from population_manager import PopulationManager
        temp_population_manager = PopulationManager()
        
        # 初始化个体
        individuals = self.initialize_individuals(temp_population_manager, difficulty_level=difficulty_level)
        
        # 初始化目标
        goals = self.initialize_goals(difficulty_level=difficulty_level)
        
        # 初始化障碍物
        obstacles = self.initialize_obstacles(difficulty_level=difficulty_level)
        
        return individuals, goals, obstacles

    def update_course_elements(self, goals, obstacles, difficulty_level=None):
        """更新课程元素（目标和障碍物），根据难度级别调整更新逻辑
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
            difficulty_level: 难度级别，如果为None则使用当前难度
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
            
        # 根据难度级别调用对应的更新方法
        if difficulty_level == 1:
            # 第一级难度：原始更新方法
            self._update_course_elements_level_1(goals, obstacles)
        elif difficulty_level == 2:
            # 第二级难度：跑酷障碍物系统和动态目标生成
            self._update_course_elements_level_2(goals, obstacles)
        elif difficulty_level == 3:
            # 第三级难度：待补充
            self._update_course_elements_level_3(goals, obstacles)
        elif difficulty_level == 4:
            # 第四级难度：待补充
            self._update_course_elements_level_4(goals, obstacles)
        else:
            # 第五级难度：待补充
            self._update_course_elements_level_5(goals, obstacles)

    def _update_course_elements_level_1(self, goals, obstacles):
        """第一级难度课程元素更新：静态元素管理
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
        """
        # 使用难度一专用元素管理器更新静态元素
        if self.level_managers[1]:
            self.level_managers[1].update_goals_level_1(goals)
            self.level_managers[1].update_obstacles_level_1(obstacles)

    def _update_course_elements_level_2(self, goals, obstacles):
        """第二级难度课程元素更新：跑酷障碍物系统和动态目标生成
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
        """
        # 使用难度二专用元素管理器更新跑酷障碍物和动态目标
        if self.level_managers[2]:
            self.level_managers[2].update_obstacles_level_2(obstacles)
            self.level_managers[2].update_goals_level_2(goals)
            
    def _update_course_elements_level_3(self, goals, obstacles):
        """第三级难度课程元素更新：高级动态元素管理
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
        """
        # 使用难度三专用元素管理器更新高级动态元素
        if self.level_managers[3]:
            self.level_managers[3].update_goals_level_3(goals)
            self.level_managers[3].update_obstacles_level_3(obstacles)

    def _update_course_elements_level_4(self, goals, obstacles):
        """第四级难度课程元素更新：待补充
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
        """
        # 第四级难度：待补充 - 请在此处添加第四级难度的更新逻辑
        pass

    def _update_course_elements_level_5(self, goals, obstacles):
        """第五级难度课程元素更新：待补充
        
        参数:
            goals: 目标列表
            obstacles: 障碍物列表
        """
        # 第五级难度：待补充 - 请在此处添加第五级难度的更新逻辑
        pass