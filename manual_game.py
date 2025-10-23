import pygame
import os
import re
import sys
from graphics import GameGraphics
from population_manager import PopulationManager
from course_manager import CourseManager
from entities.individual import Individual
import setting

class ManualGame:
    """
    手动控制游戏类，用于测试参数调整效果
    支持手动控制和AI控制两种模式
    """
    
    def __init__(self, render=True, use_ai=False):
        # 训练状态
        self.running = True
        
        # 帧计数器
        self.frame_count = 0  # 统一的帧计数器
        
        # 控制模式
        self.use_ai = use_ai  # 是否使用AI控制
        
        # 当前选中的个体索引
        self.current_individual_index = 0
        
        # 游戏暂停状态
        self.paused = False
        
        # 创建课程管理器
        self.course_manager = CourseManager()
        
        # 让用户选择课程难度
        self._select_course_difficulty()
        
        # 种群管理器，用于加载模型
        self.population_manager = PopulationManager()
        
        # 加载的种群列表
        self.loaded_population = []
        self.loaded_generation = 0
        self.loaded_fitness_scores = []
        
        # 尝试加载最新模型
        self._load_latest_model()
        
        # 创建图形处理对象（在选择课程难度后）
        self.graphics = GameGraphics()
        
        # 创建一个手动控制的个体或AI控制的个体
        pos_x = setting.SCREEN_WIDTH // 2
        pos_y = setting.SCREEN_HEIGHT // 2
        
        # 获取遗传算法的神经网络配置
        network_config = self.population_manager.genetic_algorithm.get_network_config()
        
        if self.use_ai and self.loaded_population:
            # 使用AI控制，从加载的种群中选择个体
            neural_network = self._get_current_individual_weights()
            self.individual = Individual(
                                  width=setting.INDIVIDUAL_WIDTH,
                                  height=setting.INDIVIDUAL_HEIGHT,
                                  pos_x=pos_x, pos_y=pos_y, individual_id=0, 
                                  neural_network=neural_network, frame_counter_ref=0,
                                  input_size=network_config['input_size'],
                                  hidden1_size=network_config['hidden1_size'],
                                  hidden2_size=network_config['hidden2_size'],
                                  hidden3_size=network_config['hidden3_size'],
                                  hidden4_size=network_config['hidden4_size'],
                                  output_size=network_config['output_size'],
            )
            print(f"使用AI控制，当前个体索引: {self.current_individual_index}, 适应度: {self._get_current_individual_fitness():.3f}")
        else:
            # 使用手动控制
            self.individual = Individual(width=setting.INDIVIDUAL_WIDTH, height=setting.INDIVIDUAL_HEIGHT, pos_x=pos_x, pos_y=pos_y, individual_id=999, neural_network=None, 
                                      frame_counter_ref=lambda: self.frame_count,
                                      input_size=network_config['input_size'],
                                      hidden1_size=network_config['hidden1_size'],
                                      hidden2_size=network_config['hidden2_size'],
                                      hidden3_size=network_config['hidden3_size'],
                                      hidden4_size=network_config['hidden4_size'],
                                      output_size=network_config['output_size'])
            print("使用手动控制")
        
        # 创建目标和障碍物
        self.goals = []
        self.obstacles = []
        self.initialize_course_elements()
        
        # 适应度
        self.fitness = 0
        
        # 控制说明
        self._print_controls()
    
    def _select_course_difficulty(self):
        """让用户选择课程难度，使用CourseManager的API"""
        print("\n=== 课程选择 ===")
        
        # 使用CourseManager的API获取难度信息
        for level in range(1, 4):  # 只显示1-3级难度
            difficulty_info = self.course_manager.get_difficulty_info(level)
            print(f"{level}. {difficulty_info}")
        
        print("==================")
        
        while True:
            try:
                choice = int(input("请选择课程难度 (1-3): "))
                if 1 <= choice <= 3:
                    # 使用CourseManager的API设置难度
                    self.course_manager.set_difficulty(choice)
                    self.selected_course_level = choice
                    print(f"已选择课程难度: {choice}")
                    break
                else:
                    print("请输入有效的数字 (1-3)")
            except ValueError:
                print("请输入有效的数字 (1-3)")
    
    def _load_latest_model(self):
        """加载最新的模型文件"""
        # 检查模型目录是否存在
        save_dir = "saved_models"
        if not os.path.exists(save_dir):
            print("模型目录不存在")
            return
            
        # 获取所有模型文件
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.npz')]
        
        if not model_files:
            print("未找到模型文件")
            return
            
        # 从文件名中提取代数
        def extract_generation(filename):
            match = re.search(r'generation_(\d+)_fitness', filename)
            if match:
                return int(match.group(1))
            return 0
        
        # 按代数排序，选择最新的模型
        latest_model = max(model_files, key=extract_generation)
        
        try:
            # 构造完整的文件路径
            model_path = os.path.join(save_dir, latest_model)
            
            # 加载模型
            genetic_algorithm = self.population_manager.genetic_algorithm
            self.loaded_population, self.loaded_generation, self.loaded_fitness_scores, _, _ = genetic_algorithm.load_population(model_path)
            print(f"加载模型: {latest_model}, 代数: {self.loaded_generation}, 种群大小: {len(self.loaded_population)}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.loaded_population = []
    
    def _get_current_individual_weights(self):
        """获取当前选中个体的权重"""
        if not self.loaded_population or self.current_individual_index >= len(self.loaded_population):
            return None
        
        # 确保从字典中获取权重数组
        if isinstance(self.loaded_population[self.current_individual_index], dict) and 'weights' in self.loaded_population[self.current_individual_index]:
            return self.loaded_population[self.current_individual_index]['weights']
        else:
            # 如果不是字典格式，可能是旧格式，直接使用
            return self.loaded_population[self.current_individual_index]
    
    def _get_current_individual_fitness(self):
        """获取当前选中个体的适应度"""
        if not self.loaded_fitness_scores or self.current_individual_index >= len(self.loaded_fitness_scores):
            return 0
        return self.loaded_fitness_scores[self.current_individual_index]
    
    def _switch_individual(self, direction):
        """切换个体"""
        if not self.loaded_population:
            print("没有加载的模型，无法切换个体")
            return
        
        # 根据方向调整索引
        if direction == "next":
            self.current_individual_index = (self.current_individual_index + 1) % len(self.loaded_population)
        elif direction == "prev":
            self.current_individual_index = (self.current_individual_index - 1) % len(self.loaded_population)
        
        # 更新个体的神经网络权重
        if self.use_ai:
            neural_network = self._get_current_individual_weights()
            self.individual.set_neural_network_weights(neural_network)
            print(f"切换到个体 {self.current_individual_index}, 适应度: {self._get_current_individual_fitness():.3f}")
        
        # 重置游戏状态
        self.reset_game()
    
    def _print_controls(self):
        """打印控制说明"""
        print("=== 手动控制游戏说明 ===")
        if self.use_ai:
            print("当前模式: AI控制")
            print("A/←: 切换到上一个个体")
            print("D/→: 切换到下一个个体")
        else:
            print("当前模式: 手动控制")
            print("W/↑: 向上移动")
            print("S/↓: 向下移动")
            print("A/←: 向左移动")
            print("D/→: 向右移动")
        
        print("R: 重置位置")
        print("M: 切换控制模式 (手动/AI)")
        print("ESC: 退出游戏")
        print("======================")
    
    def initialize_course_elements(self):
        """根据选择的课程难度初始化目标和障碍物"""
        # 获取指定难度的课程元素
        individuals, goals, obstacles = self.course_manager.get_course_elements(self.selected_course_level)
        
        # 设置目标
        self.goals = goals
        
        # 设置障碍物
        self.obstacles = obstacles
    
    def handle_input(self):
        """处理键盘输入"""
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    pygame.quit()
                    return
                # 切换课程难度 (数字键1-3)
                elif event.key == pygame.K_1:
                    self.selected_course_level = 1
                    self.reset_game()
                elif event.key == pygame.K_2:
                    self.selected_course_level = 2
                    self.reset_game()
                elif event.key == pygame.K_3:
                    self.selected_course_level = 3
                    self.reset_game()
                elif event.key == pygame.K_r:
                    # 重置位置和状态
                    self.reset_game()
                    print("游戏已重置")
                elif event.key == pygame.K_m:
                    # 切换控制模式
                    self.use_ai = not self.use_ai
                    if self.use_ai and self.loaded_population:
                        # 切换到AI模式，加载当前个体的权重
                        neural_network = self._get_current_individual_weights()
                        self.individual.set_neural_network_weights(neural_network)
                        print(f"切换到AI控制，当前个体索引: {self.current_individual_index}, 适应度: {self._get_current_individual_fitness():.3f}")
                    else:
                        # 切换到手动模式，清除神经网络权重
                        self.individual.set_neural_network_weights(None)
                        print("切换到手动控制")
                    
                    self._print_controls()
                    self.reset_game()
                elif event.key == pygame.K_SPACE:
                    # 空格键暂停/继续游戏
                    self.paused = not self.paused
                    if self.paused:
                        print("游戏已暂停")
                    else:
                        print("游戏继续")
                elif self.use_ai:
                    # AI控制模式下的按键事件
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self._switch_individual("prev")
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self._switch_individual("next")
        
        # 只有当个体存活时才处理移动控制
        if self.individual.active and not self.use_ai:
            # 手动控制模式 - 获取持续按键状态
            keys = pygame.key.get_pressed()
            
            # 设置手动控制标志
            self.individual.manual_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            self.individual.manual_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            self.individual.manual_up = keys[pygame.K_w] or keys[pygame.K_UP]
            self.individual.manual_down = False  # 禁用向下移动
    
    # 功能: 更新游戏状态
    def update(self):
        """更新游戏状态，包括个体移动、边界检查和目标碰撞检测"""
        # 如果游戏暂停，不更新游戏逻辑
        if self.paused:
            return
        
        # 使用统一更新函数更新课程元素
        self.course_manager.update_course_elements(self.goals, self.obstacles, self.selected_course_level)
        
        # 更新个体位置（AI模式或手动模式）
        if self.individual.active:
            if self.use_ai:
                # AI控制模式：调用个体的update方法
                # 这个方法包含AI决策、移动、重力、边界检查和目标碰撞检测
                self.individual.update(self.individual, self.goals, obstacles=self.obstacles)
            else:
                # 手动控制模式：使用统一的update方法，设置manual_control=True
                # 这个方法包含移动、重力、边界检查和目标碰撞检测
                self.individual.update(self.individual, self.goals, obstacles=self.obstacles, manual_control=True)

        # 更新适应度
        self.fitness = self.individual.calculate_fitness(self.goals, self.obstacles)
        
        # 检查是否到达所有目标点
        if self.individual.has_reached_all_goals(self.goals):
            print("个体已到达所有目标点！")
            # 不再自动重置游戏，让玩家手动重置
            return

        # 检查个体是否死亡（适用于AI和手动模式）
        if not self.individual.active:
            print(f"个体死亡! 适应度: {self.fitness:.3f}")
            # 不再自动重置游戏，让玩家手动重置
            return
    
    def reset_game(self):
        """重置游戏状态"""
        # 重置帧计数器
        self.frame_count = 0
        
        # 重置个体
        self.individual.reset_individual(setting.SCREEN_WIDTH // 2, setting.SCREEN_HEIGHT // 2)
        
        # 如果是AI模式，重新加载当前个体的神经网络权重
        if self.use_ai and self.loaded_population:
            neural_network = self._get_current_individual_weights()
            self.individual.set_neural_network_weights(neural_network)
        
        # 根据选择的课程难度重新初始化目标和障碍物
        self.initialize_course_elements()
        
        # 重置适应度
        self.fitness = 0
        
        print(f"游戏已重置，当前课程难度: {self.selected_course_level}")
    
    def draw(self):
        """绘制游戏画面"""
        try:
            # 绘制游戏对象
            self.graphics.draw_game_objects([self.individual], self.goals, self.obstacles)
            
            # 绘制障碍物
            for obstacle in self.obstacles:
                obstacle.draw(self.graphics.screen)
            
            # 绘制传感器射线（在绘制游戏对象之后）
            if self.individual.active:
                self.individual.draw_sensor_inputs(self.graphics.screen, self.goals, self.obstacles)
            
            # 绘制UI信息
            individuals = [self.individual]
            individual_fitness = {0: self.fitness}
            generation = self.loaded_generation if self.use_ai else 0  # 手动模式不使用代数
            
            self.graphics.draw_ui(individuals, individual_fitness, generation, self.frame_count)
            
            # 绘制额外信息
            self.draw_extra_info()
            
            # 更新显示
            self.graphics.update_display()
        except pygame.error as e:
            if "display Surface quit" in str(e):
                print("游戏窗口已关闭，退出游戏")
                self.running = False
            else:
                raise
    
    def draw_extra_info(self):
        """绘制额外信息"""
        # 使用系统字体解决中文显示问题
        try:
            font = pygame.font.SysFont('simhei', 20)  # 尝试使用黑体
        except:
            font = pygame.font.Font(None, 20)  # 如果失败，使用默认字体
        
        # 显示控制提示
        if self.use_ai:
            controls_text = [
                f"当前模式: AI控制",
                f"当前个体: {self.current_individual_index}/{len(self.loaded_population)-1 if self.loaded_population else 0}",
                f"个体适应度: {self._get_current_individual_fitness():.3f}",
                f"代数: {self.loaded_generation}",
                "←/→: 切换个体",
                "M: 切换到手动控制",
                "R: 重置 ESC: 退出"
            ]
        else:
            controls_text = [
                f"当前模式: 手动控制",
                f"个体编号: {self.individual.entity_id}",
                f"个体适应度: {self.fitness:.3f}",
                "W/↑: 上移 S/↓: 下移",
                "A/←: 左移 D/→: 右移",
                "M: 切换到AI控制",
                "R: 重置 ESC: 退出"
            ]
        
        # 将控制信息显示在屏幕右侧
        y_offset = 10
        for text in controls_text:
            text_surface = font.render(text, True, (0, 0, 0))  # 黑色文本
            # 计算文本宽度，使其右对齐到屏幕边缘
            text_rect = text_surface.get_rect()
            text_rect.right = setting.SCREEN_WIDTH - 10  # 距离右边缘10像素
            text_rect.top = y_offset
            self.graphics.screen.blit(text_surface, text_rect)
            y_offset += 25
        
        # 显示个体状态
        status_texts = [
            f"位置: ({self.individual.pos_x:.1f}, {self.individual.pos_y:.1f})",
            f"速度: ({self.individual.vel_x:.2f}, {self.individual.vel_y:.2f})",
            f"到达目标数: {self.individual.goals_reached}",
            f"当前适应度: {self.fitness:.3f}"
        ]
        
        # 将状态信息也显示在屏幕右侧
        y_offset = setting.SCREEN_HEIGHT - 100
        for text in status_texts:
            text_surface = font.render(text, True, (0, 0, 0))  # 黑色文本
            # 计算文本宽度，使其右对齐到屏幕边缘
            text_rect = text_surface.get_rect()
            text_rect.right = setting.SCREEN_WIDTH - 10  # 距离右边缘10像素
            text_rect.top = y_offset
            self.graphics.screen.blit(text_surface, text_rect)
            y_offset += 25
    
    def run(self):
        """运行游戏主循环"""
        while self.running:
            # 处理输入
            self.handle_input()
            
            # 更新游戏状态
            self.update()
            
            # 绘制游戏画面
            self.draw()
            
            # 控制帧率
            pygame.time.Clock().tick(60)

# 主程序入口
if __name__ == "__main__":
    
    # 检查命令行参数
    use_ai = "--ai" in sys.argv
    
    render = True  # 设置是否渲染图形界面
    
    # 创建手动控制游戏实例
    game = ManualGame(render=render, use_ai=use_ai)
    game.run()