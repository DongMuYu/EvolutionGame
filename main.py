import pygame
from graphics import GameGraphics
from population_manager import PopulationManager
from course_manager import CourseManager
import setting

class Game:
    """
    训练环境主类，负责训练循环和逻辑控制
    """
    
    def __init__(self, render=True):
        # 训练状态
        self.running = True
        
        # 渲染标志
        self.render = render
        
        # 帧计数器
        self.frame_count = 0  # 统一的帧计数器
        
        # 创建图形处理对象（仅在需要渲染时创建）
        if self.render:
            self.graphics = GameGraphics()
        else:
            self.graphics = None
        
        # 创建种群管理器（专门负责种群管理）
        self.population_manager = PopulationManager()
        
        # 创建课程管理器（专门负责课程设置）
        self.course_manager = CourseManager()
        
        # 从种群管理器获取当前代数
        self.generation = self.population_manager.get_current_generation()
        
        # 个体、目标和障碍物
        self.individuals = []
        self.goals = []
        self.obstacles = []
        
        # 初始化个体、目标和障碍物
        self.individuals = self.course_manager.initialize_individuals(self.population_manager, frame_counter_ref=lambda: self.frame_count)
        self.goals = self.course_manager.initialize_goals()
        self.obstacles = self.course_manager.initialize_obstacles()
        
        # 适应度字典
        self.individual_fitness = {}
    
    def update(self):
        """更新训练状态"""

        if not self.running:
            return
        
        # 递增帧计数器
        self.frame_count += 1
        
        # 使用课程管理器的更新函数来更新课程元素（目标和障碍物）
        self.course_manager.update_course_elements(self.goals, self.obstacles)
        
        # 更新种群个体
        for individual in self.individuals:
            individual.update(individual, self.goals, self.obstacles)
        
        # 更新适应度
        self.individual_fitness = self.population_manager.update_fitness(self.individuals, self.goals, self.obstacles) 
        
        # 检查训练轮次结束条件
        self._check_game_over()
    
    def _check_game_over(self):
        """检查训练轮次结束条件"""

        # 检查是否所有种群个体都被消灭
        alive_individuals = [individual for individual in self.individuals if individual.active]  
        if not alive_individuals:
            # 只要所有个体都死亡，就立即开启下一轮训练
            # 进行遗传算法进化
            self.population_manager.evolve_population(self.individuals, self.individual_fitness, self.generation, self.goals)
            
            # 开启下一轮训练
            self.restart_game()
            return
        
        # 检查是否达到设定的帧数
        if self.frame_count >= setting.ROUND_DURATION_FRAMES:
            # 达到时间限制，开启下一轮训练
            # 进行遗传算法进化
            self.population_manager.evolve_population(self.individuals, self.individual_fitness, self.generation, self.goals)
            
            # 开启下一轮训练
            self.restart_game()
            return
    
    def draw(self):
        """绘制训练环境画面（仅在需要渲染时执行）"""
        
        # 仅在需要渲染时执行绘制操作
        if self.render and self.graphics:
            # 绘制训练对象
            self.graphics.draw_game_objects(self.individuals, self.goals, self.obstacles)  
            
            # # 绘制传感器射线（在游戏对象之后绘制，避免被覆盖）
            # for individual in self.individuals:
            #     if individual.alive:
            #         individual.draw_sensor_inputs(self.graphics.screen, self.goals, self.obstacles)
            
            # 绘制UI信息
            self.graphics.draw_ui(self.individuals, self.individual_fitness, self.generation, self.frame_count)  
            
            # 更新显示
            self.graphics.update_display()
    
    def restart_game(self):
        """重新开始训练轮次"""

        self.generation += 1
        self.frame_count = 0  # 重置帧计数器
        
        # 重新初始化个体、目标和障碍物
        self.individuals = self.course_manager.initialize_individuals(self.population_manager, frame_counter_ref=lambda: self.frame_count)
        self.goals = self.course_manager.initialize_goals()
        self.obstacles = self.course_manager.initialize_obstacles()
        
        print(f"开始第{self.generation}代")
    
    def _save_on_exit(self):
        """程序退出时保存当前模型"""
        try:
            # 获取当前种群和适应度
            fitness_scores = []
            for i in range(len(self.individuals)):
                if i in self.individual_fitness:
                    fitness_scores.append(self.individual_fitness[i])
                else:
                    fitness_scores.append(0)
            
            # 保存当前代数的模型
            self.population_manager.genetic_algorithm.save_population(
                self.population_manager.current_population, 
                self.generation, 
                fitness_scores
            )
            print(f"程序退出，已保存第{self.generation}代模型")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def run(self):
        """运行训练环境主循环"""
        
        while self.running:
            
            # 仅在渲染模式下处理pygame事件
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # 退出前保存模型
                        self._save_on_exit()
                        self.running = False
                        pygame.quit()
                        return
            
            # 更新训练状态
            self.update()
            
            # 绘制训练环境画面
            self.draw()

# 主程序入口
if __name__ == "__main__":
    render = False  # 设置是否渲染图形界面
    
    # 创建训练环境实例
    game = Game(render=render)
    game.run()

