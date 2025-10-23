import pygame
import os
import setting

class GameGraphics:
    """
    游戏图形处理类，负责所有图形绘制和窗口显示
    """
    
    def __init__(self):
        # 初始化pygame
        pygame.init()
        
        # 创建游戏窗口
        self.screen = pygame.display.set_mode((setting.SCREEN_WIDTH, setting.SCREEN_HEIGHT))
        pygame.display.set_caption(setting.CAPTION)
        
        # 创建时钟对象，用于控制帧率
        self.clock = pygame.time.Clock()
        
        # 创建字体 - 支持中文显示
        if os.name == 'nt':  # Windows系统
            self.font = pygame.font.SysFont('simhei', 18)  # 使用黑体
            self.small_font = pygame.font.SysFont('simhei', 14)  # 小号字体
            self.large_font = pygame.font.SysFont('simhei', 24)  # 大号字体
        else:
            self.font = pygame.font.Font(None, 18)
            self.small_font = pygame.font.Font(None, 14)
            self.large_font = pygame.font.Font(None, 24)
    
    def draw_game_objects(self, individuals, goals=None, obstacles=None):
        """绘制所有游戏对象"""
        # 清屏
        self.screen.fill(setting.SKY_BLUE)
        
        # 绘制目标
        if goals:
            for goal in goals:
                goal.draw(self.screen)
        
        # 绘制障碍物
        if obstacles:
            for obstacle in obstacles:
                obstacle.draw(self.screen)
        
        # 绘制种群个体
        for individual in individuals:
            individual.draw(self.screen, goals)
    
    def draw_ui(self, individuals, individual_fitness, generation, frame_count):
        """绘制用户界面（绿色，绘制在最上层）"""
        
        # 绘制存活种群个体数量
        alive_individuals = len([individual for individual in individuals if individual.active])
        individual_text = self.font.render(f"存活种群个体: {alive_individuals}/{setting.POPULATION_SIZE}", True, setting.GREEN)
        self.screen.blit(individual_text, (10, 40))
        
        # 绘制游戏帧数
        frame_text = self.font.render(f"游戏帧数: {frame_count}", True, setting.GREEN)
        self.screen.blit(frame_text, (10, 70))
        
        # 绘制代数信息
        generation_text = self.font.render(f"代数: {generation}", True, setting.GREEN)
        self.screen.blit(generation_text, (10, 100))
        
        # 绘制当前最高适应度
        if individual_fitness:
            max_fitness = max(individual_fitness.values()) if individual_fitness else 0
            fitness_text = self.font.render(f"最高适应度: {max_fitness:.3f}", True, setting.GREEN)
            self.screen.blit(fitness_text, (10, 130))
    
    def update_display(self):
        """更新显示"""
        pygame.display.flip()
        # 控制帧率
        self.clock.tick(setting.FPS)