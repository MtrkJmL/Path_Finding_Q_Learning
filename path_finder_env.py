import numpy as np
import pygame
import time
from collections import deque

class PathFinderEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.action_space = 4  # up, down, left, right
        self.observation_space = grid_size * grid_size
        
        # Initialize Pygame
        pygame.init()
        self.cell_size = 50
        self.screen = pygame.display.set_mode((grid_size * self.cell_size, grid_size * self.cell_size))
        pygame.display.set_caption("RL Path Finder")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        self.reset()
    
    def _is_valid_path(self, obstacles):
        """Check if there's a valid path from start to goal using BFS"""
        visited = set()
        queue = deque([self.agent_pos])
        
        while queue:
            current = queue.popleft()
            if current == self.goal_pos:
                return True
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                next_pos = (next_x, next_y)
                
                if (0 <= next_x < self.grid_size and 
                    0 <= next_y < self.grid_size and 
                    next_pos not in obstacles and 
                    next_pos not in visited):
                    queue.append(next_pos)
        
        return False
    
    def _generate_valid_obstacles(self):
        """Generate obstacles while ensuring a valid path exists"""
        obstacles = set()
        max_obstacles = int(self.grid_size * self.grid_size * 0.2)
        
        # Try to add obstacles one by one
        while len(obstacles) < max_obstacles:
            # Try to add a new obstacle
            new_obstacle = (np.random.randint(0, self.grid_size), 
                          np.random.randint(0, self.grid_size))
            
            # Skip if the position is start, goal, or already an obstacle
            if (new_obstacle == self.agent_pos or 
                new_obstacle == self.goal_pos or 
                new_obstacle in obstacles):
                continue
            
            # Temporarily add the obstacle and check if path still exists
            temp_obstacles = obstacles.copy()
            temp_obstacles.add(new_obstacle)
            
            if self._is_valid_path(temp_obstacles):
                obstacles.add(new_obstacle)
            
            # If we can't add more obstacles after many attempts, break
            if len(obstacles) == 0 and len(temp_obstacles) > 0:
                break
        
        return obstacles
    
    def reset(self):
        # Start at top-left corner
        self.agent_pos = (0, 0)
        # Goal at bottom-right corner
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        
        # Generate obstacles ensuring a valid path exists
        self.obstacles = self._generate_valid_obstacles()
        
        return self._get_state()
    
    def _get_state(self):
        # Convert position to state index
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def step(self, action):
        # 0: up, 1: right, 2: down, 3: left
        x, y = self.agent_pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # right
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 3:  # left
            x = max(0, x - 1)
        
        new_pos = (x, y)
        
        # Check if new position is valid
        if new_pos in self.obstacles:
            reward = -10
            done = False
        else:
            self.agent_pos = new_pos
            if self.agent_pos == self.goal_pos:
                reward = 100
                done = True
            else:
                reward = -1
                done = False
        
        return self._get_state(), reward, done
    
    def render(self):
        self.screen.fill(self.WHITE)
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            rect = pygame.Rect(obstacle[0] * self.cell_size, obstacle[1] * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.BLACK, rect)
        
        # Draw agent
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, 
                               self.agent_pos[1] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.BLUE, agent_rect)
        
        # Draw goal
        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size,
                              self.goal_pos[1] * self.cell_size,
                              self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.GREEN, goal_rect)
        
        pygame.display.flip()
        time.sleep(0.01)  # Slow down visualization
    
    def close(self):
        pygame.quit() 