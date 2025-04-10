from path_finder_env import PathFinderEnv
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def train(episodes=1000, grid_size=10, save_interval=100, continue_training=True):
    env = PathFinderEnv(grid_size=grid_size)
    agent = QLearningAgent(env.observation_space, env.action_space)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Try to load existing model if continue_training is True
    if continue_training:
        model_files = [f for f in os.listdir('models') if f.startswith('q_table_')]
        if model_files:
            # Separate numbered models and final model
            numbered_models = [f for f in model_files if not f.endswith('final.npy')]
            final_model = 'models/q_table_final.npy' if 'q_table_final.npy' in model_files else None
            
            # Try to load the latest numbered model first, fall back to final model
            if numbered_models:
                latest_model = max(numbered_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                model_path = f"models/{latest_model}"
            elif final_model:
                model_path = final_model
            else:
                model_path = None
                
            if model_path:
                print(f"Loading existing model: {model_path}")
                agent.load_q_table(model_path)
            else:
                print("No valid models found, starting training from scratch")
        else:
            print("No existing models found, starting training from scratch")
    else:
        print("Starting new training from scratch")
    
    rewards_history = []
    steps_history = []
    running = True
    
    try:
        for episode in range(episodes):
            if not running:
                break
                
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                if not running:
                    break
                
                # Choose and perform action
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                # Learn from the experience
                agent.learn(state, action, reward, next_state, done)
                
                # Update state and counters
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render the environment
                env.render()
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                agent.save_q_table(f"models/q_table_episode_{episode+1}.npy")
                print(f"Model saved at episode {episode + 1}")
            
            # Print episode summary
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1}")
                print(f"Total Reward: {total_reward}")
                print(f"Steps: {steps}")
                print(f"Exploration Rate: {agent.exploration_rate:.2f}")
                print("-" * 30)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model and training results
        agent.save_q_table("models/q_table_final.npy")
        
        # Plot training results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(steps_history)
        plt.title("Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        
        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.close()
        
        env.close()
        print("Training results and model saved successfully")

def load_and_test_model(grid_size=10):
    env = PathFinderEnv(grid_size=grid_size)
    agent = QLearningAgent(env.observation_space, env.action_space)
    
    # Try to load the most recent model
    model_files = [f for f in os.listdir('models') if f.startswith('q_table_')]
    if not model_files:
        print("No saved models found")
        return
    
    # Separate numbered models and final model
    numbered_models = [f for f in model_files if not f.endswith('final.npy')]
    final_model = 'models/q_table_final.npy' if 'q_table_final.npy' in model_files else None
    
    # Try to load the latest numbered model first, fall back to final model
    if numbered_models:
        latest_model = max(numbered_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model_path = f"models/{latest_model}"
    elif final_model:
        model_path = final_model
    else:
        print("No valid models found")
        return
    
    print(f"Loading model: {model_path}")
    agent.load_q_table(model_path)
    
    # Test the loaded model
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)  # This will now use the loaded Q-table
        state, reward, done = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)  # Slow down for visualization
    
    print(f"Test run completed with total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    import pygame  # Import here to avoid circular imports
    # To train from scratch:
    # train(episodes=1000, grid_size=10, continue_training=False)
    
    # To continue training from existing model:
    train(episodes=1000, grid_size=10, continue_training=True)
    
    # To test a model:
    # load_and_test_model(grid_size=10) 