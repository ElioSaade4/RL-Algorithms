import gymnasium as gym
import numpy as np
from Agent import Agent
 
if __name__ == '__main__':

    # Initialise the environment
    env = gym.make( "LunarLander-v3", continuous=True, render_mode=None )

    agent = Agent( alpha=0.001, beta=0.001, input_dims=env.observation_space.shape[0], tau=0.005, env=env, 
                   batch_size=100, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0] )
    
    n_games = 1000
    filename = 'plots/' + 'LunarLanderContinuous_' + str( n_games ) + '_games.png'

    best_score = -float( 'inf ')
    score_history = []

    observation, _ = env.reset( seed=0 )

    # agent.load_models()  if you want to load a trained model

    for i in range( n_games ):
        observation, info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action( observation )
            new_observation, reward,  terminated, truncated, info = env.step( action )
            done = terminated or truncated
            agent.remember( observation, action, new_observation, reward, done )
            agent.learn()
            score += reward
            observation = new_observation

        score_history.append( score )
        avg_score = np.mean( score_history[ -100: ] )

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print( 'epsiode', i, 'score %.2f' % score, 'average score %.2f' % avg_score )