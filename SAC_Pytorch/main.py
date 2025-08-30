import gymnasium as gym
import pybullet_envs_gymnasium
import numpy as np
from Agent import Agent


if __name__ == '__main__':

    env = gym.make( "InvertedPendulumBulletEnv-v0", render_mode='human' ) 

    n_states = sum( env.observation_space.shape )
    n_actions =  sum( env.action_space.shape )  

    agent = Agent( input_dims=n_states, env=env, n_actions=n_actions )
    
    load_checkpoint = False
    if load_checkpoint:   # Add logic to be able to load a checkpoint and continue training
        agent.load_models()
        # env.render(mode='human')

    n_games = 250

    best_score = -float( 'inf ')
    score_history = []

    observation, _ = env.reset( seed=0 )

    for i in range( n_games ):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            # print( f'observation: {observation}')

            action = agent.choose_action( observation )
            new_observation, reward,  terminated, truncated, _ = env.step( action )
            done = terminated or truncated
            agent.remember( observation, action, new_observation, reward, done )

            if not load_checkpoint:
                agent.learn()

            score += reward
            observation = new_observation

        score_history.append( score )
        avg_score = np.mean( score_history[ -100: ] )

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print( 'epsiode', i, 'score %.2f' % score, 'average score %.2f' % avg_score )