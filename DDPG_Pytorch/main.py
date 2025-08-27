from Agent import Agent
import numpy as np
import gymnasium as gym


# Initialise the environment
env = gym.make( "LunarLander-v3", continuous=True, render_mode="human" )

agent = Agent( alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, 
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2 )

observation, _ = env.reset( seed=0 )

score_history = []

for i in range( 1000 ):  # number of episodes (episode = full training sequence until terminated or truncated)

    done = False
    score = 0
    observation, info = env.reset()

    while not done:
        action = agent.choose_action( observation )
        new_state, reward,  terminated, truncated, info = env.step( action )
        done = terminated or truncated
        agent.remember( observation, action, reward, new_state, done )
        agent.learn()
        score += reward
        observation = new_state

    score_history.append( score )
    print( f"episode %d, score %.2f, 100 game average %.2f" % ( i, score, np.mean( score_history[ -100: ] ) ) )

    if i % 50 == 0:
        agent.save_models()