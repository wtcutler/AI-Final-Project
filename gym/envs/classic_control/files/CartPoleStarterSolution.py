import gym
import numpy as np
import random

## Defining the simulation related constants
NUM_EPISODES = 1000

def simulate(starting_epsilon=0):
    ## Initialize the "Cart-Pole" environment
    env = gym.make('CartPole-v0')

    for episode in range(NUM_EPISODES):

        done = False
        # Reset the environment
        env.reset()

        initial_action = 0

        total_reward = 0
        steps = 0

        while done != True:
            env.render()

            # Select an action
            if episode == 0:
                action = initial_action

            # Execute the action
            obv, reward, done, _ = env.step(action)

            total_reward += reward
            steps +=1


            #here are some if statements -- this doesn't work well
            if obv[2] < 0:
                action = 1 #move right
            else:
                action = 0 #move left

        print("Episode", episode, "finished after", steps, "time steps with total reward", total_reward)

if __name__ == "__main__":
    simulate()

