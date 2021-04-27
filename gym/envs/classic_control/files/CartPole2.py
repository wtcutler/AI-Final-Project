import gym
import numpy as np
import random

## Defining the simulation related constants
NUM_EPISODES = 1000

def state_to_bucket(state, state_bounds, num_buckets):
    #state comes as a tuple of 4 values: (cart position, cart velocity, pole angle, pole velocity at tip)
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_bounds[i][1]:
            bucket_index = num_buckets[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (num_buckets[i]-1)*state_bounds[i][0]/bound_width
            scaling = (num_buckets[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)

    return tuple(bucket_indice)


def select_action(state, explore_rate, env, q_table):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def simulate(starting_epsilon):
    ## Initialize the "Cart-Pole" environment
    env = gym.make('CartPole-v0')

    num_buckets = (1, 1, 6, 3)  # (cart position, cart velocity, pole angle, pole velocity at tip)
    num_actions = env.action_space.n  # (left, right)

    # Bounds for each discrete state
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bounds[3] = [-0.5, 0.5]

    ## Creating a Q-Table for each state-action pair
    q_table = np.zeros(num_buckets + (num_actions,))

    ## Instantiating the learning related parameters
    learning_rate = .2
    explore_rate = starting_epsilon
    discount_factor = 0.99  # since the world is unchanging

    for episode in range(NUM_EPISODES):

        done = False
        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv, state_bounds, num_buckets)

        steps = 0
        total_reward = 0
        while done != True:
            if episode == NUM_EPISODES -1:
                env.render()

            # Select an action
            action = select_action(state_0, explore_rate, env, q_table)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # sum total reward for episode
            total_reward += reward

            # Observe the result
            state = state_to_bucket(obv, state_bounds, num_buckets)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            steps +=1

        # reduce the eploration rate for the next episode
        reduction = starting_epsilon / NUM_EPISODES
        explore_rate -= reduction
        print("Episode", episode, "finished after", steps, "time steps with total reward", total_reward)

if __name__ == "__main__":
    simulate(starting_epsilon=1.0)

