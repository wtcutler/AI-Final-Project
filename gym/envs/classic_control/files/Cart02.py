import numpy as np
import gym
import random

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

NUM_BUCKETS = (19, 15)  # (position, velocity)
NUM_ACTIONS = env.action_space.n #0 = LEFT; 1 = REST; 2=RIGHT
EPISODES = 15000

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
print(observation_space)
print(action_space)

#(min,max) for the position and velocity
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
print(STATE_BOUNDS)

q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
# q_table= np.random.uniform(low=-1, high=1,
#                       size=(19, 15,
#                             env.action_space.n))

def state_to_bucket(state):
    bucket_indice = [0,0]
    bucket_indice[0] = int(np.round((state[0] - env.observation_space.low[0])*10))
    bucket_indice[1] = int(np.round((state[1] - env.observation_space.low[1])*100))
    return tuple(bucket_indice)

def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def simulate(starting_epsilon):

    explore_rate = starting_epsilon
    learning_rate = .2
    discount_factor = .9
    min_eps = 0

    reward_list = []
    ave_reward_list = []
    reduction = (explore_rate - min_eps) / EPISODES

    # Run Q learning algorithm
    for i in range(EPISODES):
        done = False
        # Reset the environment
        obv = env.reset()
        tot_reward, reward = 0, 0

        # the initial state
        state_0 = state_to_bucket(obv)

        while done != True:

            if i == EPISODES -1:
                env.render()
            # Select an action
            action = select_action(state_0, explore_rate)


            # Execute the action
            obv, reward, done, _ = env.step(action)
            # Observe the result
            state = state_to_bucket(obv)
            # Update the Q based on the result
            best_q = np.amax(q_table[state])

            if done and obv[0] >= .5:
                q_table[state_0 + (action,)] = reward
            else:
                q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state
            #env.step(env.action_space.sample()) # take a random action

            tot_reward += reward

        # Decay epsilon
        if explore_rate > min_eps:
            explore_rate -= reduction

        reward_list.append(tot_reward)
        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))


if __name__ == "__main__":
    simulate(starting_epsilon = .2)
