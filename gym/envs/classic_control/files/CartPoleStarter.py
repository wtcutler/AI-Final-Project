#CartPoleStarter

import gym

## Defining the simulation related constants
NUM_EPISODES = 1000

def simulate():
    ## Initialize the "Cart-Pole" environment
    env = gym.make('CartPole-v0')

    for episode in range(NUM_EPISODES):

        done = False
        # Reset the environment
        obv = env.reset()

        initial_action = 0 #initial action is to move the cart to the left (arbitrary)

        total_reward = 0
        steps = 0

        while done != True:
            # render the simulation
            env.render()

            # Select an action
            if episode == 0:
                action = initial_action

            # Execute the action
            obv, reward, done, _ = env.step(action)
            print(obv)


            total_reward += reward
            steps +=1

            #TODO:
            #change the action here based on the obv
            #make action = 0 (left) or action = 1 (right) based on if-statements

        print("Episode", episode, "finished after", steps, "time steps with total reward", total_reward)

if __name__ == "__main__":
    simulate()

