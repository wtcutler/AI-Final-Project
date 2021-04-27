import gym

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

#actions:  0 = LEFT; 1 = REST; 2=RIGHT
#observations: (position, velocity)

for _ in range(1000):
    env.render()

    action = 2 # go to the right
    # Execute the action
    obv, reward, done, _ = env.step(action)
    print(obv)


env.close()
