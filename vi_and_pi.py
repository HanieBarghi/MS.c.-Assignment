### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    new_value_function = value_function.copy()
    i = 0
    while i <= 100 or np.sum(np.sqrt(np.square(new_value_function - value_function))) > tol:
        i += 1
        value_function = new_value_function.copy()
        for state in range(nS):
            result = P[state][policy[state]]
            new_value_function[state] = np.array(result)[:, 2].mean()
            for num in range(len(result)):
                (probability, nextstate, reward, terminal) = result[num]
                new_value_function[state] += (gamma * probability * value_function[nextstate])
    ############################
    return new_value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):

    ############################
    # YOUR IMPLEMENTATION HERE #
    q_function = np.zeros([nS, nA])
    for state in range(nS):
        for action in range(nA):
            result = P[state][action]
            for num in range(len(result)):
                (probability, nextstate, reward, terminal) = result[num]
                q_function[state][action] = reward
                q_function[state][action] += (gamma * probability * value_from_policy[nextstate])
    new_policy = np.argmax(q_function, axis=1)
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    i = 0
    new_policy = policy.copy()
    while i <= 200 or np.sum(np.sqrt(np.square(new_policy - policy))) > tol:
        i += 1
        policy = new_policy
        V = policy_evaluation(P, nS, nA, policy)
        new_policy = policy_improvement(P, nS, nA, value_function, policy)
    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    idx = 1
    new_validation_function = value_function.copy()
    while idx <= 20 or np.sum(np.sqrt(np.square(new_validation_function - value_function))) > tol:
        idx += 1
        value_function = new_validation_function
        for state in range(nS):
            max_result = -10
            max_idx = 0
            for action in range(nA):
                result = P[state][action]
                temp = np.array(result)[:, 2].mean()
                # temp = result[0][2]
                for num in range(len(result)):
                    (probability, nextS, reward, terminal) = result[num]
                    temp += gamma * probability * value_function[nextS]
                    if max_result < temp:
                        max_result = temp
                        max_idx = action
            new_validation_function[state] = max_result
            policy[state] = max_idx
    ############################

    return value_function, policy


def render_single(env, policy, max_steps=100):

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
