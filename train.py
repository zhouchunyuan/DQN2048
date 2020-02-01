import game2048 as game
import numpy as np
import random
import keyboard
import time

import tensorflow
import tensorflow.keras as keras # current tensorflow.keras has memory leakage in model.predict
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal,RandomUniform
from tensorflow.keras import backend as K

from collections import deque

GAME = '2048' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 3000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 1024 # size of minibatch
LEARNING_RATE = 1e-6 # learning rate

weightfile = 'weight2048.h5'

def createNetwork():
    model = tensorflow.keras.Sequential([
        #tensorflow.keras.layers.Conv2D(4, (1, 1),padding='same', activation='relu', input_shape=(4, 4, 1),kernel_initializer = RandomUniform()),
        keras.layers.Flatten(input_shape = (4,4,1)),
        #keras.layers.Dense(256,activation='relu')
        keras.layers.Dense(1024,activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(512,activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(256, activation='relu',kernel_initializer = RandomUniform()),
        keras.layers.Dense(4, activation='relu',kernel_initializer = RandomUniform())
    ])
    model.compile(loss='mse',
                  optimizer=Adam(lr=LEARNING_RATE))
    return model

def trainNetwork(model):
    game_state = game.GameState()
    # store the previous observations in replay memory
    D = deque()

    # get the first state by moving down
    actions = np.zeros(ACTIONS)
    actions[1] = 1
    s_t, r_0, terminal = game_state.frame_step_quick(actions)# status, reward, terminal
    s_t = np.array(s_t).astype('float32')

    try:
        model.load_weights(weightfile)
        print("Successfully loaded:")
    except:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    human_training = True 
    last_state =np.ndarray((4,4)).astype('float32')
    state_changed = True # for test if state has changed over step 
    while(1):
        # choose an action epsilon greedily
        state = s_t.reshape(1,4,4,1)
        readout_t = model.predict(state)

        #####  key control input ######
        try:
            if keyboard.is_pressed('q'):
                model.save(weightfile)
                print('model saved, You Pressed q Key!')
                break  # finishing the loop
            if keyboard.is_pressed('m'):
                human_training = True
            if keyboard.is_pressed('a'):
                human_training = False
                print('a pressed')
        except:
            print('key board input error')
            break
        ################# make decicsion ##########
        a_t = np.zeros([ACTIONS])
        if not human_training:
            ### perform random actions if ...
            if random.random() <= epsilon or (not state_changed):
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else: 
            try:  # used try so that if user pressed other than the given key error will not be shown
                action_index = -1

                while ( keyboard.is_pressed('i') or 
                        keyboard.is_pressed('j') or
                        keyboard.is_pressed('k') or
                        keyboard.is_pressed('l')):
                    pass # wait key release
                while action_index == -1:
                    keycode = keyboard.read_key()
                    if keycode =='i':  #up
                        action_index = 0
                    if keycode =='k':  #down
                        action_index = 1
                    if keycode =='j':  #left
                        action_index = 2
                    if keycode =='l':  #right
                        action_index = 3
                    
                a_t[action_index] = 1
            except:
                break 

        ############################
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        # run the selected action and observe next state and reward
        s_t1, r_t, terminal = game_state.frame_step_quick(a_t)
        s_t1 = np.array(s_t1).astype('float32')
        
        # check not available actions (state unchanged)
        state_changed = not np.array_equal(last_state,s_t1)
        last_state =np.array( s_t1,copy=True )

        #print(s_t)
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH,4,4,1)
            target = model.predict(state_batch)
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH,4,4,1)
            readout_j1_batch = model.predict(next_state_batch)
            for i in range(0, len(minibatch)):
                termi = minibatch[i][4]
                # if terminal, only equals reward
                if termi:
                    target[i][np.argmax(a_batch[i])] = r_batch[i]
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    target[i][np.argmax(a_batch[i])] = r_batch[i] + GAMMA * (np.max(readout_j1_batch[i]))

            # perform gradient step
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH,4,4,1)
            model.train_on_batch(state_batch,target)
            
        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            model.save(weightfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"    
        acts={0:'up',1:'dn',2:'lf',3:'rt'}
        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", acts[action_index], "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))

        if human_training:
            time.sleep(0.1)
        if terminal:
            game_state.reset()
            

            
def playGame():
    model = createNetwork()
    trainNetwork(model)

def main():
    playGame()

if __name__ == "__main__":
    main()
