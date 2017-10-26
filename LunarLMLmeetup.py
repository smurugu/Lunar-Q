from IPython import get_ipython   #
get_ipython().magic('reset -sf')  # clear all variables i.e. spyder console
import numpy as np 
import matplotlib.pyplot as plt
import gym

def get_features(ob):
    
    features=np.zeros(4)
    features[0]=ob[0]*.5+ob[2] # next x translation
    features[1]=ob[1]*.5+ob[3] # next y translation
    features[2]=(features[0]-ob[4])*.5-ob[5] # angle alignment want vertical
    features[3]=ob[6]*2+ob[7] # discretize legs

    return features

def get_weights(w,error,alpha,features):
    for i in range(w.size):
         w[i]=.99*w[i]+(1.0/alpha)*error*features[i]+(np.random.rand(1)-.5)*.01 # small weight decay
    
    return w

def get_action(q_val,index,epsilon):
    
    action=0
    q_max=0
    explore=np.random.sample() # should we explore or exploit
    if explore>epsilon: # if yes explore
        action=np.round(np.random.sample()*3) # uniform rand var 0 or 1 action
    else: # else exploit the q-values for max in this state, take action
        
        action=np.argmax(q_val[index])
        q_max=q_val[index][action]
       
    return int(action),q_max

def get_index(features): 
    index=np.zeros(len(features),dtype=np.int8)
    discrete=np.zeros([3,4])
    discrete[0]=[-.4,-.15, .15,  .4] 
    discrete[1]=[-.15,-.05,.05, .15]
    discrete[2]=[-.15,-.05,.05, .15]
    for i in range(len(features)-1): #cleaner than a switch or massive if-else
        j=0
        index[i]=4
        while j<(len(discrete[i][:])):
            
            if features[i]<discrete[i][j]:
                index[i]=j
                j=5
            j+=1    
    
    index[3]=features[3] # 0:no legs, 1:R leg 2:L leg 3:both legs
             
    return tuple(index)


if __name__ == '__main__':

    np.random.seed(0)   #04
    env = gym.make('LunarLander-v2')
    env.seed(0) # random seed , checked a half dozen to make sure not a fluke, epsilon is nearly 1
    episode_count = 150000

    alpha=0  # learning rate... will be set to 1/n
    gamma=.5  # discount factor 1=infinite memory depth
    epsilon=.99# for exploration vs exploitation

    q_val=0.0001*np.random.rand(5,5,5,4,4) # init q values small random #'s
    qhld=np.zeros([5,5,5,4,4]) # hold the last q values

    w=np.zeros(4) # 4 weights for the 4 features we chose from observation space (i.e. pos, vel,ang,anb-vel)
    whold=np.zeros(4) # hold the last weights


    rewardhld=np.zeros(30) #reward circular buffer to hold last 30 values
    rwdcnt=0 # circular buffer counter
    pltr=[]  # avg reward vector for plotting
    
    ###### episode loop  
    for i in range(episode_count):
        t=0           # reset the timer counter for printing purposes
        reward_cnt=0  # reset reward sum for the episode
        alpha=1+i*.5  # reset the learning parameter - seems to do better than letting it converge to 0
        epsilon=epsilon+0.0016 # lower exlploration variable as we learn more

        ob = env.reset()              # reset environment get intial ob
        features=get_features(ob)     # get features from current state
        index_old=get_index(features) # discretize the features and return index
    
        whold[0:4]=w[0:4] # save the last episode's weights
        qhld[:,:,:,:,:]=q_val[:,:,:,:,:]
        
        ##### rollout loop
        while True:    
            alpha=alpha+1 # changing learning parameter from .1 to 1/n
            t=t+1         # inc the episode time
            
            action,q_max = get_action(q_val,index_old,epsilon) # take a new action 
            ob, reward, done, _ = env.step(action) # get the new state/reward based on new action
            features=get_features(ob)
            index_new=get_index(features) # get index of current state
            if done and ob[6]==1 and ob[7]==1 and np.abs(ob[3])<.1 and np.abs(features[2])<.1:
               reward=reward+500 
            
            _,q_max = get_action(q_val,index_new,epsilon) # get new argmax of Q with new state           
            error=(reward+gamma*q_max)-q_val[index_old][action]# calc error for approximate update           
            w=get_weights(w,error,alpha,features) # update the weights    
            q_val[index_old][action]=np.inner(w,features) # update Q with new weights
            
            reward_cnt=reward_cnt+reward
            
            if done:
                rewardhld[rwdcnt]=reward_cnt
                rwdcnt+=1
                if rwdcnt>29:
                    rwdcnt=0
               
                if (reward_cnt<600 and i >100):
                    w[0:4]=whold[0:4]
                    #q_val[:,:,:,:,:]= qhld[:,:,:,:,:]                    
                else:
                    w[0:4]=w[0:4]+whold[0:4]/2000.0 # momentum/bias term
                    #q_val[:,:,:,:,:]= q_val[:,:,:,:,:]+qhld[:,:,:,:,:]/2000             
                print ("epi_t: {} roll_T={} roll_reward {:.0f} avg {:.0f}".format(i,t,reward_cnt,np.mean(rewardhld)))
                pltr.append(np.mean(rewardhld))
                break
                
            index_old=index_new
        ##### end of rollout loop ############
          
        if np.mean(rewardhld)>660:
            break # break out of episode loop and render
    ###### end of episode loop ###########
  
   ##### results   
    plt.plot(pltr) 
    plt.title('avg last 30 episodes')
    plt.ylabel('reward') 
    plt.xlabel('# episodes')  
    for i in range(20):
        ob = env.reset()
        while True: 
        
           features=get_features(ob)
           index_old=get_index(features)     
           action,q_max = get_action(q_val,index_old,epsilon) # take a new action 
           ob, reward, done, _ = env.step(action) # get the new state/reward based on new action
           env.render()
           if done:
               break
   ###################################    
    env.close()


print ("Successfully ran.")
    
