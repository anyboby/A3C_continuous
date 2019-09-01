import constants as Constants
from master_network import MasterNetwork
import network_shares as Netshare
import cv2

import gym
import numpy as np
import threading



class Agent(threading.Thread):
    env_lock = threading.Lock()
    
    def __init__(self, name, globalAC, cvshow = False):
        threading.Thread.__init__(self)
        self.env = gym.make(Constants.GAME).unwrapped
        self.name = name
        self.AC = MasterNetwork(name, globalAC)
        self.cvshow = cvshow

    def work(self):
        #global GLOBAL_RUNNING_R, GLOBAL_EP

        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not Netshare.COORD.should_stop() and Constants.GLOBAL_EP < Constants.MAX_GLOBAL_EP:

            #lock in case gym backend not threadsafe
            with Agent.env_lock:
                start_img = self.env.reset()

            ##### manual state space block #####
            if Constants.manual_dims:
                start_img = self._process_img(start_img)
                s = np.zeros((Constants.STATE_WIDTH, Constants.STATE_HEIGHT, Constants.STATE_STACK))
                for i in range(Constants.STATE_STACK):
                    s[:,:, i] = start_img
            ####################################
            else: 
                s = start_img

            ep_r = 0
            ep_r_max = 0
            for ep_t in range(Constants.MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)

                with Agent.env_lock:
                    #print("action: " + str(a),flush=True)
                    for Constants.SKIP_STEPS:    
                        img_rgb, r, done, info = self.env.step(a)
                
                done = True if ep_t == Constants.MAX_EP_STEP - 1 else False

                ##### specific block for manual state space      ######
                if Constants.manual_dims:
                    s_ = s
                    if not done:
                        img = self._process_img(img_rgb)

                        for i in range(Constants.STATE_STACK-1):
                            s_[:,:,i] = s_[:,:,i+1] 
                        s_[:,:,Constants.STATE_STACK-1] = img #append new state to stack
                    else: 
                        s_ = None
                #######################################################
                # else take env state as is
                else:
                    s_ = img_rgb

                #######   adding in early termination  ############
                ep_r += r  #add reward for the last step to total Rewards
                
                if ep_r > ep_r_max:
                    ep_r_max  = ep_r

                if ep_r_max - ep_r > Constants.EARLY_TERMINATION:
                    done = True
                    s_ = None
                ####################################################

                buffer_s.append([s])
                buffer_a.append([a])
                buffer_r.append((r+8)/8)    # normalize
                if total_step % Constants.UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = Netshare.SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + Constants.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                
                #skip frames for speed
                if self.cvshow:
                    cv2.imshow("image", s)
                    cv2.waitKey(Constants.WAITKEY)

                
                # assume new state
                s = s_
                total_step += 1
                if done:
                    if len(Constants.GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        Constants.GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        Constants.GLOBAL_RUNNING_R.append(0.9 * Constants.GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", Constants.GLOBAL_EP,
                        "| Ep_r: %i" % Constants.GLOBAL_RUNNING_R[-1],
                          )
                    Constants.GLOBAL_EP += 1
                    break


    def _process_img(self, img):
            """
            preprocess a state which is given as : (96, 96, 3)
            """
            img = self._rgb2gray(img, True) # squash to 96 x 96
            #img = self._zero_center(img)
            img = self._crop(img)
            #img = cv2.resize(img, (40,40))
            #ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
            return img


    def _rgb2gray(self, rgb, norm):

        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray.astype("float32") / 128 - 1 

        return gray 

    def _zero_center(self, img):
        #return np.divide(frame - 127, 127.0)
        return img - 127.0

    def _crop(self, img, length=12):
        """
        crop 96 by 96 to 84 by 84
        """
        h = len(img)
        w = len(img[0])
        return img[0:h - length, (int)(length/2):w - (int)(length/2)]