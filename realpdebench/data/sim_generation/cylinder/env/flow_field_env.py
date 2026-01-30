from xmlrpc.client import ServerProxy
import subprocess
import json
import time
import random
import numpy as np
import argparse
from gym.spaces import Box
import os
import signal
from multiprocessing import Process, Queue

def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
def start_xvfb(display_num):
    display = f':{display_num}'
    xvfb_process = subprocess.Popen(['Xvfb', display, '-screen', '0', '1024x768x16'])
    os.environ['DISPLAY'] = display
    time.sleep(2) 
    return xvfb_process

class env:
    def __init__(self, config=None, info='', local_port=None, network_port=None):
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.step_counter = 0
        self.state = [0, 0]
        self.action_interval = config.action_interval  # 10
        self.unwrapped=self
        self.unwrapped.spec = None
        self.observation_space = Box(low=-1e6,high=1e6,shape=[self.observation_dim])
        self.action_space = Box(low=-1,high=1,shape=[self.action_dim])
        self.local_port = local_port
        self.epoch_i = 0
        self.args = config

        while True:
            port = random.randint(2000, 40000)
            if not is_port_in_use(port):
                break
        
        display_num = network_port
                
        self.xvfb_process = start_xvfb(display_num)

        if local_port == None:
            self.server = subprocess.Popen(f'xvfb-run --server-num={display_num} -a /workspace/processing-4.3/processing-java --sketch={config.path_env}{config.name_env} --run {port} {info}', shell=True)
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")

    def step(self, action):
        result_ls = []
        
        action_json = {}

        if self.epoch_i==0:
            
            i = 0
            
            for key, value in self.args.dict_action.items():
                action_json[key] = float(action[value])
            
            res_str = self.proxy.connect.Step(json.dumps(action_json)) 
            [state, reward, done] = self.parseStep(res_str, self.epoch_i, i)
            self.reward, self.state, self.done = np.array(reward, dtype=np.float32), state, np.array(done, np.float32)
            result_ls.append(self.reward)
            self.epoch_i += 1
            self.action_interval = 10-1
            
        else:
            for i in range(self.action_interval):  
                                
                for key, value in self.args.dict_action.items():
                    action_json[key] = float(action[value])
                                
                res_str = self.proxy.connect.Step(json.dumps(action_json)) 
                [state, reward, done] = self.parseStep(res_str, self.epoch_i, i)
                self.reward, self.state, self.done = np.array(reward, dtype=np.float32), state, np.array(done, np.float32)
                result_ls.append(self.reward)
                                
                if self.done == True:
                    break
                
            self.epoch_i += 1
            self.action_interval = 10
            
        self.reward = np.average(result_ls)
        self.info = 0.0
        
        return self.state[:,0], self.reward, self.done, self.info

    def reset(self):
        self.done = False
        action_json = {"v1": 0, "v2": 0}
        res_str = self.proxy.connect.reset(json.dumps(action_json))
        self.state = self.parseReset(res_str)
        return self.state[:,0]

    def parseReset(self, info): 
        
        matrix_u = np.zeros((128, 128, 1))
        matrix_v = np.zeros((128, 128, 1))
        matrix_p = np.zeros((128, 128, 1))
        boundary_x = np.zeros((self.args.num_structure, 40, 1))
        boundary_y = np.zeros((self.args.num_structure, 40, 1))
        
        if 'angle' in self.args.all_features:
            angle_ls = np.zeros((1, self.args.num_structure))
        if 'CD' in self.args.all_features:
            cd = np.zeros((1, self.args.num_structure)) 
        if 'CL' in self.args.all_features:
            cl = np.zeros((1, self.args.num_structure))
        if 'move' in self.args.all_features:
            move = np.zeros((1, self.args.num_structure))
        
        all_info = json.loads(info)        
        
        state_u = json.loads(all_info['state'][0])
        state_v = json.loads(all_info['state'][1])
        state_p = json.loads(all_info['state'][2])
        done = all_info['done']
        bdy_x = json.loads(all_info['bdy'][0])
        bdy_y = json.loads(all_info['bdy'][1])
        
        if 'angle' in self.args.all_features:
            angle = json.loads(all_info['state'][self.args.dict_observe['angle']])
        if 'CD' in self.args.all_features:
            force_0 = json.loads(all_info['state'][self.args.dict_observe['CD']])
        if 'CL' in self.args.all_features:
            force_1 = json.loads(all_info['state'][self.args.dict_observe['CL']])
        if 'move' in self.args.all_features:
            move = json.loads(all_info['state'][self.args.dict_observe['move']])
        
        if 'angle' in self.args.all_features:   
            for i in range(self.args.num_structure):
                angle_ls[:,i] = angle[f'{i}'] 
        if 'CD' in self.args.all_features:
            for i in range(self.args.num_structure):
                cd[:,i] = force_0[f'C{i}'] 
        if 'CL' in self.args.all_features:
            for i in range(self.args.num_structure):
                cl[:,i] = force_1[f'C{i}']  
        
        matrix_uvp = np.concatenate((matrix_u,matrix_v,matrix_p), axis=-1).reshape(-1,1)
        cd = cd.reshape(-1,1)
        cl = cl.reshape(-1,1)
        angle_ls = angle_ls.reshape(-1,1)

        out_bdy = np.concatenate((boundary_x,boundary_y), axis=-1)    
        out_bdy = out_bdy.reshape(-1,1)
                        
        state = np.concatenate((matrix_uvp,out_bdy,cd,cl,angle_ls), axis=0) 
                
        assert state.shape[0]==self.args.observation_dim

        return state
        
    def parseStep(self, info, epoch_i, i): 
            
        state = []
        matrix_u = np.zeros((128, 128, 1))
        matrix_v = np.zeros((128, 128, 1))
        matrix_p = np.zeros((128, 128, 1))
        boundary_x = np.zeros((self.args.num_structure, 40, 1))
        boundary_y = np.zeros((self.args.num_structure, 40, 1))
        done = False
        
        if 'angle' in self.args.all_features:
            angle_ls = np.zeros((1, self.args.num_structure))
        if 'CD' in self.args.all_features:
            cd = np.zeros((1, self.args.num_structure)) 
        if 'CL' in self.args.all_features:
            cl = np.zeros((1, self.args.num_structure))
        if 'move' in self.args.all_features:
            move = np.zeros((1, self.args.num_structure))
        
        if epoch_i > self.args.store_step:
            if i%9 == 0 and i>0:
                all_info = json.loads(info)
                
                state_u = json.loads(all_info['state'][0])
                state_v = json.loads(all_info['state'][1])
                state_p = json.loads(all_info['state'][2])
                done = all_info['done']
                bdy_x = json.loads(all_info['bdy'][0])
                bdy_y = json.loads(all_info['bdy'][1])
                
                if 'angle' in self.args.all_features:
                    angle = json.loads(all_info['state'][self.args.dict_observe['angle']])
                if 'CD' in self.args.all_features:
                    force_0 = json.loads(all_info['state'][self.args.dict_observe['CD']])
                if 'CL' in self.args.all_features:
                    force_1 = json.loads(all_info['state'][self.args.dict_observe['CL']])
                if 'move' in self.args.all_features:
                    move = json.loads(all_info['state'][self.args.dict_observe['move']])
                
                
                for i in range(matrix_u.shape[0]):
                    for j in range(matrix_u.shape[1]):
                        key = f"{i},{j}"
                        matrix_u[i, j] = state_u[i*matrix_u.shape[1]+j][key]
                        matrix_v[i, j] = state_v[i*matrix_v.shape[1]+j][key]
                        matrix_p[i, j] = state_p[i*matrix_p.shape[1]+j][key]
                        
                if 'angle' in self.args.all_features:   
                    for i in range(self.args.num_structure):
                        angle_ls[:,i] = angle[f'{i}'] 
                if 'CD' in self.args.all_features:
                    for i in range(self.args.num_structure):
                        cd[:,i] = force_0[f'C{i}'] 
                if 'CL' in self.args.all_features:
                    for i in range(self.args.num_structure):
                        cl[:,i] = force_1[f'C{i}']  

                for i in range(boundary_x.shape[0]):
                    for j in range(boundary_x.shape[1]):
                        key = f"{i},{j}"
                        boundary_x[i, j] = bdy_x[i*boundary_x.shape[1]+j][key]
                        boundary_y[i, j] = bdy_y[i*boundary_y.shape[1]+j][key]
                        
        matrix_uvp = np.concatenate((matrix_u,matrix_v,matrix_p), axis=-1).reshape(-1,1)
        out_bdy = np.concatenate((boundary_x,boundary_y), axis=-1)    
        out_bdy = out_bdy.reshape(-1,1)
        cd = cd.reshape(-1,1)
        cl = cl.reshape(-1,1)
        angle_ls = angle_ls.reshape(-1,1)
        reward = 0.0
                        
        state = np.concatenate((matrix_uvp,out_bdy,cd,cl,angle_ls), axis=0) 
        
        assert self.args.observation_dim == state.shape[0]
        
        return state, reward, done

    def parseState(self, state):
        state = json.loads(json.loads(state)['state'][0])
        state['SparsePressure'] = list(map(float, state['SparsePressure'].split('_')))

        state_ls = [state['delta_y'], state['y_velocity'], state['eta'], state['delta_theta'], state['x_velocity'],
                    state['theta_velocity']] + state['SparsePressure']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        os.killpg(pid, signal.SIGTERM)  
        
    def close(self):
        if self.local_port == None:
            self.server.terminate()
