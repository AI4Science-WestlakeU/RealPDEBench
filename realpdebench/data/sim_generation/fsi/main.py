import os
os.system('pkill -f processing')
os.system('pkill -f xvfb')
os.system('pkill -f Xvfb')
from env.flow_field_env import env
import argparse
import numpy as np
import gym
import random
import multiprocessing
from functools import partial

class Lilypad:
    def __init__(self, u_x, u_y, u_c, body_n, bd_n, xy_c, env, num_envs, init_parameters, args):
        self.u_x = u_x
        self.u_y = u_y
        self.u_c = u_c
        self.body_n = body_n
        self.bd_n = bd_n
        self.xy_c = xy_c
        self.u_end = 442368 
        self.bd_end = args.dict_state['bd']
        
        if 'angle' in args.dict_state:
            self.theta = args.dict_state['angle']
        if 'CD' in args.dict_state:
            self.f_0_cd = args.dict_state['CD']
        if 'CL' in args.dict_state:
            self.f_1_cd = args.dict_state['CL']
        if 'move' in args.dict_state:
            self.move = args.dict_state['move']
            
        self.env = env
        self.num_envs = num_envs
        self.init = init_parameters
        self.args = args
        
    def Lilypad_reset(self, w_t, index):
        
        init_state = self.env.reset() 
        
        init_parameters = self.init[index]
        
        all_return = {}
                
        w_t = np.concatenate((w_t,init_parameters.reshape(1,5)), axis=1)  
                        
        next_state, reward, done, info = self.env.step(w_t)
        
        init_u = next_state[:,:self.u_end].reshape(num_envs,self.u_x,self.u_y,self.u_c)  # (b,128,128,3)
        init_bd = next_state[:,self.u_end:self.bd_end].reshape(self.num_envs,self.body_n,self.bd_n,self.xy_c)  # (b,2,40,2)
        
        all_return['u'] = init_u
        all_return['bd'] = init_bd
        
        if 'angle' in self.args.dict_state:
            init_theta = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                init_theta[:self.num_envs,i] = next_state[:self.num_envs,self.theta+i]
            all_return['angle'] = init_theta
                
        if 'CD' in self.args.dict_state:
            init_f_0 = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                init_f_0[:self.num_envs,i] = next_state[:self.num_envs,self.f_0_cd+i]
            all_return['CD'] = init_f_0
            
        if 'CL' in self.args.dict_state:
            init_f_1 = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                init_f_1[:self.num_envs,i] = next_state[:self.num_envs,self.f_1_cd+i]
            all_return['CL'] = init_f_1
                
        if 'move' in self.args.dict_state:
            init_move = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                init_move[:self.num_envs,i] = next_state[:self.num_envs,self.move+i]
            all_return['move'] = init_move
                
        return all_return

    def Lilypad_step(self, w_t, sim_id, t, index):
                
        init_parameters = self.init[index]
        w_t = np.concatenate((w_t,init_parameters.reshape(1,5)), axis=1)
       
        all_return = {}
        
        # w_t: (b,2), sim_id: (b,1), t: (b,1)
        next_state, reward, done, info = self.env.step(w_t)  

        next_u = next_state[:,:self.u_end].reshape(num_envs,self.u_x,self.u_y,self.u_c)  # (b,128,128,3)
        next_bd = next_state[:,self.u_end:self.bd_end].reshape(self.num_envs,self.body_n,self.bd_n,self.xy_c)  # (b,2,40,2)
        all_return['u'] = next_u
        all_return['bd'] = next_bd
        
        if 'angle' in self.args.dict_state:
            next_theta = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                next_theta[:self.num_envs,i] = next_state[:self.num_envs,self.theta+i]
            all_return['angle'] = next_theta
                
        if 'CD' in self.args.dict_state:
            next_f_0 = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                next_f_0[:self.num_envs,i] = next_state[:self.num_envs,self.f_0_cd+i]
            all_return['CD'] = next_f_0
            
        if 'CL' in self.args.dict_state:
            next_f_1 = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                next_f_1[:self.num_envs,i] = next_state[:self.num_envs,self.f_1_cd+i]
            all_return['CL'] = next_f_1
            
        if 'move' in self.args.dict_state:
            next_move = np.zeros((self.num_envs,self.body_n))  # (b,2)
            for i in range(self.body_n):
                next_move[:self.num_envs,i] = next_state[:self.num_envs,self.move+i]
            all_return['move'] = next_move
        
        return all_return, done
    
def data_load(path, num_files):
    point_start_files = [f for f in os.listdir(f'{path}/point_path') if f.endswith('.npz')]
    angles_files = [f for f in os.listdir(f'{path}/angle') if f.endswith('.npz')]
    target_files = [f for f in os.listdir(f'{path}/target') if f.endswith('.npz')]

    sl_point_start_files = random.sample(point_start_files, num_files)
    sl_angles_files = random.sample(angles_files, num_files)
    sl_target_files = random.sample(target_files, num_files)

    def load_files(file_list, folder_name):
        arrays = []
        for file_name in file_list:
            file_path = os.path.join(f'{path}/{folder_name}', file_name)
            with np.load(file_path) as data:
                array = data['a']
                arrays.append(array)
        return np.array(arrays)

    point_start = load_files(sl_point_start_files, 'point_path')
    angles = load_files(sl_angles_files, 'angle')
    target = load_files(sl_target_files, 'target')
    
    print('Complete Data Load')
    
    return point_start, angles, target

def run_simulation(i, args, num_envs, init_parameters):
    s_state = []
    s_bd = []
    s_angle = []
    s_f0 = []
    s_f1 = []
    s_act = []
    s_Re = []
    s_epi_C = []
    s_massR_C = []
    s_dia0 = []
    s_dia1 = []
    s_move = []

    env = gym.vector.make('lilypad-v1', num_envs=num_envs, asynchronous=False, wrappers=None, config=args, network_port=i)
    u_x, u_y, u_c, body_n, bd_n, xy_c = 384, 384, 3, args.num_structure, 40, 2

    lilypad = Lilypad(u_x, u_y, u_c, body_n, bd_n, xy_c, env, num_envs, init_parameters, args)
    
    s_Re.append(int(init_parameters[i][0]))
    s_epi_C.append(init_parameters[i][1])
    s_massR_C.append(init_parameters[i][2])
    s_dia0.append(init_parameters[i][3])
    s_dia1.append(init_parameters[i][4])

    action_space, action_dim = env.single_action_space.shape[0], env.single_action_space.shape[0]
    obs_space, observation_dim = env.single_observation_space.shape[0], env.single_observation_space.shape[0]  # 2, 2

    w_t = np.zeros((num_envs,2))/args.action_interval 
    init_state  = lilypad.Lilypad_reset(w_t, i)
    done = [False] * num_envs  

    store_index = 1
    all_action = np.load('action.npy')

    while not any(done):
        
        w_t = np.zeros((num_envs,2))/args.action_interval  
            
        sim_id = np.ones((num_envs,1))
        t = np.ones((num_envs,1))
        step_return, done = lilypad.Lilypad_step(w_t, sim_id, t, i)
        
        if store_index > args.store_step:
            if not any(done):
                s_state.append(step_return['u'])
                s_bd.append(step_return['bd'])
                s_act.append(w_t)
                
                if 'angle' in args.dict_state:
                    s_angle.append(step_return['angle'])
                if 'CD' in args.dict_state:
                    s_f0.append(step_return['CD'])
                if 'CL' in args.dict_state:
                    s_f1.append(step_return['CL'])
                if 'move' in args.dict_state:
                    s_move.append(step_return['move'])
                        
        store_index += 1

    filename = f"{i:06}.npz"  
    np.savez(f'state/{filename}', np.array(s_state))
    np.savez(f'bd/{filename}', np.array(s_bd))
    np.savez(f'angle/{filename}', np.array(s_angle))
    np.savez(f'f_0/{filename}', np.array(s_f0))
    np.savez(f'f_1/{filename}', np.array(s_f1))
    np.savez(f'Re/{filename}', np.array(s_Re))
    np.savez(f'action/{filename}', np.array(s_act))
    np.savez(f'epi_C/{filename}', np.array(s_epi_C))
    np.savez(f'massR_C/{filename}', np.array(s_massR_C))
    np.savez(f'dia_0/{filename}', np.array(s_dia0))
    np.savez(f'dia_1/{filename}', np.array(s_dia1))

    env.close()

def generate_ranges(start, end, num_processes):
    step = (end - start) // num_processes
    ranges = [(i, min(i + step, end)) for i in range(start, end, step)]
    return ranges

def find_missing_numbers(folder_path, total_files=50000):
    files = os.listdir(folder_path)
    numbers = [int(file.split('.')[0]) for file in files if file.endswith('.npz')]
    missing_numbers = sorted(list(set(range(0, total_files)) - set(numbers)))
    return missing_numbers

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='inference 2d close loop control')
    parser.add_argument('--traj_start_i', default=0, type=int,
                        help='multiple machine generate, the start parameter')
    parser.add_argument('--traj_end', default=5, type=int,
                        help='multiple machine generate, the end parameter')
    parser.add_argument('--num_processes', default=1, type=int,
                        help='multiprocess number')
    parser.add_argument('--action_interval', default=10, type=int,  
                        help='gap of computation between two actions')
    parser.add_argument('--num_env', default=1, type=int,  
                        help='environment number of gym')
    
    parser.add_argument('--path_env', default='fluid_structure_interaction', type=str,
                        help='path of lilypad file')     
    parser.add_argument('--name_env', default='fluid_structure_interaction', type=str,
                        help='name of lilypad file')      
    
    parser.add_argument('--observation_dim', default=442534, type=int,
                        help='dimensition of state')      
    parser.add_argument('--action_dim', default=7, type=int,
                        help='dimensition of action')     
    parser.add_argument('--num_structure', default=2, type=int,
                        help='number of structure in lilypad')  
    parser.add_argument('--all_features', default=['angle', 'CD', 'CL'], type=str,
                        help='output of lilypad, CD, CL, angle, move')      
    parser.add_argument('--dict_action', default={"v1": 0, "v2": 1, "Re": 2, "epi_C": 3, "massR_C": 4, "dia_0": 5, "dia_1": 6}, type=dict,
                        help='dictionary of action')      
    parser.add_argument('--dict_observe', default={"angle": 3, "CD": 4, "CL": 5}, type=dict,
                        help='dictionary of observation from lilypad')     
    parser.add_argument('--dict_state', default={"bd": 442528, "angle": 442532, "CD": 442528, "CL": 442530}, type=dict,
                        help='dictionary of state when process the observation from lilypad and its index')  #    
    parser.add_argument('--folder_name', default=["state", "bd", "angle", "f_0", "f_1", "action","Re", "epi_C", "massR_C", "dia_0", "dia_1"], type=list,
                        help='the folder name stored')     
    parser.add_argument('--store_step', default=1000, type=int,
                        help='from which step can interact and store, related to the calculation length')  #    
    
    args = parser.parse_args()
    
    num_envs = args.num_env
    folders = args.folder_name
    traj_start = args.traj_start_i
    traj_end = args.traj_end
    num_processes = args.num_processes
    
    init_parameters = np.load('parameters.npy')
    
    for folder in folders:  
        if not os.path.exists(f'realpedbenchmark/simulation/{folder}'):
            os.makedirs(f'realpedbenchmark/simulation/{folder}')
            
    folder_path = f'realpedbenchmark/simulation/Re'
    
    while True:
        missing_numbers = find_missing_numbers(folder_path, (traj_end-traj_start))
        if len(missing_numbers) == 0:
            print("所有文件均已生成，无缺失文件编号。")
            break
    
        filtered_missing_numbers = [x for x in missing_numbers if traj_start <= x < traj_end]
            
        pool = multiprocessing.Pool(processes=num_processes)  
        func = partial(run_simulation, args=args, num_envs=num_envs, init_parameters=init_parameters)
        
        try:
            result = pool.map_async(func, filtered_missing_numbers)
            results = result.get()
        except Exception as e:
            print(f"执行任务时发生错误：{e}")
            pool.terminate()  
            pool.join()  
            os.system('pkill -f processing')
            os.system('pkill -f xvfb')
            os.system('pkill -f Xvfb')
        
        pool.close()
        pool.join()