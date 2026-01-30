import os
os.system('pkill -f processing')
os.system('pkill -f xvfb')
os.system('pkill -f Xvfb')
from env.flow_field_env import env
import argparse
import numpy as np
import gym
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
        self.u_end = args.dict_state['u']
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
        
        w_t = np.concatenate((w_t,init_parameters.reshape(1,-1)), axis=1)
                        
        next_state, reward, done, info = self.env.step(w_t)
        
        init_u = next_state[:,:self.u_end].reshape(num_envs,self.u_x,self.u_y,self.u_c)  # (b,128,128,3)
        init_bd = next_state[:,self.u_end:self.bd_end].reshape(self.num_envs,self.body_n,self.bd_n,self.xy_c)  # (b,1,40,2)
        
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
        w_t = np.concatenate((w_t,init_parameters.reshape(1,-1)), axis=1)
                
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

def run_simulation(i, args, num_envs, init_parameters):
    s_state = []
    s_bd = []
    s_angle = []
    s_f0 = []
    s_f1 = []
    s_act = []
    s_move = []
    
    s_parameter = []
    s_Re = []
    s_gap_angle = []
    s_gap = []
    s_dia0 = []
    s_dia1 = []
    
    # args.index = i
    env = gym.vector.make('lilypad-v1', num_envs=num_envs, asynchronous=False, wrappers=None, config=args, network_port=i)
    u_x, u_y, u_c, body_n, bd_n, xy_c = 128, 128, 3, args.num_structure, 40, 2

    lilypad = Lilypad(u_x, u_y, u_c, body_n, bd_n, xy_c, env, num_envs, init_parameters, args)
    
    for j in range(init_parameters[i].shape[0]):
        if j==0:
            s_parameter.append(int(init_parameters[i][j]))
        else:
            s_parameter.append(init_parameters[i][j])


    w_t = 0*np.ones((num_envs,args.num_structure))/args.action_interval
    init_state = lilypad.Lilypad_reset(w_t, i)
    done = [False] * num_envs  

    store_index = 1

    while not any(done):
        
        w_t = np.zeros((num_envs,args.num_structure))  

        sim_id = np.ones((num_envs,1))
        t = np.ones((num_envs,1))

        step_return, done = lilypad.Lilypad_step(w_t, sim_id, t, i)

        if store_index > args.store_step:
            if not any(done):
                s_state.append(step_return['u'])
                s_bd.append(step_return['bd'])
                
                if 'angle' in args.dict_state:
                    s_angle.append(step_return['angle'])
                if 'CD' in args.dict_state:
                    s_f0.append(step_return['CD'])
                if 'CL' in args.dict_state:
                    s_f1.append(step_return['CL'])
                        
        store_index += 1
                
    store_list = []
    store_list.append(np.array(s_state))
    store_list.append(np.array(s_bd))
    
    if 'angle' in args.dict_state:
        store_list.append(np.array(s_angle))
    if 'CD' in args.dict_state:
        store_list.append(np.array(s_f0))
    if 'CL' in args.dict_state:
        store_list.append(np.array(s_f1))
    for j in range(init_parameters[i].shape[0]):
        store_list.append(np.array(s_parameter[j]))
        
    filename = f"{i:06}.npz"
    for key in range(len(args.folder_name)):
        
        print(args.folder_name[key])
        np.savez(f'{args.store_path}/{args.folder_name[key]}/{filename}', store_list[key])

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
    
    parser = argparse.ArgumentParser(description='generate fixed cylinder simulated data')
    parser.add_argument('--traj_start_i', default=0, type=int,
                        help='multiple machine generate, the start parameter')
    parser.add_argument('--traj_end', default=20000, type=int,
                        help='multiple machine generate, the end parameter')
    parser.add_argument('--num_processes', default=1, type=int,
                        help='multiprocess number')
    parser.add_argument('--action_interval', default=10, type=int, 
                        help='gap of computation between two actions')
    parser.add_argument('--num_env', default=1, type=int, 
                        help='environment number of gym')
    
    parser.add_argument('--path_env', default='fixed_cylinder', type=str,
                        help='path of lilypad file')  
    parser.add_argument('--store_path', default='path', type=str,
                        help='path of dataset') 
    parser.add_argument('--check_path', default='path', type=str,
                        help='check the left number of data')    
    parser.add_argument('--name_env', default='fixed_cylinder', type=str,
                        help='name of lilypad file')    
    
    parser.add_argument('--observation_dim', default=49235, type=int,
                        help='dimensition of state')    
    parser.add_argument('--action_dim', default=3, type=int,
                        help='dimensition of action')    
    parser.add_argument('--num_structure', default=1, type=int,
                        help='number of structure in lilypad')    
    parser.add_argument('--all_features', default=['angle', 'CD', 'CL'], type=str,
                        help='output of lilypad, CD, CL, angle, move')    
    parser.add_argument('--dict_action', default={"v1": 0, "Re": 1, "dia_0": 2}, type=dict,
                        help='dictionary of action')    
    parser.add_argument('--dict_observe', default={"angle": 3, "CD": 4, "CL": 5}, type=dict,
                        help='dictionary of observation from lilypad')    
    parser.add_argument('--dict_state', default={"u": 49152, "bd": 49232, "angle": 49234, "CD": 49232, "CL": 49233}, type=dict,
                        help='dictionary of state when process the observation from lilypad and its index')    
    parser.add_argument('--folder_name', default=["state", "bd", "angle", "f_0", "f_1", "Re", "dia_0"], type=list,
                        help='the folder name stored')    
    parser.add_argument('--store_step', default=199, type=int,
                        help='from which step can interact and store, related to the calculation length')    
    
    args = parser.parse_args()
    
    num_envs = args.num_env
    folders = args.folder_name
    traj_start = args.traj_start_i
    traj_end = args.traj_end
    num_processes = args.num_processes
    
    init_parameters = np.load('parameters.npy')
    
    for folder in folders:
        if not os.path.exists(f'{args.store_path}/{folder}'):
            os.makedirs(f'{args.store_path}/{folder}')
            
    folder_path = args.check_path
    
    while True:
        missing_numbers = find_missing_numbers(folder_path, (traj_end-traj_start)*1)

        if len(missing_numbers) == 0:
            print("所有文件均已生成，无缺失文件编号。")
            break

        filtered_missing_numbers = [x for x in missing_numbers if traj_start <= x < traj_end]
        
        pool = multiprocessing.Pool(processes=num_processes) 
        func = partial(run_simulation, args=args, num_envs=num_envs, init_parameters=init_parameters)
        
        try:
            result = pool.map_async(func, filtered_missing_numbers)
            results = result.get(timeout=1200)
        except multiprocessing.TimeoutError:
            print("任务超时，跳过当前任务。")
            pool.close()  
            pool.join()  
            os.system('pkill -f processing')
            os.system('pkill -f xvfb')
            os.system('pkill -f Xvfb')
        except Exception as e:
            print(f"执行任务时发生错误：{e}")
            pool.close() 
            pool.join()  
            os.system('pkill -f processing')
            os.system('pkill -f xvfb')
            os.system('pkill -f Xvfb')
        
        pool.close()
        pool.join()