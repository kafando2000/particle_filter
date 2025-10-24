import time
# for creating normal distro for multivariate
from scipy.stats import multivariate_normal
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec



# class for particle filter algorithm
class ParticleFilter:
    def __init__(self):
        np.random.seed(42)

    def resample(self,weightvalues):
        try :
            N,l= weightvalues.shape
            N = max(N,l)
            print("N : ",N)
            self.c_value_.append(float(weightvalues[:,[0]]))
            # cumulative proba
            for j in range(N-1):
                self.c_value_.append(self.c_value_[j]+ float(weightvalues[:,[j+1]]))
            # pick a starting point
            start_point = np.random.uniform(low=0.0,high=1/N)

            # draw the indices of the state t to use as resampled particles
            SampledIndices = []
            for i in range(N):
                courrent_val = start_point + (1/N)*(i)
                k = 0
                while(courrent_val>self.c_value_[k] and k<N-1):
                    k += 1
                    
                SampledIndices.append(k)
            return SampledIndices
        except:
            print("something went wrong! maybe the initial values are not convenient")
            return []

    def set_distribution(self):
        self.NoiseDistro = multivariate_normal(mean=self.noise_mean_.reshape((self.C_.shape[0],)),cov=self.noise_cov_)
        self.ProcessDistro = multivariate_normal(mean=self.state_mean_.reshape((self.initial_state.shape[0],)),cov=self.state_cov_)
        return
    
    def initialize_discrete(self,A_d: np.array,B_d: np.array,C_d: np.array,State_cov: np.array,noise_cov: np.array,initial_stateX: np.array,control_U: np.array):
        try:
            self.sim_time =10.0
            self.time_step =0.01

            self.ControlInput = control_U
            self.c_value_ = []
            self.state_cov_ = State_cov
            self.noise_cov_ = noise_cov
            self.initial_state = initial_stateX
            self.A_ = A_d;self.B_= B_d;self.C_ = C_d
            self.noise_mean_ = np.zeros((self.C_.shape[0],))
            self.state_mean_ = np.zeros(shape=(self.initial_state.shape[0],1))
            self.set_distribution()

            self.XGuess = self.initial_state
            Points = np.mgrid[self.XGuess[0]-0.08 : self.XGuess[0]+0.08 : .08,self.XGuess[1]-0.08 : self.XGuess[1]+0.08 : .01]
           
            Xvect = Points[0].reshape((-1,1),order="C")
            Yvect = Points[1].reshape((-1,1),order="C")
        
            # horizontal concatenation
            self.current_states = np.hstack((Xvect,Yvect)).transpose()
            self.dim1,self.ParticleNumber = self.current_states.shape
            self.iteration = int(self.sim_time/self.time_step)
            self.weights_ = np.zeros(shape=(1,self.ParticleNumber))
            
            self.state_trajectory = np.zeros(shape=(self.initial_state.shape[0],self.iteration))
            self.output = np.zeros(shape=(self.C_.shape[0],self.iteration))
            return
        except:
            print('initialization has failed')
            return
    def create_simulate_particle(self):
        
        
        self.estimated_state = self.initial_state
        
        self.estimated_states = np.zeros(shape=(self.dim1,self.iteration))
        self._error_sequence = np.zeros(shape=(self.dim1,self.iteration))
        self.weights_ = (1/self.ParticleNumber)*np.ones((1,self.ParticleNumber))
        
       
        # calculating in batch basis
        for i in range(self.iteration):
            ControlInputBatch = self.ControlInput*np.ones((1,self.ParticleNumber))
            NewStates = np.matmul(self.A_,self.current_states) + np.matmul(self.B_,ControlInputBatch) + self.ProcessDistro.rvs(size=self.ParticleNumber).transpose()
            self.current_states = NewStates
            self.weights_ = (1/self.ParticleNumber)*np.ones((1,self.ParticleNumber))
            NewWeights = np.zeros((1,self.ParticleNumber))
            for j in range(self.ParticleNumber):
                current_noise_mean_ = np.matmul(self.C_,NewStates[:,[j]])
                Distro = multivariate_normal(mean=current_noise_mean_,cov=self.noise_cov_)
                NewWeights[:,[j]] = (Distro.pdf(self.output[:,[i]]))*self.weights_[:,[j]]

            # normalize the weights
            self.weights_ = (1/NewWeights.sum())*NewWeights
            

            # calculate Neff
            W_squared_sum = self.weights_.dot(self.weights_.transpose())

            Neff = 1/W_squared_sum

            if Neff < (self.ParticleNumber/3) :
                Indices = self.resample(self.weights_)
                self.current_states = NewStates[:,Indices]
                self.weights_ = (1/self.ParticleNumber)*np.ones((1,self.ParticleNumber))
            self.estimated_state = np.zeros(self.state_mean_.shape)

            for l in range(self.ParticleNumber):
                self.estimated_state = self.estimated_state + (self.weights_[:,[l]])*self.current_states[:,[l]]
            
            self.estimated_states[:,[i]] = self.estimated_state
            self._error_sequence[:,[i]] = self.estimated_states[:,[i]] - self.state_trajectory[:,[i]]
            print("iteration : ", i)
        
    def compute_true_states(self):
        self.set_distribution()
        self.state_trajectory[:,[0]] = self.initial_state
        for i in range(self.iteration-1):
            self.state_trajectory[:,[i+1]] = np.matmul(self.A_,self.state_trajectory[:,[i]]) +\
                  np.matmul(self.B_,self.ControlInput) + (self.ProcessDistro.rvs().transpose()).reshape(self.initial_state.shape)
            
            self.output[:,[i]]= np.matmul(self.C_,self.state_trajectory[:,[i]]) + (self.NoiseDistro.rvs().transpose()).reshape((self.C_.shape[0],-1))
       

    def update_control(self,control:np.array):
        self.ControlInput = control

    # customized simulation params
    def set_sim_time(self,sim_time,time_step):
        self.sim_time = sim_time
        self.time_step = time_step

    # simulate the process and measurement
    def simulate(self,option='nothing'):
        self.compute_true_states()
        self.create_simulate_particle()
        if option =='visualize':
            self.visualize_simulation()
        elif option == 'animate':
            self.animate_simulation()

    # visualize the simulation
    def visualize_simulation(self):
        
        fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
        gs=gridspec.GridSpec(4,3)
        ax_main=fig.add_subplot(gs[0:,0:2],facecolor=(0.99,0.99,0.99))
        x_min = self.state_trajectory[0,0:100].min()
        x_max = self.state_trajectory[0,0:100].max()
        time_max = self.sim_time+self.time_step
        tim_min = 0

        t = np.arange(tim_min,time_max-self.time_step,self.time_step)
        plt.xlim(tim_min,t[100])
        plt.ylim(x_min,x_max)
        """ plt.xticks(np.arange(tim_min,time_max,4*self.time_step))
        plt.yticks(np.arange(x_min,x_max,4*(x_max-x_min)/self.iteration)) """

        ax_main.plot(t[0:100],self.state_trajectory[0,0:100],'r-')

        ax_main.plot(t[0:100],self.estimated_states[0,0:100],'b-')
        plt.grid(True)
        error_ax=fig.add_subplot(gs[0:1,2:],facecolor=(0.99,0.99,0.99))

        error_ax.plot(t[0:100],self._error_sequence[0,0:100],'b-')
        plt.grid(True)
        plt.show()

    # animation
    def animate_simulation(self):
        fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
        gs=gridspec.GridSpec(4,3)
        ax_main=fig.add_subplot(gs[0:3,0:2],facecolor=(0.9,0.9,0.9))
        plt.xlim(self.state_trajectory[0,:].min()-self.state_trajectory[0,:].min()/3,self.state_trajectory[0,:].max()+self.state_trajectory[0,:].max()/3)
        plt.ylim(self.state_trajectory[1,:].min()-self.state_trajectory[1,:].min()/3,self.state_trajectory[1,:].max()+self.state_trajectory[1,:].max()/3)
        plt.xticks(np.arange(0,self.sim_time+self.time_step,self.time_step))
        plt.yticks(np.arange(0,self.sim_time+self.time_step,self.time_step))
        
        platform,=ax_main.plot([],[],'b',linewidth=18)
        cube,=ax_main.plot([],[],'k',linewidth=14)
    def update_plot(self,num):
        pass