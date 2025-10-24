
from particle_filter import ParticleFilter
from scipy.stats import multivariate_normal
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

# example of mass-spring damping system

m = 5 # kg
ks = 200 # N/m
kd = 30 # N/m
h = 0.01
simtime = 1000
A = np.array(
    [
        [0,1],
        [-ks/m,-ks/m]
    ]
)
B = np.array([
    [0],
    [1/m]
]
)
C = np.array([[1,0]])
X_0 = np.array([[0.1],[0.01]])

    
# calculate discrete matrix
Ad = np.linalg.inv(np.eye(2)-h*A)
Bd = np.matmul(Ad,B)
Cd = C

Cov1 = np.array([[0.002,0],
                 [0,0.002]
                 ])

Cov2 = np.array([[0.002]])
U = 100*np.ones((1,1))
""" noisemean = np.zeros(shape=X_0.shape)

NoiseDistro = multivariate_normal(mean=noisemean,cov=Cov1) """

Model = ParticleFilter()
Model.initialize_discrete(Ad,Bd,Cd,Cov1,Cov2,X_0,U)
Model.simulate(option='visualize')
