import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np


dt = np.pi/6
t_0 = 0
t_end = 48*np.pi
t = np.arange(t_0,t_end+dt,dt)

theta = dt*t/2
r = 24
x_t = r*np.cos(theta)
y_t = r*np.sin(theta)
z_t = 2*t

def update_data(num):
    #traject.set_data(x_t[0:num],y_t[0:num]): can be also used
    traject.set_xdata(x_t[0:num])
    traject.set_ydata(y_t[0:num])
    traject.set_3d_properties(z_t[0:num])

    return traject,


fig = plt.figure(figsize=(16,9),dpi=120,facecolor=(0.9,.9,.9))
gs = gridspec.GridSpec(2,2)
Axe1 = fig.add_subplot(gs[:,:],projection="3d")
traject, = Axe1.plot([],[],[],'c-',linewidth=3,label="elipse ")
Axe1.set_xlim(min(x_t),max(x_t))
Axe1.set_ylim(min(y_t),max(y_t))
Axe1.set_zlim(min(z_t),max(z_t))
Axe1.set_xlabel("position x",fontsize=10)
Axe1.set_ylabel("position y",fontsize=10)
Axe1.set_zlabel("position z",fontsize=10)
plt.grid(True)
plt.legend(loc="upper right",fontsize="large")

frame_amount = len(t)
anim = animation.FuncAnimation(fig,update_data,frames=frame_amount,interval=20,repeat=True,blit=True)
plt.show()