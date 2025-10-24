import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as ainimation
import numpy as np


t_0 =0
t_end =0.5
dt = 0.01
time_ = np.arange(t_0,t_end+1,dt)

# x_t function
x_t = 400*time_**2
y_t = 0.88*time_**2+1


############################ animation fonction for the animation##################################

# num will increase at every call except for the first one: 
# the num be =[0,0,0,1,2,3,........,frame_amount]

# then the frame amount can't be greater than len(vector to plot)

def update_frame(num):
    # we use the object my_car here in order to update data
    my_car.set_data(x_t[0:num],y_t[0:num])
    # initialisation of the lines :car_long.set_data([xi,xf],[yi,yf])
    car_long.set_data([x_t[num]-60,x_t[num]+60],[y_t[num],y_t[num]])
    car_uper1.set_data([x_t[num]+20,x_t[num]+40],[y_t[num]+0.45,y_t[num]])
    car_uper2.set_data([x_t[num]-40,x_t[num]-30],[y_t[num]+0.35,y_t[num]])
    car_lower1.set_data([x_t[num]+20,x_t[num]+40],[y_t[num]-0.45,y_t[num]])
    car_lower2.set_data([x_t[num]-40,x_t[num]-30],[y_t[num]-0.35,y_t[num]])
    tex1_.set_text(str(round(time_[num],ndigits=2))+ " "+"hours")
    x_value.set_data(time_[0:num],x_t[0:num])
    x_point.set_data([time_[num],time_[num]],[x_t[num]-0.5,x_t[num]])
    y_value.set_data(time_[0:num],y_t[0:num])
    
    
    x_hori.set_data([time_[0],time_[num]],x_t[num])
    vertic.set_data(time_[num],[x_t[0],x_t[num]])
    
    return my_car,car_long,car_lower1,car_lower2,car_uper1,car_uper2,tex1_,x_value,y_value,x_hori,vertic,x_point
    
# define a figure for the plots. dpi deals with de zoom the greater the dpi the greater zoom in
fig = plt.figure(figsize=(16,9),dpi = 120,facecolor=(0.9,0.9,0.9))
# slicing the figure
gs = gridspec.GridSpec(2,2)
# adding a subplot in the figure
ax_0 = fig.add_subplot(gs[0,:],facecolor=(0.96,1,1))

# create a plot object
# why the comma? : just to extract data from the list: example
# a=[1] if we print(a) will see [1] but if we do a, = [1] and print(a) we will see 1
my_car,= ax_0.plot([],[],'c.',linewidth=1) # an alternative is my_car = ax_0.plot([],[],'r',linewidth=2)[0]
car_long,= ax_0.plot([],[],'g',linewidth=15,solid_capstyle="butt") # solid_cpstyle: change the width without changing the length
car_uper1,= ax_0.plot([],[],'g',linewidth=10,solid_capstyle="butt") 
car_uper2,= ax_0.plot([],[],'g',linewidth=6,solid_capstyle="butt") 
car_lower1,= ax_0.plot([],[],'g',linewidth=10,solid_capstyle="butt") 
car_lower2,= ax_0.plot([],[],'g',linewidth=6,solid_capstyle="butt") 


builing_1,= ax_0.plot([100,100],[0,1],'k',linewidth=15,solid_capstyle="butt") 
builing_2,= ax_0.plot([400,400],[0,1.2],'k',linewidth=6,solid_capstyle="butt") 
builing_3,= ax_0.plot([600,600],[0,1.4],'k',linewidth=17,solid_capstyle="butt") 
builing_4,= ax_0.plot([1000,1000],[0,1],'k',linewidth=16,solid_capstyle="butt") 


# defining the axes limits
plt.xlim(x_t[0],x_t[-1])
plt.ylim(0,y_t[0]+2)
# title of the subplot
plt.title("l'Avion en l'air")
plt.grid('minor')
plt.xlabel("distance in x direction")
plt.ylabel("distance in y direction")

box_ = dict(boxstyle="round",ec=(0., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),alpha=0.5,pad=1)
tex1_ = ax_0.text(300,2.5,'',fontsize=8,color=(0.0,0.0,0.0),bbox=box_)

ax_1 = fig.add_subplot(gs[1,0],facecolor=(0.96,1,1))
x_value,= ax_1.plot([],[],'r-',linewidth=1)
x_point,= ax_1.plot([],[],'b.',linewidth=5)
x_hori,= ax_1.plot([],[],'r:',linewidth=1)
vertic,= ax_1.plot([],[],'b:',linewidth=1)
plt.xlim(time_[0],time_[-1])
plt.ylim(x_t[0],x_t[-1])
plt.grid(True)


ax_2 = fig.add_subplot(gs[1,1],facecolor=(0.96,1,1))
y_value,= ax_2.plot([],[],'b-',linewidth=2)
plt.xlim(time_[0],time_[-1])
plt.ylim(y_t[0],y_t[-1])
plt.grid(True)
"""
defining a second subplot
ax_1 = fig.add_subplot(gs[1,0],facecolor=(0.9,0.9,0.8))
my_car1,= ax_1.plot([],[],'g',linewidth=20)
plt.xlim(x_t[0],x_t[-1])
plt.ylim(0,y_t[0]+2)

"""
#
# to speed up the animation we just need to diminish the frame amount and the xlim or ylim end_values
frame_amount = len(time_)
my_car_anime =ainimation.FuncAnimation(fig,update_frame,frames=frame_amount,interval=10,repeat=True,blit=True)
plt.show() # bring the animation to the screen