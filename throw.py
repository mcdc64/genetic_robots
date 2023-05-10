import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import copy

class RobotArm:
    def __init__(self,k_p,k_d,target_theta,target_omega,length,max_accel=1,xpos=0,ypos=0,theta_0=0,omega_0=0):
        self.k_p = k_p
        self.k_d = k_d
        self.theta_0 = theta_0
        self.omega_0 = omega_0
        self.theta = theta_0 # initial theta
        self.omega = omega_0 # initial omega
        self.target_theta = target_theta # desired theta at release
        self.target_omega = target_omega # desired omega at release
        self.max_accel = max_accel # maximum allowed angular acceleration

        self.xpos = xpos
        self.ypos = ypos
        self.length = length

    def deriv(self,theta,t): # calculate acceleration to use with PD controller
        theta_diff = min(self.target_theta-self.theta,(np.deg2rad(360)+(self.target_theta-self.theta)))
        accel = self.k_p*(theta_diff) + self.k_d*(self.target_omega-self.omega)
        return [self.omega,np.clip(accel,-self.max_accel,self.max_accel)]

    def at_target(self,theta_tol = np.deg2rad(5),omega_tol = 0.1): # check if state is close enough to target state
        if (np.abs(self.theta-self.target_theta)<theta_tol and np.abs(self.omega - self.target_omega)<omega_tol):
            return True
        return False

    def rot_to_linear(self): # output Cartesian coordinates of the arm's end, and its velocity
        x_out = self.xpos + self.length*np.cos(self.theta)
        y_out = self.ypos + self.length*np.sin(self.theta)
        xdot_out = -self.length*np.sin(self.theta)*self.omega
        ydot_out = self.length*np.cos(self.theta)*self.omega
        return [x_out,y_out,xdot_out,ydot_out]

    def step(self,dt): # propagate arm's motion forward in time
        new_theta,new_omega = integrate.odeint(self.deriv,[self.theta,self.omega],[0,dt])[1]
        self.theta = new_theta
        self.omega = new_omega

        if(self.theta<0):
            self.theta +=np.deg2rad(360)
        if(self.theta>np.deg2rad(360)):
            self.theta -= np.deg2rad(360)

class Projectile:
    def __init__(self,x0,y0,xdot0,ydot0):
        self.x = x0
        self.y = y0
        self.xdot = xdot0
        self.ydot = ydot0
        self.g = 9.81
        self.active = False

    def deriv(self,x,t):
        return [self.xdot,self.ydot,0,-self.g]
    def step(self,dt):
        new_x,new_y,new_xdot,new_ydot = integrate.odeint(self.deriv,[self.x,self.y,self.xdot,self.ydot],[0,dt])[1]
        self.x = new_x
        self.y = new_y
        self.xdot = new_xdot
        self.ydot = new_ydot

def kill_propagate(robotarms,cost_funcs): # kill off worst performing robots and make new ones from the best
    indices = np.asarray(cost_funcs).argsort()
    num_arms = len(robotarms)
    sorted_arms = [None]*num_arms

    for k in range(0,len(robotarms)):
        sorted_arms[k] = robotarms[indices[k]]
    num_arms = len(sorted_arms)
    half_num_arms = num_arms//2 # assume even no. of robot arms
    new_arms = [None]*num_arms
    for a in range(0,num_arms):
        old_arm = sorted_arms[a]
        first_parent = sorted_arms[np.random.randint(0,25)]
        second_parent = sorted_arms[np.random.randint(0,25)]


        new_kp = first_parent.k_p#np.asarray([first_parent.k_p,second_parent.k_p])[np.random.randint(0,2)] * np.random.uniform(0.9, 1.1)
        new_kd = first_parent.k_d#np.asarray([first_parent.k_d,second_parent.k_d])[np.random.randint(0,2)]  * np.random.uniform(0.9, 1.1)
        new_target_theta = np.asarray([first_parent.target_theta,second_parent.target_theta])[np.random.randint(0,2)] + np.random.uniform(np.deg2rad(-5),np.deg2rad(5))
        new_target_omega = np.asarray([first_parent.target_omega,second_parent.target_omega])[np.random.randint(0,2)] + np.random.uniform(-0.5, 0.5)

        if(new_target_theta<0):
            new_target_theta += 2*np.pi

        if(new_target_theta>np.pi*2):
            new_target_theta -= 2*np.pi

        new_arms[a] = RobotArm(new_kp,new_kd,new_target_theta,new_target_omega,first_parent.length,
                                        first_parent.max_accel,first_parent.xpos,first_parent.ypos,first_parent.theta_0,first_parent.omega_0)
    return new_arms
    for a in range(0,num_arms): # replace the worst performers with new ones
        if(a<=half_num_arms):
            new_arms[a] = sorted_arms[a]
            new_arms[a].theta = 0
            new_arms[a].omega = 0
        else:
            better_arm = sorted_arms[a-half_num_arms]
            new_kp = better_arm.k_p*np.random.uniform(0.9,1.1)
            new_kd = better_arm.k_d*np.random.uniform(0.9,1.1)
            new_target_theta = better_arm.target_theta * np.random.uniform(0.9, 1.1)
            new_target_omega = better_arm.target_omega * np.random.uniform(0.9, 1.1)


            new_arms[a] = RobotArm(new_kp,new_kd,new_target_theta,new_target_omega,better_arm.length,
                                        better_arm.max_accel,better_arm.xpos,better_arm.ypos,better_arm.theta_0,better_arm.omega_0)
    return new_arms

def evaluate_cost_func(robotarm,target_x,target_y,dt = 0.05,maxtime = 10): # get the cost function of a single robot arm
    robotarm.theta = 0
    robotarm.omega = 0
    time = 0

    while(not(robotarm.at_target(theta_tol=np.deg2rad(10),omega_tol=0.2))):
        robotarm.step(dt)
        time += dt

        if(time>maxtime): # means the PD controller is not good enough to reach the target position - useless
            return np.sqrt(target_x**2+target_y**2)

    release_time = time
    x,y,xdot,ydot = robotarm.rot_to_linear()

    projectile = Projectile(x,y,xdot,ydot)
    mindist = 10000
    while(projectile.y>=-robotarm.length or (time-release_time)<5): # propagate projectile and see how close it gets to the target
        projectile.step(dt)
        time+=dt
        dist = np.sqrt((projectile.x-target_x)**2 + (projectile.y-target_y)**2)

        if(dist<mindist):
            mindist = dist
    return mindist

max_accel = 20
k_p = 10
k_d = 30
target_theta = np.deg2rad(135)
target_omega = -8
length = 1


target_theta_range = [0,np.pi//2]
target_omega_range = [-10,10]

target_x = 7
target_y = 5

num_arms = 60
generations = 50
robotarms = []

dt = 0.02

for i in range(num_arms):
    target_theta = np.random.uniform(*target_theta_range)
    target_omega = np.random.uniform(*target_omega_range)
    robotarms.append(RobotArm(k_p,k_d,target_theta,target_omega,length,max_accel = max_accel))

for i in range(generations):
    cost_funcs = []
    angles = []
    for j in range(num_arms):
        angles.append(robotarms[j].target_theta)
        cost_funcs.append(evaluate_cost_func(robotarms[j],target_x,target_y,dt=dt))
    print("Generation "+str(i+1)+": Min "+str(min(cost_funcs))+", Max "+str(max(cost_funcs)))
    robotarms = kill_propagate(robotarms,cost_funcs)


cost_funcs = []
for j in range(num_arms):
        cost_funcs.append(evaluate_cost_func(robotarms[j],target_x,target_y,dt=dt))

indices = np.asarray(cost_funcs).argsort()
sorted_arms = [None]*len(indices)

for k in range(0,len(robotarms)):
    sorted_arms[k] = robotarms[indices[k]]

robotarm = sorted_arms[0]
robotarm.theta = 0
robotarm.omega = 0
projectile = Projectile(0,0,0,0)


fig = plt.figure(figsize=[3,3])



ax = plt.axes()
plt.xlim([-2,8])
plt.ylim([-2,8])
armline, = ax.plot([0,robotarm.length], [0,0], "b",ms=2)
ax.plot([target_x],[target_y],"bs")
projectileline, = ax.plot([0],[0],"rs",ms = 2)

def init():
    armline.set_data([0,robotarm.length],[0,0])
    projectileline.set_data([0],[0])
    ax.set_aspect('equal')

    return armline,projectileline,

target_reached = False

def animate(n):
    global target_reached
    robotarm.step(dt)
    x,y,xdot,ydot = robotarm.rot_to_linear()
    armline.set_data([0,x],[0,y])
    if(robotarm.at_target(theta_tol = np.deg2rad(5),omega_tol = 0.5) and not target_reached):
        target_reached = True
        projectile.x = x
        projectile.y = y
        projectile.xdot = xdot
        projectile.ydot = ydot
        print("At target ("+str(np.rad2deg(robotarm.target_theta))+" deg, "+str(robotarm.target_omega)+" rad s^-1)")
    if(target_reached):
        projectile.step(dt)
        projectileline.set_data([projectile.x],[projectile.y])



    return armline,projectileline,
mng = plt.get_current_fig_manager()
mng.resize(300,300)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=30, interval=50, blit=True, repeat=True)
plt.show()
