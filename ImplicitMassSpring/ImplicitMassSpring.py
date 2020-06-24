import taichi as ti
import numpy as np

ti.init(arch=ti.opengl)
#ti.core.toggle_advanced_optimization(False)

dim = 2
n_nodes_x = 7
n_nodes_y = 7
node_mass = 10
node_mass_inv = 1 / node_mass
n_nodes = n_nodes_x * n_nodes_y

dt = 1e-3#1e-3#3e-4
#dx = 1 / 64
#dx0 = 1 / 32
sx0, ex0 = 0.2, 0.8
sy0, ey0 = 0.4, 0.9
sx, ex = 0.2, 0.8
sy, ey = 0.4, 0.9#0.5, 0.9
groundy = 0.2
step = int(np.round(1/60/dt)) #30
kd = 20#1000#20
k1 = 5000
k2 = 1000
k3 = 500
g = 1#9.8
eps = 4e-2
staticEpsilon = 4e-3
jacobiStep = 30#100

color_1 = 0x4FB99F
color_2 = 0x9F4FB9
color_3 = 0xB99F4F
color_white = 0xFFFFFF
color_gray = 0x777777

gAcc = ti.Vector(dim, dt=ti.f32, shape=())
connMat = ti.var(dt=ti.f32, shape=(n_nodes, n_nodes))
restLengthMat = ti.var(dt=ti.f32, shape=(n_nodes, n_nodes))
I2d = ti.Matrix(dim, dim, dt=ti.f32, shape=())
fParXMat = ti.Matrix(dim, dim, dt=ti.f32, shape=(n_nodes,))
fParVMat = ti.Matrix(dim, dim, dt=ti.f32, shape=(n_nodes,))
AMat = ti.Matrix(dim, dim, dt=ti.f32, shape=(n_nodes,))
BVec = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,))

x02d = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,))

x2d = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,), needs_grad=False)
v2d = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,))
v2dNext = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,))
f2d = ti.Vector(dim, dt=ti.f32, shape=(n_nodes,))
fix2d = ti.var(dt=ti.i32, shape=(n_nodes,))

static = ti.var(ti.i32, shape=())
maxvn2 = ti.var(ti.f32, shape=())

@ti.func
def lerp(x, s, e): 
    return (1-x)*s+x*e

@ti.func
def initConstant():
    #fix
    fix2d[n_nodes_y-1] = 1
    fix2d[n_nodes-1] = 1
    
    static[None] = 0
    maxvn2[None] = 0.0
    
    gAcc[None] = [0.0, -g]

@ti.kernel
def init():
    d = ti.static(dim)
    for di in ti.static(range(d)):
        I2d[None][di, di] = 1.0

    for idx in ti.static(range(n_nodes)):
        i = idx // n_nodes_y
        j = idx - i *  n_nodes_y
        x = i / (n_nodes_x - 1)
        y = j / (n_nodes_y - 1)
        
        x02d[idx] = [lerp(x, sx0, ex0), lerp(y, sy0, ey0)]
        x2d[idx] = [lerp(x, sx, ex), lerp(y, sy, ey)]
        
        v2d[idx] = [0, 0]
        f2d[idx] = [0, 0]
        #f2dNext[idx] = [0, 0]
        fix2d[idx] = 0
        #print(idx)
    initConstant()

    for idx, idx2 in connMat:
        length = (x02d[idx] - x02d[idx2]).norm()
        #print(idx, idx2, length)
        restLengthMat[idx, idx2] = length
        restLengthMat[idx2, idx] = length
        i = idx // n_nodes_y
        j = idx - i *  n_nodes_y
        i2 = idx2 // n_nodes_y
        j2 = idx2 - i2 * n_nodes_y
        if ti.abs(i - i2) == 0 and ti.abs(j - j2) == 1:
            connMat[idx, idx2] = k1
            connMat[idx2, idx] = k1
        elif ti.abs(i - i2) == 1 and ti.abs(j - j2) == 0:
            connMat[idx, idx2] = k1
            connMat[idx2, idx] = k1
        elif ti.abs(i - i2) == 1 and ti.abs(j - j2) == 1:
            connMat[idx, idx2] = k2
            connMat[idx2, idx] = k2
        elif ti.abs(i - i2) == 0 and ti.abs(j - j2) == 2:
            connMat[idx, idx2] = k3
            connMat[idx2, idx] = k3
        elif ti.abs(i - i2) == 2 and ti.abs(j - j2) == 0:
            connMat[idx, idx2] = k3
            connMat[idx2, idx] = k3
        else:
            connMat[idx, idx2] = 0.0
            connMat[idx2, idx] = 0.0
            
@ti.kernel
def initRunning():
    for idx in range(n_nodes):
        i = idx // n_nodes_y
        j = idx - i *  n_nodes_y
        x = i / (n_nodes_x - 1)
        y = j / (n_nodes_y - 1)
        
        #x02d[idx] = [lerp(x, sx0, ex0), lerp(y, sy0, ey0)]
        x2d[idx] = [lerp(x, sx, ex), lerp(y, sy, ey)]
        
        v2d[idx] = [0, 0]
        f2d[idx] = [0, 0]
        #f2dNext[idx] = [0, 0]
        fix2d[idx] = 0
        #print(idx)
    initConstant()

@ti.func
def calElasticForce(taridx, srcidx, k):
    tarx2d = x2d[taridx]
    srcx2d = x2d[srcidx]
    curDiff = tarx2d - srcx2d
    curLength = ti.sqrt(curDiff.dot(curDiff))
    restLength = restLengthMat[taridx, srcidx]
    direction = curDiff / curLength
    deformValue = curLength - restLength
    elastic = -k * deformValue * direction
    return elastic

@ti.func
def calDampForce(taridx, kd):
    damp = -kd * v2d[taridx]# * v2d[taridx].norm()
    return damp

@ti.kernel
def calTotalForce():
    for taridx in f2d:
        f2d[taridx] = calDampForce(taridx, kd) + node_mass * gAcc
    for taridx, srcidx in connMat:
        if connMat[taridx, srcidx] != 0.0:
            f2d[taridx] += calElasticForce(taridx, srcidx,
                                           connMat[taridx, srcidx])

@ti.func
def calElasticPartialOnX(taridx, srcidx, k):
    """
    f = -k(||tarx-srcx||-restLength)*normalize(tarx-srcx)
    par(||x||)/par(x) = normalize(x)^T
    par(normalize(x))/par(x) = (I - normalize(x)*normalize(x)^T)/||x||
    """
    tarx2d = x2d[taridx]
    srcx2d = x2d[srcidx]
    curDiff = tarx2d - srcx2d
    curLength = ti.sqrt(curDiff.dot(curDiff))
    restLength = restLengthMat[taridx, srcidx]
    direction = curDiff / curLength
    deformValue = curLength - restLength
    cross = direction @ direction.transpose()
    elasticPartialOnX = k * (deformValue / curLength * (cross - I2d) - cross)
    return elasticPartialOnX

@ti.func
def calDampPartialOnV(taridx, kd):
    return -kd * I2d

@ti.kernel
def calTotalForcePartial():
    d = ti.static(dim)
    for taridx in f2d:
        fParVMat[taridx] = calDampPartialOnV(taridx, kd)
    for taridx, srcidx in connMat:
        for i, j in ti.static(ti.ndrange((0, d),(0, d))):
            fParXMat[taridx][i,j] = 0.0
        if connMat[taridx, srcidx] != 0.0:
            fParXMat[taridx] += \
                calElasticPartialOnX(taridx, srcidx, connMat[taridx, srcidx])

@ti.kernel
def bulidLinearSystem(): #Backward Euler
    for taridx in range(n_nodes):
        if fix2d[taridx] == 0:
            AMat[taridx] = I2d - dt*node_mass_inv*fParVMat[taridx] \
                - dt*dt*node_mass_inv*fParXMat[taridx]
            BVec[taridx] = dt*node_mass_inv*f2d[taridx] \
                + v2d[taridx] - dt*node_mass_inv*fParVMat[taridx]@v2d[taridx]

@ti.kernel
def solve(): #Jacobi Solver
    for taridx in range(n_nodes):
        if fix2d[taridx] == 0:
            for t in range(jacobiStep): #If ti.static, needs more compile time
                for i in ti.static(range(dim)):
                    r = BVec[taridx][i]
                    for j in ti.static(range(dim)):
                        if i != j:
                            r -= AMat[taridx][i, j] * v2d[taridx][j]
                    v2dNext[taridx][i] = r / AMat[taridx][i, i]
                for i in ti.static(range(dim)):
                    v2d[taridx][i] = v2dNext[taridx][i]

@ti.kernel
def integrate():
    for i in range(n_nodes):
        if fix2d[i] == 0:
            # Collide with ground
            if x2d[i][1] < groundy:
                x2d[i][1] = groundy
                v2d[i][1] = 0
            x2d[i] = x2d[i] + dt*v2d[i]
        else:
            for j in ti.static(range(dim)):
                v2d[i][j] = 0.0

@ti.kernel
def checkStatic():
    static[None] = 1
    maxvn2[None] = 0.0
    for i in ti.static(range(n_nodes)):
        v2dn2 = v2d[i].dot(v2d[i])
        ti.atomic_max(maxvn2[None], v2dn2)
        if v2dn2 >= staticEpsilon * staticEpsilon:
            static[None] = 0

@ti.kernel
def findTarget(mousex: ti.f32, mousey: ti.f32)->ti.i32:
    taridx = -1
    for i in ti.static(range(n_nodes)):
        x, y = x2d[i][0], x2d[i][1]
        if (mousex-x)*(mousex-x) + (mousey-y)*(mousey-y) < eps*eps:
            taridx = i
            #break
    return taridx

bindTarget = None

gui = ti.GUI("Implicit Mass Spring", (640, 640), background_color=0x112F41)

init()
print("init complete")

framecount = 0
maxframecount = -1#600
staticFrame = -1
staticConCount = 0

c2to1 = lambda i, j: i * n_nodes_y + j

while gui.running:
    action = None#f'Frame/frame_{framecount:05d}.png'
    for s in range(step):
        calTotalForce()
        calTotalForcePartial()
        bulidLinearSystem()
        #print("before solve()")
        solve()
        #print("after solve()")
        integrate()
    while gui.get_event(ti.GUI.PRESS):
        pass
    mousex, mousey = gui.get_cursor_pos()
    taridx = findTarget(mousex, mousey)
    if gui.is_pressed(ti.GUI.LMB):
        if bindTarget is None:
            if taridx >= 0:
                bindTarget = taridx
        else:
            fix2d[bindTarget] = 1
            x2d[bindTarget][0] = mousex
            x2d[bindTarget][1] = mousey
    else:
        bindTarget = None
    if gui.is_pressed(ti.GUI.RMB):
        if taridx >= 0:
            fix2d[taridx] = 0
            
    if gui.is_pressed(gui.SPACE):
        framecount = 0
        initRunning()
    
    x_np = x2d.to_numpy()
    #print(x_np[0])
    for i in range(n_nodes):
        if fix2d[i] > 0:
            gui.circle((x_np[i][0], x_np[i][1]),
                       color=color_white,
                       radius=5)
        
    for i in range(n_nodes_x - 1):
        for j in range(n_nodes_y - 1):
            gui.line((x_np[c2to1(i,j)][0], x_np[c2to1(i,j)][1]),
                     (x_np[c2to1(i,j+1)][0], x_np[c2to1(i,j+1)][1]),
                     radius=1,
                     color=color_1)
            gui.line((x_np[c2to1(i,j)][0], x_np[c2to1(i,j)][1]),
                     (x_np[c2to1(i+1,j)][0], x_np[c2to1(i+1,j)][1]),
                     radius=1,
                     color=color_1)
            gui.line((x_np[c2to1(i,j+1)][0], x_np[c2to1(i,j+1)][1]),
                     (x_np[c2to1(i+1,j+1)][0], x_np[c2to1(i+1,j+1)][1]),
                     radius=1,
                     color=color_1)
            gui.line((x_np[c2to1(i+1,j)][0], x_np[c2to1(i+1,j)][1]),
                     (x_np[c2to1(i+1,j+1)][0], x_np[c2to1(i+1,j+1)][1]),
                     radius=1,
                     color=color_1)
            
            gui.line((x_np[c2to1(i,j)][0], x_np[c2to1(i,j)][1]),
                     (x_np[c2to1(i+1,j+1)][0], x_np[c2to1(i+1,j+1)][1]),
                     radius=1,
                     color=color_2)
            gui.line((x_np[c2to1(i,j+1)][0], x_np[c2to1(i,j+1)][1]),
                     (x_np[c2to1(i+1,j)][0], x_np[c2to1(i+1,j)][1]),
                     radius=1,
                     color=color_2)
    
    
    for i in range(n_nodes_y - 2):
        for j in range(n_nodes_x - 2):
            gui.line((x_np[c2to1(i,j)][0], x_np[c2to1(i,j)][1]),
                     (x_np[c2to1(i,j+2)][0], x_np[c2to1(i,j+2)][1]),
                     radius=1,
                     color=color_3)
            gui.line((x_np[c2to1(i,j)][0], x_np[c2to1(i,j)][1]),
                     (x_np[c2to1(i+2,j)][0], x_np[c2to1(i+2,j)][1]),
                     radius=1,
                     color=color_3)
            gui.line((x_np[c2to1(i,j+2)][0], x_np[c2to1(i,j+2)][1]),
                     (x_np[c2to1(i+2,j+2)][0], x_np[c2to1(i+2,j+2)][1]),
                     radius=1,
                     color=color_3)
            gui.line((x_np[c2to1(i+2,j)][0], x_np[c2to1(i+2,j)][1]),
                     (x_np[c2to1(i+2,j+2)][0], x_np[c2to1(i+2,j+2)][1]),
                     radius=1,
                     color=color_3)
            
    if taridx >= 0:
        gui.circle((x_np[taridx][0], x_np[taridx][1]),
                   color=color_gray,
                   radius=5)
    
    #gui.circles(node_x, radius=1.5, color=0x3241f4)
    gui.line((0.00, groundy), (1.0, groundy), color=color_white, radius=3)
    checkStatic()
    #print(([v.dot(v) for v in v2d.to_numpy()]), '%.6f'%maxvn2[None])
    staticText = ''
    if static[None] == 1:
        staticConCount += 1
        if staticConCount > 30:
            if staticFrame < 0:
                staticFrame = framecount
            staticText = ' (static at %d)'%staticFrame
    else:
        staticConCount = 0
        staticFrame = -1
    gui.text('Implicit Mass Spring B-Euler (dt = %.6f, framecount = %d)' \
             %(dt, framecount), 
             (0.0, 0.13), font_size=24, color=0xFFFFFF)
    gui.text('(max sqrvel = %.6f)'%maxvn2[None] + staticText,
             (0.0, 0.07), font_size=24, color=0xFFFFFF)
    gui.show(action)
    framecount += 1
    if maxframecount >= 0 and framecount > maxframecount:
        break
        #gui.running = False
#while gui.running:
#    while gui.get_event(ti.GUI.PRESS):
#        pass
gui.core = None
del gui