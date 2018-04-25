import numpy as np
import math
import matplotlib.pylab as plt
import statistics as mths
from scipy.stats import skew
from scipy.stats import kurtosis
'''

heat_map_list:- stores the information of how much time is the person looking at a perticular block for all the 40 trials
heat_title_label: - contains the information of what the user think of the gui (congruent or not) for all the 40 trial
act_list = contains the actual trail type number for all the 40 trial


'''

act_x = [481,481,481,481,481,481,1440,1440,1440,1440,1440,1440]
act_y = [125.5,297.5,470,642.5,815,990,125.5,297.5,470,642.5,815,990]




t_dist = 0
dx_dist = 0
dy_dist = 0
heat_map_list = [0] * 12 * 40
heat_title_label = []
act_list = []
dx_dist_list = []
dy_dist_list = []
t_dist_list = []
mean_list = []
var_list = []
max_list = []
max_index_list = []
skew_list = []
kurt_list = []
std_list = []
mean_trans_list = []
var_trans_list = []
max_trans_list = []
max_trans_index_list = []
skew_trans_list = []
kurt_trans_list = []
std_trans_list = []
class Point:  pass

'''
def plot_line(list_x,list_y,label_str):
    if(label_str == 'Yes'):
        plt.subplot(121)
        plt.plot(list_x,list_y)
        plt.title('Congruent')
    if(label_str == 'No'):
        plt.subplot(122)
        plt.plot(list_x,list_y)
        plt.title('Not Congruent')
    plt.suptitle('Gaze map of human eye in Line Plot')
'''

def calculate_heat_map(list_x,list_y,label_str,count):
    heat_title_label.append(label_str)
    for i in range(0,len(list_x)) :
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 112 and list_y[i] < 157):
            heat_map_list[0 + 12*count - 12 ] += 1
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 289 and list_y[i] < 315):
            heat_map_list[1 + 12*count - 12 ] += 1
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 461 and list_y[i] < 488):
            heat_map_list[2 + 12*count - 12 ] += 1
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 629 and list_y[i] < 666):
            heat_map_list[3 + 12*count - 12 ] += 1
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 802 and list_y[i] < 836):
            heat_map_list[4 + 12*count - 12 ] += 1
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 980 and list_y[i] < 1015):
            heat_map_list[5 + 12*count - 12 ] += 1

        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 40 and list_y[i] < 210):
            heat_map_list[6 + 12*count - 12 ] += 1
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 215 and list_y[i] < 380):
            heat_map_list[7 + 12*count - 12 ] += 1
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 385 and list_y[i] < 555):
            heat_map_list[8 + 12*count - 12 ] += 1
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 560 and list_y[i] < 725):
            heat_map_list[9 + 12*count - 12 ] += 1
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 730 and list_y[i] < 900):
            heat_map_list[10 + 12*count - 12 ] += 1
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 705 and list_y[i] < 1075):
            heat_map_list[11 + 12*count - 12 ] += 1

'''
def draw_heat_map():
    clms = 10
    rows = 4
    cnt = 0
    y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    newList = [0] * 12
    act_str = ''
    fig, ax_array = plt.subplots(rows,clms)
    for i,ax_row in enumerate(ax_array):
        for j,axis in enumerate(ax_row):
            if(act_list[cnt] < 20):
                act_str = 'Yes'
            elif(act_list[cnt] >=20 and act_list[cnt] <40):
                act_str = 'No'
            axis.set_title(act_str +'  '+heat_title_label[cnt])
            axis.set_yticklabels(y)
            newList = [x / sum(heat_map_list[ cnt * 12 : cnt * 12 + 12]) for x in heat_map_list[ cnt * 12 : cnt * 12 + 12]]
            axis.plot(newList)
            newList[:] = []
            cnt += 1
    plt.suptitle('Time Spend looking a block in a GUI in probability [Left: Correct Judgment & Right: User Judgement]')
'''

def find_Mean():
    mean_list[:] = []
    for cnt in range(0,40):
        mean_list.append(sum(heat_map_list[ cnt * 12 : cnt * 12 + 12])/float(12.0))
    #plot_stats(mean_list,'Mean Gazed Block Plot')
    #return mean_list


def find_Max():
    max_list[:] = []
    max_index_list[:] = []
    for cnt in range(0,40):
        max_list.append(max(heat_map_list[ cnt * 12 : cnt * 12 + 12]))
        max_index_list.append(heat_map_list[ cnt * 12 : cnt * 12 + 12].index( max(heat_map_list[ cnt * 12 : cnt * 12 + 12]) ) + 1 )
    #plot_stats(max_list,'Max Gazed Block Plot')
    #return max_list, max_index_list



def find_Std():
    std_list[:] = []
    for cnt in range(0,40):
        std_list.append(mths.stdev(heat_map_list[ cnt * 12 : cnt * 12 + 12]))
    #plot_stats(std_list,'Standard Deviation Plot')
    #return std_list

def find_Var():
    var_list[:] = []
    for cnt in range(0,40):
        var_list.append(mths.variance(heat_map_list[ cnt * 12 : cnt * 12 + 12]))
    #plot_stats(var_list,'Variance Plot')
    #return var_list

def find_Skew():
    skew_list[:] = []
    for cnt in range(0,40):
        skew_list.append(skew(heat_map_list[ cnt * 12 : cnt * 12 + 12]))
    #plot_stats(skew_list,'Skew Plot')
    #return skew_list


def find_Kurt():
    kurt_list[:] = []
    for cnt in range(0,40):
        kurt_list.append(kurtosis(heat_map_list[ cnt * 12 : cnt * 12 + 12]))
    #plot_stats(kurt_list,'Kurtosis Plot')
    #return kurt_list

'''
def plot_stats(my_list, my_title):
    c_list = []
    w_list = []
    for i in range(0,len(act_list)):
        if i < 20:
            c_list.append(my_list[i])
        else:
            w_list.append(my_list[i])
    c_newList = [x / sum(c_list) for x in c_list]
    w_newList = [x / sum(w_list) for x in w_list]
    low = min (min(c_newList),min(w_newList))
    high = max (max(c_newList),max(w_newList))
    plt.figure()
    plt.subplot(121)
    plt.plot(c_newList)
    plt.title('Congruent GUI')
    plt.ylim((low,high))
    plt.subplot(122)
    plt.plot(w_newList)
    plt.title('Not Congruent GUI')
    plt.ylim((low,high))
    plt.suptitle(my_title)
'''


def stats_transition(Matrix):
    i,j = np.unravel_index(Matrix.argmax(), Matrix.shape)
    Matrix = Matrix.flatten()
    max_trans_index_list.append(i+1)
    #mean_trans_list.append(sum(Matrix)/float(12*12))
    std_trans_list.append(mths.stdev(Matrix))
    var_trans_list.append(mths.variance(Matrix))
    skew_trans_list.append(skew(Matrix))
    kurt_trans_list.append(kurtosis(Matrix))



def stats():
    find_Max()
    find_Std()
    find_Var()
    find_Kurt()
    find_Mean()
    find_Skew()

    #print ('Mean of each trial')
    #print(find_Mean())
    #print ('Max of each trial')
    #print (find_Max())
    #print ('Std of each trial')
    #print (find_Std())
    #print ('Variance of each trial')
    #print (find_Var())
    #print ('Skewness of each trial')
    #print (find_Skew())
    #print ('Kurtosis of each trial')
    #print (find_Kurt())
    return 0


def getArrow(p1,p2,i,ax):
    # we need to subtract some from each end
    # slope = m
    w = p2.x - p1.x
    h = p2.y - p1.y
    #print p1.x,p1.y
    #print p2.x,p2.y
    #print 'w',w,'h',h

    dr = 0.03
    if w == 0:
        dy = dr
        dx = 0
    else:
        theta = np.arctan(np.abs(h/w))
        dx = dr*np.cos(theta)
        dy = dr*np.sin(theta)
    #print 'dx',dx,'dy',dy

    if w < 0:  dx *= -1
    if h < 0:  dy *= -1
    #print 'dx',dx,'dy',dy
    w -= 2*dx
    h -= 2*dy
    #print 'w',w,'h',h
    x = p1.x + dx
    y = p1.y + dy
    #print 'x',x,'y',y

    #a = ax.arrow(x,y,w,h,width=5,zorder=i+1)
    #a.set_facecolor('0.7')
    #a.set_edgecolor('b')
    return a


def plotpath(list_x,list_y,path_list):
    N = len(path_list)
    pL = list()
    for i in range(0,N,1):
        p = Point()
        p.x,p.y = act_x[path_list[i]-1],act_y[path_list[i]-1]
        pL.append(p)

    t_dist = 0
    dx_dist = 0
    dy_dist = 0
    # no a comment
    #fig = plt.figure()
    #ax = plt.axes()
    for i,p in enumerate(pL):
        if i:
            #a = getArrow(pL[i-1],p,i,ax)
            t_dist = t_dist + math.hypot(p.x - pL[i-1].x, p.y - pL[i-1].y)
            dx_dist = dx_dist + math.fabs(p.x - pL[i-1].x)
            dy_dist = dy_dist + math.fabs(p.y - pL[i-1].y)
            #ax.add_patch(a)
        #plt.scatter(p.x,p.y,s=250,zorder=1)
    t_dist_list.append(t_dist)
    dx_dist_list.append(dx_dist)
    dy_dist_list.append(dy_dist)
    mean_trans_list.append(t_dist/float(12.0))
    #ax.set_xlim(min(list_x) - 200,max(list_x) + 200)
    #ax.set_ylim(min(list_y) - 200,max(list_y) + 200)
    # not a comment
    #plt.show()


def cal_matrix(path_list):
    #output_matrix_file = open('../Experiment_Results/output_matrix.txt','a')
    len_mat = 12
    Matrix = []
    #print (path_list)
    for i in range(len_mat):
        Matrix.append([])
        for j in range(len_mat):
            Matrix[i].append(0)
    for i in range(len(path_list)-1):
        j = i+1
        x = path_list[i] - 1
        y = path_list[j] - 1
        Matrix[path_list[i] - 1][path_list[j] - 1] += 1
    #sum1 = 0
    #for row in range (len_mat):
    #    for col in range(len_mat):
    #        sum1 = sum1 + Matrix[row][col]
    stats_transition(np.array(Matrix))
    #output_matrix_file.write(str(np.array(Matrix)))
    #output_matrix_file.write('\n')
    #print (Matrix, len(path_list), sum1)





def calpath(list_x,list_y,list_time,count):
    path_list = []
    thresh = 6
    n = 0
    f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True,True
    for i in range(0,len(list_x)) :
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 112 and list_y[i] < 157 and f_1 == True ):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 112 and list_y[j] < 157 and f_1 == True):
                    pass
                else:
                    f_1 = False
                    break
            if f_1 == True:
                path_list.append(1)
            #path_list.append(1)
            f_1 = False
            f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 289 and list_y[i] < 315 and f_2 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 289 and list_y[j] < 315 and f_2 == True):
                    pass
                else:
                    f_2 = False
                    break
            if f_2 == True:
                path_list.append(2)
            #path_list.append(2)
            f_2 = False
            f_1,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 461 and list_y[i] < 488 and f_3 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 461 and list_y[j] < 488 and f_3 == True):
                    pass
                else:
                    f_3 = False
                    break
            if f_3 == True:
                path_list.append(3)
            #path_list.append(3)
            f_3 = False
            f_1,f_2,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 629 and list_y[i] < 666 and f_4 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 629 and list_y[j] < 666 and f_4 == True):
                    pass
                else:
                    f_4 = False
                    break
            if f_4 == True:
                path_list.append(4)
            #path_list.append(4)
            f_4 = False
            f_1,f_2,f_3,f_5,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 802 and list_y[i] < 836 and f_5 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 802 and list_y[j] < 836 and f_5 == True):
                    pass
                else:
                    f_5 = False
                    break
            if f_5 == True:
                path_list.append(5)
            #path_list.append(5)
            f_5 = False
            f_1,f_2,f_3,f_4,f_6,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 421 and list_x[i] < 536 and list_y[i] > 980 and list_y[i] < 1015 and f_6 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 421 and list_x[j] < 536 and list_y[j] > 980 and list_y[j] < 1015 and f_6 == True):
                    pass
                else:
                    f_6 = False
                    break
            if f_6 == True:
                path_list.append(6)
            #path_list.append(6)
            f_6 = False
            f_1,f_2,f_3,f_4,f_5,f_7,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True

        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 40 and list_y[i] < 210 and f_7 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 40 and list_y[j] < 210 and f_7 == True):
                    pass
                else:
                    f_7 = False
                    break
            if f_7 == True:
                path_list.append(7)
            #path_list.append(7)
            f_7 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_8,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 215 and list_y[i] < 380 and f_8 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 215 and list_y[j] < 380 and f_8 == True):
                    pass
                else:
                    f_8 = False
                    break
            if f_8 == True:
                path_list.append(8)
            #path_list.append(8)
            f_8 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_9,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 385 and list_y[i] < 555 and f_9 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 385 and list_y[j] < 555 and f_9 == True):
                    pass
                else:
                    f_9 = False
                    break
            if f_9 == True:
                path_list.append(9)
            #path_list.append(9)
            f_9 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_10,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 560 and list_y[i] < 725 and f_10 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 560 and list_y[j] < 725 and f_10 == True):
                    pass
                else:
                    f_10 = False
                    break
            if f_10 == True:
                path_list.append(10)
            #path_list.append(10)
            f_10 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_11,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 730 and list_y[i] < 900 and f_11 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if ( list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 730 and list_y[j] < 900 and f_11 == True):
                    pass
                else:
                    f_11 = False
                    break
            if f_11 == True:
                path_list.append(11)
            #path_list.append(11)
            f_11 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_12 = True,True,True,True,True,True,True,True,True,True,True
        if( list_x[i] > 1329 and list_x[i] < 1550 and list_y[i] > 705 and list_y[i] < 1075 and f_12 == True):
            n = i + thresh
            if n >= len(list_x):
                n = len(list_x)
            for j in range(i, n):
                if (list_x[j] > 1329 and list_x[j] < 1550 and list_y[j] > 705 and list_y[j] < 1075 and f_12 == True):
                    pass
                else:
                    f_12 = False
                    break
            if f_12 == True:
                path_list.append(12)
            #path_list.append(12)
            f_12 = False
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10,f_11 = True,True,True,True,True,True,True,True,True,True,True
        i = n
    plotpath(list_x,list_y,path_list)
    cal_matrix(path_list)
    path_list[:] = []
    return 0




def read_file(filename):
    file_to_read = open(filename + '.txt','r')
    count = 1
    list_x = []
    list_y = []
    list_time = []
    actual = -1
    act_str = ''
    while(count <= 40):
        line = file_to_read.readline()
        str =  line.split( )
        #if('start' in str):
        #    line = file_to_read.readline()
        #    str =  line.split( )
        while ('Stop' not in str):
            line = file_to_read.readline()
            str =  line.split( )
            if ('start' in str):
                continue
            if ('Yes' in str ):
                #plot_line(list_x[::count],list_y[::count],'Yes')
                calculate_heat_map(list_x[::count],list_y[::count],'Yes',count)
                calpath(list_x,list_y,list_time,count)
                list_x[:] = []
                list_y[:] = []
                list_time[:] = []
                continue
            if ('No' in str):
                #plot_line(list_x[::count],list_y[::count],'No')
                calculate_heat_map(list_x[::count],list_y[::count],'No',count)
                calpath(list_x,list_y,list_time,count)
                list_x[:] = []
                list_y[:] = []
                list_time[:] = []
                continue
            float_str = [float(i) for i in str]
            list_x.append(float_str[2])
            list_y.append(float_str[3])
            actual = float_str[1]
            #print float_str
        if '-1' in str:
            continue
        else:
            act_list.append(actual)
            count = count + 1
    #print ('total distance covered ')
    #print(t_dist_list)
    #print ('total distance covered in x direction')
    #print(dx_dist_list)
    #print ('total distance covered in y direction')
    #print(dy_dist_list)
    #draw_heat_map()
    stats()
    #plt.show()
    file_to_read.close()



def write_on_result(exp_num,output_file):
    #print (len(std_trans_list))
    for i in range(0,40):
        trial_num = i + 1
        trial_typ = ''
        if act_list[i] < 20:
            trial_typ = '1'
        elif act_list[i] >= 20 and act_list[i] < 28:
            trial_typ = '4'
        elif act_list[i] >= 28 and act_list[i] < 34:
            trial_typ = '2'
        else:
            trial_typ = '3'

        output_file.write(str(exp_num) + ',' + str(trial_num) + ',' + heat_title_label[i] + ',' +
                          str(skew_list[i]) + ',' +
                          str(kurt_list[i]) + ',' +
                          str(mean_list[i]) + ',' +
                          str(var_list[i]) + ',' + str(std_list[i]) + ',' + str(max_index_list[i]) + ',' +
                          str(sum(heat_map_list[ i * 12 : i * 12 + 12])) + ',' +
                          str(t_dist_list[i]) + ',' + str(dx_dist_list[i]) + ',' + str(dy_dist_list[i]) + ',' +
                          str(mean_trans_list[i]) + ',' + str(var_trans_list[i]) + ','+ str(max_trans_index_list[i]) + ',' +
                          str(skew_trans_list[i]) + ',' + str(kurt_trans_list[i]) + ',' +
                          str(std_trans_list[i]) + ',' +
                          trial_typ + '\n')


def start_file_read():
    main_file = open('filename.txt','r')
    output_file = open('final-output.csv','w')
    line_to_write = ['Output file needed for ML task\n','columns representation is as follows: -\n',
                     '1. Experiment Number\n',
                     '2. Trial Number\n',
                     '3. User Judgement of the GUI Yes (Congruent), No (Not Congruent)\n',
                      # time based calculations
                     '4. skew of the time spend in each trial\n',
                     '5. Kurtosis of the time spend in each trial\n'
                     '6. Mean of the time spend in each trial\n',
                     '7. Variance of the time spend in each trial\n',
                     '8. Standard Deviation of the time spend in each trial\n',
                     '9. Block number on which maximum amount of time is spend (1 - 6 Labels and 7 - 12 pictures)\n',
                     '10. Total time spend in one trial\n',
                     '11. Total gaze distance\n',
                     '12. Total gaze distance in X - axis\n',
                     '13. Total gaze distance in Y - axis\n',
                      # matrix based calculations
                     '14. Mean of the total Transition \n',
                     '15. Variance of the total transition\n',
                     '16. Block number on which maximum transition are made (1 - 6 Labels and 7 - 12 pictures)\n',
                     '17. Skew of the total transition in a trial\n',
                     '18. Kurtosis of the total transition in a trial\n',
                     '19. Standard Deviation of the total transition in a trial\n',
                     '20. Trial Type C (Correct) = 1, W (Wrong) = 2, S (Shuffle) = 3, P (one Permutation) = 4\n',
                     '\n*****************************************************************\n']
    output_file.writelines(line_to_write)
    output_file.write('exp,trial,userjudgement,time_skew,time_kurtosis,time_mean,time_variance,t_sd,time_maxblock,timefortrial,gazedistall,gazedistX,gazedistY,trans_mean,trans_var,trans_maxblock,trans_skew,trans_kurtosis,trans_sd,correctlabel\n')
    exp_num = 0
    for line in main_file:
        exp_num += 1
        print(line)
        read_file(line.strip())
        write_on_result(exp_num,output_file)
        heat_map_list[:] = [0] * 12 * 40
        heat_title_label[:] = []
        act_list[:] = []
        dx_dist_list[:] = []
        dy_dist_list[:] = []
        t_dist_list[:] = []
        mean_list[:] = []
        var_list[:] = []
        max_list[:] = []
        max_index_list[:] = []
        skew_list[:] = []
        kurt_list[:] = []
        std_list[:] = []
        mean_trans_list[:] = []
        var_trans_list[:] = []
        max_trans_index_list[:] = []
        skew_trans_list[:] = []
        kurt_trans_list[:] = []
        std_trans_list[:] = []
    main_file.close()
    output_file.close()



def main():
    start_file_read()

if __name__ == '__main__':
    main()
