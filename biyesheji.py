import numpy as np
from random import random
from gekko import GEKKO
import matplotlib.pyplot as plt
import tclab
import time
import pymysql
from itertools import chain
import xlrd
import requests as r
def get_uty():
    book = xlrd.open_workbook('E:\\学习资料\\大四上\\科研\\apm\\1.1实验.xlsx')
    sheet1 = book.sheets()[0]
    T_measure = sheet1.col_values(1)
    time0 = sheet1.col_values(0)
    u=sheet1.col_values(2)
    return time0,u,T_measure
    # for i in range(300):
    #     T_measure.append([sheet1.cell(i,0).value+273,sheet1.cell(i,1).value+273])

class control():
    def __init__(self,t=0,u=0,y=0):
        self.t = t
        self.u = u
        self.y = y
    def mysql_link(self,db_name):
        try:
            db = pymysql.connect(host='rm-2ze77qng1ddlfur9g4o.mysql.rds.aliyuncs.com', port=3306, user='xajd',password='Xajd123#', database=db_name, charset='utf8')
            #print("Have connected to mysql server !")
            return db
        except:
            print("Could not connect to mysql server !")

    # 操作数据库中的数据
    def mysql_query(self,db_name,sql):
        db=self.mysql_link(db_name)              #链接数据库
        cursor = db.cursor()                #获取操作游标
        try:
            cursor.execute(sql)             #执行sql语句
            resultlist = list(chain.from_iterable(cursor))#获取查询结果 使用了itertools包中的chain，因此获得的数据为列表形式
            # results = cursor.fetchall()     #获取查询结果 若使用此行代码并且返回值为resuits，则获得的数据为元组，不利于后续处理
            # db.commit()                     #若执行的sql语句为插入命令，则必须执行此行代码用来在数据库中进行更新
        except Exception as e:
            db.rollback()                   #若出现错误则回滚
        finally:
            cursor.close()                  #关闭游标
            db.close()                      #关闭连接
        return resultlist
        # return results

    def identification(self):
        """time (t),input (u),output(yp)"""
    
        m = GEKKO(remote=False)
        m.time = self.t; time = m.Var(0); m.Equation(time.dt()==1)

        K = m.FV(2,lb=0,ub=100);      K.STATUS=1
        tau = m.FV(3,lb=0,ub=200);  tau.STATUS=1
        theta = m.FV(0,lb=0,ub=300); theta.STATUS=1

        # create cubic spline with t versus u
        uc = m.Var(self.u); tc = m.Var(self.t); m.Equation(tc==time-theta)
        m.cspline(tc,uc,t,u,bound_x=False)

        ym = m.Param(self.y); yp = m.Var(self.y)
        m.Equation(tau*yp.dt()+(yp-y[0])==K*(uc-u[0]))

        m.Minimize((yp-ym)**2)

        m.options.IMODE=5
        m.solve()

        print('Kp: ', K.value[0])
        print('taup: ',  tau.value[0])
        print('thetap: ', theta.value[0])

      
        self.kp = K.value[0]
        self.taup = tau.value[0]
        self.thetap = theta.value[0]

        Kp = self.kp 
        taup = self.taup
        thetap = self.thetap

        

        tauc = max(0.1*taup,0.8*thetap)
        Kc = (1/Kp)*(taup+0.5*thetap)/(tauc+0.5*thetap)
        tauI = taup + 0.5*thetap
        tauD = taup*thetap / (2*taup+thetap)

        # Parameters in terms of PID coefficients
        KP = Kc
        KI = Kc/tauI
        KD = Kc*tauD
        print ("kp:"+str(KP)+"  ki:"+str(KI)+"  kd:"+str(KD))

    def control_val(self,target=20.0,device='aa'):
        # Kp = self.kp 
        # taup = self.taup
        # thetap = self.thetap

        

        # tauc = max(0.1*taup,0.8*thetap)
        # Kc = (1/Kp)*(taup+0.5*thetap)/(tauc+0.5*thetap)
        # tauI = taup + 0.5*thetap
        # tauD = taup*thetap / (2*taup+thetap)

        # # Parameters in terms of PID coefficients
        # KP = Kc
        # KI = Kc/tauI
        # KD = Kc*tauD
        db_name="xjturl"
        sql="SELECT  sampleData FROM `history` WHERE Imei = 868474043212842; "    #编写此行sql语句，要求使用双引号里面的内容即可在终端中查询到相应数据，前面的两个函数可以认为是格式，整个的核心（python与mysql之间的桥梁）还是这个sql,关于sql的详细内容可以看我的pdf
        res=self.mysql_query(db_name,sql)
        length_last = len(res)
        length_now = 0
        KP = 10.5
        KI = 0.1
        KD = 218
  
        value_last=float(res[-1][0:3])
        value_now=float(res[-1][0:3])
        error = target-value_now
        ierror=0.0
        control_value=0.0

        control_high=100.0
        control_low=0.0

        while (1):
            value_last=value_now
            res=self.mysql_query(db_name,sql)
            value_now = float(res[-1][0:3])
            length_last = length_now
            length_now = len(res)
            
            if length_now == length_last:
                print("温度还没更新")
                time.sleep(60)
                continue

            error=target-value_now
            ierror = ierror + error
            control_value=control_high + KP * error + KI * ierror + KD * (value_now-value_last)

            if control_value < control_low or control_value > control_high:
                ierror = ierror - error 
                control_value = max(control_low,min(control_high,control_value))
            
            if (control_value>90):
                control_value=int(10)
            elif (control_value>80 and control_value <= 90):
                control_value=int(9)
            elif (control_value>70 and control_value <= 80):
                control_value=int(8)
            elif (control_value>60 and control_value <= 70):
                control_value=int(7)
            elif (control_value>50 and control_value <= 60):
                control_value=int(6)
            elif (control_value>40 and control_value <= 50):
                control_value=int(5)
            elif (control_value>30 and control_value <= 40):
                control_value=int(4)
            elif (control_value>20 and control_value <= 30):
                control_value=int(3)
            elif (control_value>10 and control_value <= 20):
                control_value=int(2)
            else:
                control_value=int(1)
   
            print("控制变量的值为" + str(control_value))
            device='864016058149890'
            a = r.get('http://101.201.66.125:8797/api/controlValve?Imei=864016058149890&crack='+str(control_value)+'&autoControl=0')
            print(a.text)

            time.sleep(60)





    def pid(self,sp,pv,pv_last,ierr,dt):
        # Parameters in terms of PID coefficients
        KP = Kc
        KI = Kc/tauI
        KD = Kc*tauD
        # ubias for controller (initial heater)
        op0 = 0
        # upper and lower bounds on heater level
        ophi = 100
        oplo = 0
        # calculate the error
        error = sp-pv
        # calculate the integral error
        ierr = ierr + KI * error * dt
        # calculate the measurement derivative
        dpv = (pv - pv_last) / dt
        # calculate the PID output
        P = KP * error
        I = ierr
        D = -KD * dpv
        op = op0 + P + I + D
        # implement anti-reset windup
        if op < oplo or op > ophi:
            I = I - KI * error * dt
            # clip output
            op = max(oplo,min(ophi,op))
        # return the controller output and PID terms
        return [op,P,I,D]

    def pid_cal(self):
        Kp = self.kp 
        taup = self.taup
        thetap = self.thetap

        # -----------------------------
        # Calculate Kc,tauI,tauD (IMC Aggressive)
        # -----------------------------
        tauc = max(0.1*taup,0.8*thetap)
        Kc = (1/Kp)*(taup+0.5*thetap)/(tauc+0.5*thetap)
        tauI = taup + 0.5*thetap
        tauD = taup*thetap / (2*taup+thetap)
        n = 600  # Number of second time points (10 min)
        tm = np.linspace(0,n-1,n) # Time values
        lab = tclab.TCLab()
        T1 = np.zeros(n)
        Q1 = np.zeros(n)
        # step setpoint from 23.0 to 60.0 degC
        SP1 = np.ones(n)*23.0
        SP1[10:] = 60.0
        Q1_bias = 0.0
        ierr = 0.0
        for i in range(n):
            # record measurement
            T1[i] = lab.T1

            # --------------------------------------------------
            # call PID controller function to change Q1[i]
            # --------------------------------------------------
            [Q1[i],P,ierr,D] = pid(SP1[i],T1[i],T1[max(0,i-1)],ierr,1.0)

            lab.Q1(Q1[i])
            if i%20==0:
                print(' Heater,   Temp,  Setpoint')
            print(f'{Q1[i]:7.2f},{T1[i]:7.2f},{SP1[i]:7.2f}')
            # wait for 1 sec
            time.sleep(1)
        lab.close()
        # Save data file
        data = np.vstack((tm,Q1,T1,SP1)).T
        np.savetxt('PID_control.csv',data,delimiter=',',\
                header='Time,Q1,T1,SP1',comments='')

        # Create Figure
        # plt.figure(figsize=(10,7))
        # ax = plt.subplot(2,1,1)
        # ax.grid()
        # plt.plot(tm/60.0,SP1,'k-',label=r'$T_1$ SP')
        # plt.plot(tm/60.0,T1,'r.',label=r'$T_1$ PV')
        # plt.ylabel(r'Temp ($^oC$)')
        # plt.legend(loc=2)
        # ax = plt.subplot(2,1,2)
        # ax.grid()
        # plt.plot(tm/60.0,Q1,'b-',label=r'$Q_1$')
        # plt.ylabel(r'Heater (%)')
        # plt.xlabel('Time (min)')
        # plt.legend(loc=1)
        # plt.savefig('PID_Control.png')
        # plt.show()

    def MPC():
        m = GEKKO()
        #time
        m.time = np.linspace(0,20,41)
        #constants
        mass = 500
        #Parameters
        b = m.Param(value=50)
        K = m.Param(value=0.8)
        #Manipulated variable
        p = m.MV(value=0, lb=0, ub=100)
        #Controlled Variable
        v = m.CV(value=0)
        #Equations
        m.Equation(mass*v.dt() == -v*b + 2*K*b*p)
        #% Tuning
        #global
        m.options.IMODE = 6 #control
        #MV tuning
        p.STATUS = 1 #allow optimizer to change
        p.DCOST = 0.1 #smooth out gas pedal movement
        p.DMAX = 20 #slow down change of gas pedal
        #CV tuning
        #setpoint
        v.STATUS = 1 #add the SP to the objective
        m.options.CV_TYPE = 2 #L2 norm
        v.SP = 60 #set point
        v.TR_INIT = 1 #setpoint trajectory
        v.TAU = 5 #time constant of setpoint trajectory
        #% Solve
        m.solve(disp=False)
        print(len(v.value))
        #% Plot solution
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(m.time,p.value,'b-',LineWidth=2)
        plt.ylabel('gas')
        plt.subplot(2,1,2)
        plt.plot(m.time,v.value,'r--',LineWidth=2)
        plt.ylabel('velocity')
        plt.xlabel('time')
        plt.show()

# # process model
# Kp = 0.9
# taup = 175.0
# thetap = 15.0


if __name__ == '__main__':
    # t,u,y = get_uty()
    # print(t)
    # print(u)
    # print(y)
    # a=control()
    # a.control_val()
    a = r.get('http://101.201.66.125:8797/api/controlValve?Imei=864016058439267&crack='+str(10)+'&autoControl=0')
    print(a.text)



    

