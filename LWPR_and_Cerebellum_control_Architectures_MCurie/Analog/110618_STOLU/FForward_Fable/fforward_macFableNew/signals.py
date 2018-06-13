'''
Created on May 11, 2016

@author: stolu
'''
import math
import matplotlib.pyplot as plt

#change amplitude
class Sinampl(object):

    def __init__(self, ampls, switch=15.0, w=2*math.pi, samplingfreq=100.0):
        self.ampls = ampls
        self.w = w
        self.sf = samplingfreq
        self.switch = switch
        self.t = 0.0
        self.i1 = 0
        self.A1 = ampls[self.i1]
        self.i2 = 0
        self.A2 = ampls[self.i2]


    def __call__(self):
        q1 = self.A1/(self.w*self.w)*math.sin(self.w*self.t)
        q2 = self.A2/(self.w*self.w)*math.sin(self.w*self.t+math.pi/2)
        self.t += 1.0/self.sf
        if self.t > (self.i1+1)*self.switch:
            self.i1 +=1
            if (self.i1 < len(self.ampls)):
                self.A1 = self.ampls[self.i1]
        if self.t > (self.i2+1.25)*self.switch:
            self.i2 +=1
            if (self.i2 < len(self.ampls)):
                self.A2 = self.ampls[self.i2]
        return (q1, q2)


# ref: https://en.wikipedia.org/wiki/Chirp
#change frequency
class Chirp(object):

    def __init__(self, A, f0, k, w=2*math.pi, samplingfreq=100.0):
        self.A = A
        self.f0 = f0
        self.k = k
        self.w = w
        self.sf = samplingfreq
        self.t = 0.0

    def __call__(self):
        q1 = self.A/(self.w*self.w)*math.sin(self.w*(self.f0*self.t+self.k/2*self.t*self.t))
        q2 = self.A/(self.w*self.w)*math.sin(math.pi/2+self.w*(self.f0*self.t+self.k/2*self.t*self.t))
        self.t += 1.0/self.sf
        return (q1, q2)


if __name__ == '__main__':

    T = 10.0
    samplingfreq = 100.0

    #signal = Sinampl([500, 700, 900, 500],3, 2*math.pi, samplingfreq)
    signal = Chirp(500, 0.5, 0.1, 2*math.pi, samplingfreq)
    y1 = []
    y2 = []
    for i in range(int(T*samplingfreq)):
        y = signal()
        y1.append(y[0])
        y2.append(y[1])

    t = [x / samplingfreq for x in range(int(T*samplingfreq))]
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.show()