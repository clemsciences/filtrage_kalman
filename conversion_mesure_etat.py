#-*- coding: utf-8 -*-
from math import sqrt
import numpy as np
from structures import Point, Velocity
__author__ = "spraakforskaren"


class Converter:
    """
    Les équations ont été résolues pour les positions suivantes
    balise 1 (-1500,0)
    balise 2 (1500,1000)
    balise 3 (-1500,2000)
    """
    l = 2  # largeur de la table
    L = 3  # longueur de la table

    def __init__(self, pos1=None, pos2=None, pos3=None):
        self.set_beacon_positions(pos1, pos2, pos3)

    def set_beacon_positions(self, pos1=None, pos2=None, pos3=None):
        L = Converter.L
        l = Converter.l
        STANDARD_POSITION1 = Point(-L/2., 0)
        if pos1 is None:
            self.beacon1 = STANDARD_POSITION1
        else:
            if isinstance(pos1, Point):
                self.beacon1 = pos1
            elif type(pos1) == list:
                self.beacon1 = Point(pos1[0], pos1[1])
            else:
                self.beacon1 = STANDARD_POSITION1

        STANDARD_POSITION2 = Point(L/2., l/2.)
        if pos2 is None:
            self.beacon2 = STANDARD_POSITION2
        else:
            if isinstance(pos1, Point):
                self.beacon2 = pos2
            elif type(pos1) == list:
                self.beacon2 = Point(pos2[0], pos2[1])
            else:
                self.beacon2 = STANDARD_POSITION2

        STANDARD_POSITION3 = Point(-L/2., l)
        if pos3 is None:
            self.beacon3 = STANDARD_POSITION3
        else:
            if isinstance(pos1, Point):
                self.beacon3 = pos3
            elif type(pos1) == list:
                self.beacon3 = Point(pos3[0], pos3[1])
            else:
                self.beacon3 = STANDARD_POSITION3

    def test(self, mesures):
        print("OOOOOOOOOOOOOOOOOOOOOOOO")
        for var in [0.001, 0.01, 0.1]:
            for (x,y) in mesures:
                print(x, y, var)
                #conversion
                m1, m2, m3 = self.get_measures_from_state(x, y)
                #bruitage
                m1, m2, m3 = m1 + np.random.randn()*var, m2 + np.random.randn()*var, m3 + np.random.randn()*var
                print(m1*1000, m2*1000, m3*1000)
                print(self.equation1_1_2(m1, m2))
                print(self.equation2_1_2(m1, m2))
                print(self.equation1_2_3(m2, m3))
                print(self.equation2_2_3(m2, m3))
                print(self.equation1_3_1(m3, m1))
                print(self.equation2_3_1(m3, m1))
                print("-------------------")

    def get_measures_from_state(self, x, y):
        """
        (x,y) sont les coordonnées du robot
        """
        point = Point(x, y)
        d1 = point.distance(self.beacon1)
        d2 = point.distance(self.beacon2)
        d3 = point.distance(self.beacon3)
        # d1 = sqrt((L/2.-x)**2+(l/2.-y)**2)
        # d2 = sqrt((-L/2.-x)**2+(l-y)**2)
        # d3 = sqrt((-x-L/2.)**2+y**2)
        return d1-d2, d1-d3, d2-d3
    
    def equation1_1_2(self, m1, m2):
        m1 = float(m1)
        m2 = float(m2)
        a = sqrt(-(-10+m1**2)*(m1+m2)**2*(-10+m2**2)*(-4+m1**2-2*m1**2+m2**2))

        x = -((-12+6*m1**2-12*m1*m2+3*m1**3*m2+6*m2**2-6*m1**2*m2**2+3*m1*m2**3+a)/(4*(-18+5*m1**2-8*m1*m2+5*m2**2)))
        y = -(1./(4*(m1+m2)*(-18+5*m1**2-8*m1*m2+5*m2**2)))*(72*m1-28*m1**3+72*m2+4*m1**2*m2+m1**4*m2+20*m1*m2**2+m1**3*m2**2-12*m2**3-m1**2*m2**3-m1*m2**4-3*m1*a+3*m2*a)
        return x, y

    def equation2_1_2(self, m1, m2):
        m1 = float(m1)
        m2 = float(m2)
        a = sqrt(-(-10+m1**2)*(m1+m2)**2*(-10+m2**2)*(-4+m1**2-2*m1*m2+m2**2))

        x = -((-12+6*m1**2-12*m1*m2+3*m1**3*m2+6*m2**2-6*m1**2*m2**2+3*m1*m2**3-a)/(4*(-18+5*m1**2-8*m1*m2+5*m2**2)))

        y = -(1/(4*(m1+m2)*(-18+5*m1**2-8*m1*m2+5*m2**2)))*(72*m1-28*m1**3+72*m2+4*m1**2*m2+m1**4*m2+20*m1*m2**2+m1**3*m2**2-12*m2**3-m1**2* m2**3-m1*m2**4+3*m1*a-3*m2*a)
        return x, y

    def equation1_2_3(self, m2, m3):
        m3 = float(m3)
        m2 = float(m2)
        a = sqrt(-(-10+m2**2)*(-2*m2+m3)**2*(-4+m3**2)*(-10+m2**2-2*m2*m3+m3**2))
        x =  -((-12+6*m3**2+3*m2**2*m3**2-3*m2*m3**3-a)/(4*(-18+2*m2**2-2*m2*m3+5*m3**2)))
        y = (-144*m2+16*m2**3+72*m3-56*m2**2*m3+4*m2**4*m3+80*m2*m3**2-8*m2**3*m3**2-28*m3**3+5*m2**2*m3**3-m2*m3**4+3*m3*a)/(4*(2*m2-m3)*(-18+2*m2**2-2*m2*m3+5*m3**2))
        return x, y

    def equation2_2_3(self, m2, m3):
        m3 = float(m3)
        m2 = float(m2)
        a = sqrt(-(-10+m2**2)*(-2*m2+m3)**2*(-4+m3**2)*(-10+m2**2-2*m2*m3+m3**2))
        x = -((-12+6*m3**2+3*m2**2*m3**2-3*m2*m3**3+a)/(4*(-18+2*m2**2 -2*m2*m3+5*m3**2)))
        y = (-144*m2+16*m2**3+72*m3-56*m2**2*m3+4*m2**4*m3+80*m2*m3**2-8*m2**3*m3**2-28*m3**3+5*m2**2*m3**3-m2*m3**4-3*m3*a)/(4*(2*m2-m3)*(-18+2*m2**2-2*m2*m3+5*m3**2))
        return x, y

    def equation1_3_1(self, m3, m1):
        m1 = float(m1)
        m3 = float(m3)
        a = sqrt(-(-10+m1**2)*(2*m1+m3)**2*(-4+m3**2)*(-10+m1**2+2*m1*m3+m3**2))
        x = -((-12+6*m3**2+3*m1**2*m3**2+3*m1*m3**3+a)/(4*(-18+2* m1**2+2*m1*m3+5*m3**2)))
        y = (-144*m1+16*m1**3-72*m3-8*m1**2*m3+4*m1**4*m3+16*m1*m3**2+8*m1**3*m3**2+12*m3**3+5*m1**2*m3**3+m1*m3**4-3*m3*a)/(4*(2*m1+m3)*(-18+2*m1**2+2*m1*m3+5*m3**2))
        return x, y

    def equation2_3_1(self, m3, m1):
        m1 = float(m1)
        m3 = float(m3)
        a = sqrt(-(-10+m1**2)*(2*m1+m3)**2*(-4+m3**2)*(-10+m1**2+2*m1*m3+m3**2))
        x = -((-12+6*m3**2+3*m1**2*m3**2+3*m1*m3**3-a)/(4*(-18+2*m1**2+2*m1*m3+5*m3**2)))
        y = (-144*m1+16*m1**3-72*m3-8*m1**2*m3+4*m1**4*m3+16*m1*m3**2+8*m1**3*m3**2+12*m3**3+5*m1**2*m3**3+m1*m3**4+3*m3*a)/(4*(2*m1+m3)*(-18+2*m1**2+2*m1*m3+5*m3**2))
        return x, y


if __name__ == "__main__":
    measures = [(0.3, 0.5), (-1.2, 1.5), (1.4, 1.7)]
    Converter().test(measures)
