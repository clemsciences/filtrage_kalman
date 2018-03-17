# -*- coding: utf-8 -*-
from math import sqrt


__author__ = 'språkforskaren'


class Point:
    """
    Classe permettant de définir un point mathématique dans R^2 et les opérations usuelles sur les points dans R^2

    :param x: abscisse
    :type x: float

    :param y: ordonnée
    :type y: float

    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __rmul__(self, other):
        return Point(self.x*other, self.y*other)

    def __str__(self) :
        return "("+str(self.x)+"," + str(self.y) + ")"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def to_list(self):
        return [self.x, self.y]

    def unitaire(self):
        n = self.norme()
        return Point(self.x / n, self.y / n)

    def norme(self):
        return self.distance(Point(0, 0))

    def distance(self, point):
        v = self - point
        d = v.x ** 2 + v.y ** 2
        return sqrt(d)

    def copy(self):
        return Point(self.x, self.y)


class Velocity:
    """
    Classe permettant de définir une vitesse sous forme de vecteur mathématique dans R^2 et les opérations usuelles sur les vecteurs dans R^2

    :param vx: vitesse sur x
    :type vx: float

    :param vy: vitesse sur y
    :type vy: float

    """
    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def __repr__(self):
        return '(' + str(self.vx) + ',' + str(self.vy) + ')'

    def __add__(self,other):
        return Velocity(self.vx + other.vx, self.vy + other.vy)

    def __sub__(self,other):
        return Velocity(self.vx - other.vx, self.vy - other.vy)

    def __mul__(self,other):
        return Velocity(self.vx*other, self.vy*other)

    def __str__(self) :
        return "("+str(self.vx)+"," + str(self.vy) + ")"

    def norme(self):
        return sqrt(self.vx ** 2 + self.vy ** 2)

    def to_list(self):
        return [self.vx, self.vy]
