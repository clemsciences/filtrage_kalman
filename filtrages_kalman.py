# -*- coding: utf-8 -*-

import pickle

from conversion_mesure_etat import *
from math import sqrt
import collections
import scipy.linalg
from scipy.stats import norm
import generateur_chemin
from structures import Point, Velocity


___author__ = "språkforskaren"


# norm.cpf


class Kalman:
    """
    Classe  qui implémente le filtre de Kalman. Utilisé, il permet de filtrer une suite discrétisé de valeur mesurée
    """
  
    def __init__(self, x, P, F, H, R, Q):
        """
        :param x: Etat initial du truc à suivre (un vecteur position le plus souvent)
        :param P: matrice incertitude sur le modèle d'évolution
        :param F: matrice de transition du modèle d'évolution
        :param H: matrice matrice d'observation
        :param R: matrice de covariance de la mesure
        :param Q: matrice de covariance du modèle

        """
        self.x = x
        self.P = P
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q
  
    def predict(self, u=None):
        """
        C'est la partie prédiction, le modèle imagine comment l'état suivant sera
        :param u: un vecteur "déplacement", si onsait de combien ça a dû bouger, il faut le mettre
        """
        if u is None:
            u = np.zeros(self.x.shape[0])[:, np.newaxis]
        self.x = np.dot(self.F, self.x) + u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def measure(self, mes):
        """
        C'est la partie où on prend en compte la nouvelle mesure
        :param mes: vecteur de même dimension que x
        """
        y = mes - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.identity(self.x.shape[0]) - np.dot(K, self.H)), self.P)

    def filter(self, mes, u=None):
        """
        Méthode qui condense une étape du filtrage
        """
        self.predict(u)
        self.measure(mes)


class FiltrageKalman:
    """
    Classe qui utilise Kalman
    """
    def __init__(self, x0, dt=0.025):
        """

        :param x0: état initial
        :param dt: pas de la mesure (période d'échantillonage)
        """
        self.dt = dt
        x = x0  # np.array([1400, 100, 0., 0.])[:, np.newaxis] # vecteur d'état au départ
        P = np.matrix([[30.0, 0., 0., 0.], [0., 30., 0., 0.], [0., 0., 10., 0.], [0., 0., 0., 10.]])  # incertitude initiale
        F = np.matrix([[1., 0., self.dt, 0.], [0., 1., 0., self.dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])  # matrice de transition
        H = np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.]])# matrice d'observation
        R = np.matrix([[900, 0.], [0., 900]]) # incertitude sur la mesure

        # Q = np.matrix([[self.dt**3/3., self.dt**2/2., 0, 0],[self.dt**2/2.,self.dt, 0, 0],
        #            [0,0,self.dt**3/3.,self.dt**2/2],[0,0,self.dt**2/2,self.dt]])
        # Q *= 20;

        Q = np.matrix([[self.dt**3/3., 0, self.dt**2/2., 0],[0, self.dt**3/3., 0, self.dt**2/2],
                       [self.dt**2/2., 0, 4*self.dt, 0],[0, self.dt**2/2, 0, 4*self.dt]])
        # Q = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 4, 0],[0, 0, 0, 4]])
        Q *= 30
        self.kalman_filter = Kalman(x, P, F, H, R, Q)
        self.history = collections.deque(maxlen=3)
        self.rejected_values = 0
        self.acceleration = None
        
    def get_current_state(self):
        return self.kalman_filter.x

    def get_current_position(self):
        state = self.get_current_state()
        return Point(float(state[0]), float(state[1]))
        
    def update_dt(self, new_dt):
        self.dt = new_dt
        self.kalman_filter.F[0,2] = new_dt
        self.kalman_filter.F[1,3] = new_dt
    
    def get_last_state(self):
        # state = self.filtre_kalman.x
        # return Point(int(state[0]), int(state[1]))
        return self.last_point
    
    # def vitesse(self):
    #     state = self.kalman_filter.x
    #     return Vitesse(int(state[2]), int(state[3]))
                
    def update(self, x, y):
        if self.acceleration_filtering(Point(x, y)):
            self.last_point = Point(x, y)
            self.kalman_filter.predict()
            self.kalman_filter.measure(np.array([x, y])[:, np.newaxis])
            self.history.append(self.get_last_state())
        else:
            self.last_point = None
            self.kalman_filter.predict()
        
    def acceleration_filtering(self, pointm0):
        """
        Vérifie si le point est cohérent avec la position actuelle, en se basant sur l'accélération
        """
        # Pas suffisamment de valeurs précédentes pour calculer l'accélération
        if len(self.history) != 3:
            return True
            
        # 3 derniers points valides
        pointm1 = self.history[2]
        pointm2 = self.history[1]
        pointm3 = self.history[0]
        
        # Vecteurs vitesses et accélération
        vitesse_actuelle = pointm0 - pointm1
        vitesse_m1 = pointm1 - pointm2
        vitesse_m2 = pointm2 - pointm3
        acceleration_actuelle = vitesse_actuelle - vitesse_m1
        acceleration_precedente = vitesse_m1 - vitesse_m2
        jerk = acceleration_actuelle - acceleration_precedente
        
        # Produit scalaire pour savoir s'il y a accélération ou décélération
        produit = acceleration_actuelle.x * vitesse_m1.x + acceleration_actuelle.y * vitesse_m1.y
        
        # Rejette les accélérations brutales
        if acceleration_actuelle.norme() / self.dt**2 > 50000 and self.rejected_values < 3:
            # ~ print("accélération = {0}, produit = {1}, jerk = {2}".format(acceleration_actuelle.norme() / self.dt**2,
            #  produit, jerk.norme() / self.dt**3))
            self.rejected_values += 1
            return False
        else:
            self.rejected_values = 0
            return True


def get_velocity(positions, dt):
    """

    :param positions: matrice des positions
    :param dt: période d'échantillonage
    :return:
    """
    velocities = np.zeros(positions.shape)
    for i in range(1, velocities.shape[0]):
        velocities[i, :] = (positions[i, :] - positions[i-1, :])/float(dt)
    return velocities


def get_distance(point1, point2):
    """

    :param point1: couple (x,y)
    :param point2: couple (x,y)
    :return:
    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_mer(real, estimated):
    """
    Renvoie la moyenne des distances
    :param real: liste de couples (x,y)
    :param estimated: liste de couples (x,y)
    :return:
    """
    try:
        for i in range(len(real)):
            print("réel ", real[i], " estimé ", estimated[i])
            print(real[i].distance(estimated[i]))
        res = [real[i].distance(estimated[i]) for i in range(len(real))]
        # print "La moyenne des distances entre les estimations et la réalité est : "
        return sum(res)/float(len(res))
    except ValueError:
        print(real[i])
        print(estimated[i])
    # except IndexError:
    #     print(i)
    #     print(len(real))
    #     print(len(estimated))


def squared_error(real_values, filtered_values):
    """
    Renvoie l'erreur quadratique moyenne !
    :param real_values:
    :param filtered_values:
    :return:
    """
    diff = real_values[:, :2] - filtered_values[:, :2]
    res = diff.T.dot(diff)
    "L'erreur quadratique est : "
    return res


def lire_fic_opti():
    valeurs = []
    coeff_s = []
    coeff_q = []
    coeff_r = []
    with open("mesures_simulees_10.txt", "r") as f:
        res = f.read()
    for r in res.split("\n"):
        print(r)
        if len(r) != 0:
            ligne = r.split("\t")
            valeurs.append(ligne[0])
            coeff_s.append(ligne[1])
            coeff_q.append(ligne[2])
            coeff_r.append(ligne[3])
    maxi = max(valeurs)
    print(maxi)
    ind = valeurs.index(maxi)
    print(ind)
    print("Q", coeff_q[ind])
    print("R", coeff_r[ind])
    print("S", coeff_s[ind])


def script_classic_trajectory_with_real_measures():
    """
    Script utilisant le filtre de Kalman  avec des mesures réelles !
    :return:
    """
    print("script_classic_trajectory_with_real_measures")
    dt=0.025
    measures_pos = np.genfromtxt("mesures_25.txt", delimiter="\t")
    real_path = []
    for i in range(measures_pos.shape[0]):
        x = measures_pos[i, 0]
        y = measures_pos[i, 1]
        real_path.append(Point(x, y))

    l_pos_filtre = [real_path[0]]
    vite = get_velocity(measures_pos, dt)
    measures = np.concatenate((measures_pos, vite), axis=1)
    measures = np.asmatrix(measures)
    filtering = FiltrageKalman(measures[0, :].T, dt=dt)
    var = 10
    for i in range(1, measures.shape[0]):
        x = measures[i, 0]
        y = measures[i, 1]
        # print("x et y", x, y, "i" ,i)
        x_bruite, y_bruite = x + np.random.randn()*var, y + np.random.randn()*var
        filtering.update(x_bruite, y_bruite)
        pos = filtering.get_current_position()
        l_pos_filtre.append(pos)
    # print(erreur_quadratique(measures, np.asmatrix(np.array(l_pos_filtre))))
    print(get_mer(real_path, l_pos_filtre))


def script_classic_trajectory():
    """
    Script utilisant le filtre de Kalman  avec des mesures simulées à partir d'une trajectoire inventée !
    :return:
    """
    print("script_classic_trajectory")
    l_points = [[-1000., 200.], [-1000., 800.], [-400., 1200.], [500., 500.], [1100., 180.]]
    dt = 0.025
    real_path = generateur_chemin.generate_path(l_points=l_points, velocity_translation=25,
                                                            velocity_rotation=0.7, dt=dt)
    measures_pos = np.array(real_path)
    real_path_point = []
    for couple in real_path:
        x, y = couple
        pos = Point(x, y)
        real_path_point.append(pos)

    l_pos_filtre = [real_path_point[0]]
    vite = get_velocity(measures_pos, dt)
    measures = np.concatenate((measures_pos, vite), axis=1)
    measures = np.asmatrix(measures)
    filtering = FiltrageKalman(measures[0, :].T, dt=dt)
    var = 10
    for i in range(1, measures.shape[0]):
        x = measures[i, 0]
        y = measures[i, 1]
        # print "x et y", x, y, "i" ,i
        x_bruite, y_bruite = x + np.random.randn()*var, y + np.random.randn()*var
        filtering.update(x_bruite, y_bruite)
        pos = filtering.get_current_position()
        l_pos_filtre.append(pos)
    print(get_mer(real_path_point, l_pos_filtre))


if __name__ == "__main__":
    script_classic_trajectory()
