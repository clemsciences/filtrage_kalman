import math

__author__ = "spraakforskaren"


def is_in_table(point):
    """
    check if the point is in the table (for robotics cup)
    """
    x = point[0]
    y = point[1]
    return -1500 <= x <= 1500 and 0 <= y <= 2000


def get_cos_sin_vector(point1, point2):
    """
    gives the cos and sinues of the angle formed by point1 and point2
    :param point1: a couple [x, y]
    :param point2: a couple [x, y]
    :return: (cos(theta), sin(thesta) with theta the angle between point1 et point2
    """
    diff_y = point2[1] - point1[1]
    diff_x = point2[0] - point1[0]
    return float(diff_x) / math.sqrt(diff_x ** 2 + diff_y ** 2), float(diff_y) / math.sqrt(diff_x ** 2 + diff_y ** 2)


def get_absolute_angle(p1, p2):
    """

    :param p1: [x, y]
    :param p2: [x, y]
    :return: theta, angle between p1 and p2
    """
    diff_y = p2[1] - p1[1]
    diff_x = p2[0] - p1[0]
    return math.acos(float(diff_x) / math.sqrt(diff_x ** 2 + diff_y ** 2))


def point_reached(current_point, aim):
    """

    :param current_point: [x, y]
    :param aim: [x, y]
    :return: True if aim == current_point
    """
    return current_point[0] == aim[0] and current_point[1] == aim[1]


def distance(p1, p2):
    """
    distance between two points
    :param p1: [x, y]
    :param p2: [x, y]
    :return: distance between p1 and p2
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def generate_path(l_points, velocity_translation, velocity_rotation, dt):
    """

    :param l_points: list of points
    :param velocity_translation: velocity in mm/s
    :param velocity_rotation: velocity in rad/s
    :param dt: period of sampling
    :return: list of points
    """
    # the result
    l_detailed_points = []
    # checks if all the points given in l_points are in the table
    for point in l_points:
        assert is_in_table(point)
    N = len(l_points)
    # initialisation
    origin_point = l_points[0]
    next_point = l_points[1]
    origin_angle = get_absolute_angle(origin_point, next_point)
    current_point = origin_point
    current_angle = origin_angle
    for i in range(N - 2):
        # translation
        co, si = get_cos_sin_vector(origin_point, next_point)
        step_x = co * velocity_translation * dt
        step_y = si * velocity_translation * dt
        # print "step_x ", step_x
        while not point_reached(current_point, next_point):
            if distance(current_point, next_point) <= math.sqrt(step_x ** 2 + step_y ** 2):
                current_point[0] = next_point[0]
                current_point[1] = next_point[1]
            else:
                current_point[0] += step_x
                current_point[1] += step_y
            x = current_point[0]
            y = current_point[1]
            l_detailed_points.append((x, y))
        # rotation
        origin_point = l_points[i + 1]
        next_point = l_points[i + 2]
        next_angle = get_absolute_angle(origin_point, next_point)
        delta_angle = next_angle - origin_angle
        step_angle = velocity_rotation * dt
        while current_angle != next_angle:
            # print current_angle
            if next_angle - current_angle <= step_angle:
                current_angle = next_angle
            else:
                current_angle += step_angle
            x = current_point[0]
            y = current_point[1]
            l_detailed_points.append((x, y))
        origin_angle = current_angle
    # last point, last translation
    co, si = get_cos_sin_vector(origin_point, next_point)
    current_point = origin_point
    step_x = co * velocity_translation * dt
    # print "step_x ", step_x
    step_y = si * velocity_translation * dt
    while not point_reached(current_point, next_point):
        if distance(current_point, next_point) <= math.sqrt(step_x ** 2 + step_y ** 2):
            current_point[0] = next_point[0]
            current_point[1] = next_point[1]
        else:
            current_point[0] += step_x
            current_point[1] += step_y
        x = current_point[0]
        y = current_point[1]
        l_detailed_points.append((x, y))
    return l_detailed_points


if __name__ == "__main__":
    l_points = [[200., 200.], [800., 700.], [-1000., 1250.], [-1200., 1500.]]
    velocity_translation = 200
    velocity_rotation = 1.52
    dt = 0.05
    for p in generate_path(l_points, velocity_translation, velocity_rotation, dt):
        print(p)
