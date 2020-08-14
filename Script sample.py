#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Ship Rendezvous Problem

This code will:

1. Read a problem instance (data) from a CSV file;
2. Run the greedy heuristic against the problem instance to obtain a solution.
3. Output the resulting path to a CSV file.
4. Output key performance indicators of the solution to a second CSV file.

"""

import math
import numpy as np


def main(csv_file):
    '''
    Main function for running and solving an instance of the SRP.

    Keyword arguments:
    csv_file -- the csv file name and path to a data for an instance of
                ship rendezvous problem.
    '''
    # read in the csv file to a numpy array
    problem_data = read_srp_input_data(csv_file)
    initial_data = [problem_data[0, 0], problem_data[0, 1]]

    # create an index and add to the problem data
    index = np.arange(-1, problem_data.shape[0]-1)
    problem_data = np.insert(problem_data, 0, index, axis=1)

    # calculate speed of the support ship
    ss_speed = math.sqrt(problem_data[0, 3]**2)

    # target is the solution array for the SRP
    target = final_tour(problem_data, ss_speed)
    filename = 'solution.csv'
    save_to_csv(filename, target)

    # calling the kpi function
    kpi(target, initial_data, ss_speed)

    print('\n === Final result written to solution csv file ===')
    print('\n === Key performance indicators written to kpi csv file ===')
    print (kpi)

def read_srp_input_data(csv_file):
    '''
    Problem instances for the SRP are stored within a .csv file
    This function reads the problem instance into Python.
    Returns a 2D np.ndarray (4 columns).
    Skips the headers and the first column.
    Columns are:
    x-coordinate, y-coordinate, x-speed and y-speed

    Keyword arguments:
    csv_file -- the file path and name to the problem instance file
    '''

    input_data = np.genfromtxt(csv_file, delimiter=',',
                               dtype=np.float64,
                               skip_header=1,
                               usecols=tuple(range(1, 5)))

    return input_data


def solve_quadratic_equation(x_0, y_0, x_1, y_1, vx1, vy1, ss_speed):
    '''
    Returns time taken for the support ship to move from its current position
    and intercept cruise ship (i). It is found by finding the smallest
    positive root of the quadratic equation.

    Keyword arguments:
    x0 and y0 -- Support ship coordinates
    x1 and y1 -- Cruise ship coordinates
    vx1 and vy1 -- Cruise ship velocity
    ss_speed -- speed of the support ship
    '''

    a = (vx1**2) + (vy1**2) - (ss_speed**2)
    b = 2*(vx1*(x_1-x_0)+vy1*(y_1-y_0))
    c = ((x_1-x_0)**2) + ((y_1-y_0)**2)

    if a == 0:
        if (-c/b) > 0:
            return -c/b
        else:
            return -1

    if ((b**2)-(4*a*c)) < 0:
        return -1

    ans1 = (-b - (math.sqrt((b**2)-(4*a*c)))) / (2*a)
    ans2 = (-b + (math.sqrt((b**2)-(4*a*c)))) / (2*a)

    if ans1 >= 0 and ans2 >= 0:
        return min(ans1, ans2)

    if ans2 < 0 <= ans1:
        return ans1

    if ans1 < 0 <= ans2:
        return ans2

    if ans1 < 0 and ans2 < 0:
        return -1


def intercept_times(problem_data, ss_speed):
    '''
    Returns in an array the time taken for the support ship to move from
    its current position and intercept all the cruise ships. Also notes the
    position of the minimum intercept time.

    Keyword arguments:
    problem_data -- array containing coordinates from the input file
    ss_speed -- speed of the support ship
    '''
    n_times = problem_data.shape[0]-1
    ship_times = np.zeros(n_times)
    x_0 = problem_data[0, 1]
    y_0 = problem_data[0, 2]

    # calculating the intercept times and storing them in the ship_times array
    for i in range(n_times):

        root_x = solve_quadratic_equation(x_0, y_0, problem_data[i+1][1],
                                          problem_data[i+1][2],
                                          problem_data[i+1][3],
                                          problem_data[i+1][4], ss_speed)

        if root_x >= 0:
            ship_times[i] = root_x

        else:
            ship_times[i] = np.inf

    # below variables record the minimum time and its position
    nearest_cs = min(ship_times)
    nearest = np.argmin(ship_times)

    # if 2 ships have the same minimum time, below sets the nearest position...
    # ...of the cruise ship with the highest Y coordinate. If this is also...
    # ...tied, python automatically picks the ship with the smallest index
    for i in range(nearest+1, n_times):
        if ship_times[i] == nearest_cs:
            if problem_data[i+1][2] > problem_data[nearest+1][2]:
                nearest = i

    return nearest, ship_times


def update(problem_data, ss_speed):
    '''
    Updates the coordinates of all the ships. The coordinates of the nearest
    ship are noted in an array and then deleted from the problem_data to avoid
    visiting it again.

    Keyword arguments:
    problem_data -- array containing coordinates from the input file
    ss_speed -- speed of the support ship
    '''
    n_times = problem_data.shape[0]-1
    ship_order = np.full([n_times, 4], -1.0)
    time = 0

    for j in range(n_times):

        nearest_ship, s_times, = intercept_times(problem_data, ss_speed)
        cs_speed = math.sqrt((problem_data[nearest_ship+1][3]**2) +
                             (problem_data[nearest_ship+1][4]**2))

        # checks whether the support ship is faster than the cruise ship.
        # if not, the cruise ship cannot be visited and that row is deleted.
        # n_times is also updated in each loop to ensure correct number of...
        # ...loops are performed.

        if cs_speed < ss_speed:

            for i in range(n_times):
                # updating cruise ship coordinates
                problem_data[i+1][1] = problem_data[i+1][1]+(min(s_times)
                                                             * problem_data
                                                             [i+1][3])
                problem_data[i+1][2] = problem_data[i+1][2]+(min(s_times)
                                                             * problem_data
                                                             [i+1][4])
            # updating support ship coordinates
            problem_data[0, 1] = problem_data[nearest_ship+1][0:3][1]
            problem_data[0, 2] = problem_data[nearest_ship+1][0:3][2]

            # calculating the running total of intercept times and adding it...
            # ...to the array where visited ship coordinates were stored.
            next_target = problem_data[nearest_ship+1][0:3]
            time += min(s_times)
            next_target = np.append(next_target, time)

            problem_data = np.delete(problem_data, nearest_ship+1, axis=0)
            n_times = problem_data.shape[0]-1

            ship_order[j] = next_target

        else:

            problem_data = np.delete(problem_data, nearest_ship+1, axis=0)
            n_times = problem_data.shape[0]-1

    return ship_order


def final_tour(problem_data, ss_speed):
    '''
    Here the function shifts any unvisited ship to the end of the array.

    Keyword arguments:
    problem_data -- array containing coordinates from the input file
    ss_speed -- speed of the support ship
    '''
    ship_order = update(problem_data, ss_speed)
    size = len(ship_order)
    count = 0
    j = 0

    # a count is performed of the ships visited based on which an array is...
    # ...created.
    for i in range(size):

        if ship_order[i][0] != -1:
            count += 1
    final_path = np.zeros([count, 4])

    # all ships visited are added to the final_path array.
    for i in range(size):

        if ship_order[i][0] != -1:
            final_path[j] = ship_order[i]
            j += 1

    # an array of unvisited ships is created with -1 values and is added...
    # ...to the end of the final_path array.
    not_reached = np.full([size-count, 4], -1)
    final_path = np.concatenate((final_path, not_reached))

    return final_path


def kpi(target, initial_data, ss_speed):
    """
    This function calculates the different key performance indicators.

    Keyword arguments:
    target -- the final order of cruise ships to be visited
    initial_data -- slice of the original input data
    ss_speed -- speed of the support ship
    """
    # Ships visited
    n_ships = 0

    for i in range(len(target)):
        if target[i][0] >= 0:
            n_ships += 1
        else:
            continue
    if n_ships == 0:
        n_ships = -1

    # Total time for the support ship to complete its tour
    # adds the last ships intercept time with the time it takes to return...
    # ...to the initial coordinates.
    if n_ships > 0:
        total_time = target[n_ships-1, 3] + (solve_quadratic_equation
                                             (initial_data[0],
                                              initial_data[1],
                                              target[n_ships-1, 1],
                                              target[n_ships-1, 2], 0,
                                              0, ss_speed))
        max_wait = target[n_ships-1, 3]
    else:
        total_time = -1
        max_wait = -1

    # Furthest Y coord.
    if n_ships > 0:
        y_coord = np.full(n_ships, -1.0)

        for i in range(len(target)):
            if target[i][0] >= 0:
                y_coord[i] = target[i][2]

        max_y = max(y_coord)

    else:
        max_y = -1

    # Furthest distance - distance for each ship visited is calculated and...
    # ...stored in a new array from which the maximum is returned.
    fur_dist = np.zeros(len(target))

    for i in range(len(target)):
        if target[i][0] >= 0:
            fur_dist[i] = math.sqrt((initial_data[0] - target[i][1])**2 +
                                    (initial_data[1] - target[i][2])**2)
        else:
            if n_ships > 0:
                fur_dist[i] = -np.inf
            else:
                fur_dist[i] = -1

    furthest_distance = max(fur_dist)

    # Avg. waiting time
    if n_ships >= 0:
        avg_time = sum(target[:, 3][:n_ships])/n_ships
    else:
        avg_time = -1

    # Below code writes the output to a csv file
    _kpi = n_ships, total_time, max_wait, max_y, furthest_distance, avg_time
    np.savetxt("kpi.csv", _kpi, delimiter=',')
    print (_kpi)

def save_to_csv(filename, output):
    """
    This function wrties the final solution to a csv file.

    Keyword arguments:
    filename and output
    """
    np.savetxt(filename, output, delimiter=',',
               header='Ship_index,interception_x_coordinate,'
               'interception_y_coordinate,estimated_time_of_interception'
               , comments=" ")


if __name__ == '__main__':
    PROBLEM_FILE = 'sample_srp_data.csv'
    main(PROBLEM_FILE)
