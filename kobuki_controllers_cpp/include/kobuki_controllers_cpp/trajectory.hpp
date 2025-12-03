#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
// #define M_PI 3.14159265358979323846

class Trajectory
{
private:
    
public:
    Trajectory();
    ~Trajectory() = default;
    
    /*
        Trajectory parameters:
        x_ref       = sin(theta/alpha)
        y_ref       = eta * cos(theta/(2*alpha)) 
        the_ref     = arctan2(/dot{yref}, /dot{xref})
        v_ref       = sqrt((/dot{xref})^2 + (/dot{yref})^2)
        w_ref       = (/dot{xref} * /ddot{yref} - /ddot{xref} * /dot{yref}) / ((/dot{xref})^2 + (/dot{yref})^2)
        npoints     = number of points in the trajectory (from 0 to theta_end at dt intervals)
    */
    
    double eta, alpha, theta_end, dt; // Trajectory parameters
    int n_points;                     // Number of points in the trajectory

    // Trajectory vectors
    std::vector<double> x_ref;
    std::vector<double> y_ref;
    std::vector<double> th_ref;
    std::vector<double> v_ref;
    std::vector<double> w_ref;

    // Trajectory functions
    void setTrajectory(double _eta, double _alpha, double _dt, int _cycles = 1, std::vector<double> _init = {0., 0.}, std::string _type = "circle");
    std::vector<std::vector<double>> getTrajectory();
    void saveTrajectory(std::vector<std::vector<double>> _all = {},
                        std::vector<double> _x = {}, 
                        std::vector<double> _y = {}, 
                        std::vector<double> _th = {}, 
                        std::vector<double> _v = {}, 
                        std::vector<double> _w = {}, 
                        std::string filename = "reference.txt"); // To graph the trajectory reference or any other trajectory
    void setTrajectoryFromWaypoints(
        const std::vector<std::pair<double, double>>& waypoints,
        double _dt,
        double desired_velocity = 0.5,
        std::string interpolation_type = "cubic");

};

#endif