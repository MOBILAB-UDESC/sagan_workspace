/*
    CODE BASED ON https://www.youtube.com/watch?v=HxxX7Uir2H8&t=844s
*/

#include <casadi/casadi.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include "trajectory.hpp"

using namespace casadi;

class MPC{
    public:

        /**
         * @brief        Constructor for the MPC class.
         * @param N    : Prediction horizon
         * @param dt   : Time step
         * @param u_min: Minimum control input
         * @param u_max: Maximum control input
         * @param z_min: Minimum state value
         * @param z_max: Maximum state value
         * @param Q    : State cost weight
         * @param R    : Control cost weight
        */
        MPC(int N, double dt,
            std::vector<double> u_min,
            std::vector<double> u_max,
            std::vector<double> z_min,
            std::vector<double> z_max,
            std::vector<std::vector<double>> Q,
            std::vector<std::vector<double>> R);

        /**
         * @brief  Set up the constant arguments for the MPC problem and check dimensions
         * @return true if there is no dimension mismatch, false otherwise.
        */
        bool setup_constant_mpc();

        /**
         * @brief           Set up the variable arguments for the MPC problem
         * @details         This function is called in each MPC iteration to update the MPC problem
         * @param mpc_iter: Current MPC iteration
         * @param z_init  : Current state
         * @param Z_init  : Warm-start values for the states (from the previous MPC solution)
         * @param U_init  : Warm-start values for the control input (from the previous MPC solution)
        */
        void setup_variable_mpc(int mpc_iter, std::vector<double> z_init, std::vector<std::vector<double>> Z_init, std::vector<std::vector<double>> U_init);

        /**
         * @brief           Add static circular obstacles
         * @param obstacle: 3-element vector: {center_x, center_y, radius} of the obstacle.
        */
        void setup_obstacles(std::vector<double> obstacle);

        /**
         * @brief  Solve the MPC optimization problem 
         * @return A pair containing the optimized state trajectory (Z_opt) and the optimized control input sequence (U_opt) over the MPC horizon.
        */
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> solve_mpc();

        /**
         * @brief    Get the next state
         * @param T: time step
         * @param X: Optimized state trajectory from the MPC solution
         * @param U: Optimized control input sequence from the MPC solution
        */
        // void shift(double T, double& t0, std::vector<double>& x0, std::vector<std::vector<double>>& U, Function& f);
        void shift(double T, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& U);

        // CasADi Function to actualize the system state: x_next = f(x, u)
        Function f;

        // Vector of reference values
        // Define the desired trajectory
        std::vector<std::vector<double>> Z_ref;

        int n_obstacles     = 0;     // Number of obstacles
        double obstacle_tol = 0.05;  // Tolerance for obstacle avoidance in meters
        double robot_radius = 0.178; // Radius of the robot in meters

        std::vector<std::vector<double>> obstacles; // {{xobs1, yobs1, robs1}, {xobs1, yobs1, robs1}, ...}

    private:

        // MPC parameters
        int N;                              // Prediction horizon
        double dt;                          // time step
        std::vector<std::vector<double>> Q; // State cost weight
        std::vector<std::vector<double>> R; // Control cost weight

        // Control inputs and states bounds
        std::vector<double> u_min;
        std::vector<double> u_max;
        std::vector<double> z_min;
        std::vector<double> z_max;

        uint8_t nz; // Number of states
        uint8_t nu; // Number of control inputs

        // CasADi variables
        Function solver; // CasADi solver function
        DMDict args;     // Dictionary containing all arguments required by the solver
        
        /**
         * @brief           Set the reference trajectory for the MPC problem
         * @details         Update the reference based on the current MPC iteration (N first values)
         * @param mpc_iter: Index of the current point of the reference
         */
        void set_reference(int mpc_iter);

};