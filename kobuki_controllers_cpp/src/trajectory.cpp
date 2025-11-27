#include "trajectory.hpp"

// Trajectory::Trajectory()
// {
//     eta         = 1.25;
//     alpha       = 5;
//     dt          = 0.06;
//     theta_end   = 2*M_PI*alpha*2;
//     n_points    = static_cast<int>(theta_end/dt);

//     for (int i = 0; i < n_points; ++i)
//     {
//         double theta = i * dt;
//         double x = 1 * sin(theta / alpha);
//         double y = 0.1 + eta * sin(theta / (2 * alpha));

//         x_ref.push_back(x);
//         y_ref.push_back(y);
//     }

//     std::vector<double> dx_ref(x_ref.size()), dy_ref(y_ref.size());
//     std::vector<double> ddx_ref(x_ref.size()), ddy_ref(y_ref.size());

//     for (size_t i = 1; i < x_ref.size()-1; ++i) {
//         dx_ref[i] = (x_ref[i+1] - x_ref[i-1]) / (2*dt);
//         dy_ref[i] = (y_ref[i+1] - y_ref[i-1]) / (2*dt);
//     }

//     dx_ref[0] = (x_ref[1] - x_ref[0]) / dt;
//     dy_ref[0] = (y_ref[1] - y_ref[0]) / dt;
//     // Backward difference for last point
//     dx_ref.back() = (x_ref.back() - x_ref[x_ref.size()-2]) / dt;
//     dy_ref.back() = (y_ref.back() - y_ref[y_ref.size()-2]) / dt;
    
//     for (size_t i = 0; i < dx_ref.size(); ++i) {
//         th_ref.push_back(std::atan2(dy_ref[i], dx_ref[i]));
//     }
    
//     // Second derivatives
//     for (size_t i = 1; i < dx_ref.size()-1; ++i) {
//         ddx_ref[i] = (dx_ref[i+1] - dx_ref[i-1]) / (2*dt);
//         ddy_ref[i] = (dy_ref[i+1] - dy_ref[i-1]) / (2*dt);
//     }
//     ddx_ref[0] = (dx_ref[1] - dx_ref[0]) / dt;
//     ddy_ref[0] = (dy_ref[1] - dy_ref[0]) / dt;
//     ddx_ref.back() = (dx_ref.back() - dx_ref[dx_ref.size()-2]) / dt;
//     ddy_ref.back() = (dy_ref.back() - dy_ref[dy_ref.size()-2]) / dt;
    
//     // Velocities
//     for (size_t i = 0; i < dx_ref.size(); ++i) {
//         v_ref.push_back(std::sqrt(dx_ref[i]*dx_ref[i] + dy_ref[i]*dy_ref[i]));
//         if (v_ref.back() > 0.001) {
//             w_ref.push_back((ddy_ref[i]*dx_ref[i] - ddx_ref[i]*dy_ref[i]) / 
//                             (dx_ref[i]*dx_ref[i] + dy_ref[i]*dy_ref[i]));
//         } else {
//             w_ref.push_back(0);
//         }
//     }
// }

Trajectory::Trajectory()
{    
}

std::vector<std::vector<double>> Trajectory::getTrajectory()
{
    std::vector<std::vector<double>> Zref;

    for (int i = 0; i<this->x_ref.size(); ++i)
    {
        std::vector<double> zref(5);
        zref[0] = this->x_ref[i];
        zref[1] = this->y_ref[i];
        zref[2] = this->th_ref[i];
        zref[3] = this->v_ref[i];
        zref[4] = this->w_ref[i];
        Zref.push_back(zref);
    }

    return Zref;
}

void Trajectory::setTrajectory(double _eta, double _alpha, double _dt, int _cycles, std::vector<double> _init, std::string _type)
{

    /*
        _eta    = Amplitud
        _alpha  = Variable to choose how many points the curve will have
        _dt     = Sample time
        _cycles = how many times the curve will be repeated
        _type   = which trajectory will be chosen
    */

    if(!_init.size())
    {
        _init = {0.0, 0.0};
    }

    this->eta         = _eta;
    this->alpha       = _alpha;
    this->dt          = _dt;
    this->theta_end   = 2*M_PI*this->alpha*_cycles;
    this->n_points    = static_cast<int>(this->theta_end/this->dt);

    this->x_ref.clear();
    this->y_ref.clear();
    this->th_ref.clear();
    this->v_ref.clear();
    this->w_ref.clear();

    if (_type == "line")
    {

        // Line
        for (int i = 0; i < this->n_points; ++i)
        {
            this->th_ref.push_back(M_PI/4);
            
            double x        = i * this->eta * cos(M_PI/4);
            double y        = i * this->eta * sin(M_PI/4);

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
        }
    }
    else if (_type == "u")
    {
        double a = 0;
        
        // First segment (upward)
        while (a < 6) {
            // this->th_ref.push_back(M_PI/2);
            
            double x        = a * cos(M_PI/2);
            double y        = a * sin(M_PI/2);

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
            a+=this->dt;
        }
        
        double radius = 0.5;
        double angle_start = M_PI - this->dt;
        double angle_end = M_PI/2 + this->dt;
        double angle = angle_start;

        while (angle > angle_end) {
            double x = 0.5 + radius * cos(angle);
            double y = 6 + radius * sin(angle);
            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
            angle -= this->dt / radius;
        }

        a = 0;

        // Second segment (horizontal)
        while (a < 8) {
            // this->th_ref.push_back(0);
            
            double x        = 0.5 + a * cos(0);
            double y        = 6.5 + a * sin(0);

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
            a+=this->dt;
        }

        angle_start = M_PI/2-this->dt;
        angle_end = this->dt;
        angle = angle_start;

        while (angle > angle_end) {
            double x = 8.5 + radius * cos(angle);
            double y = 6 + radius * sin(angle);
            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
            angle -= this->dt / radius;
        }

        a = 0;

        // Third segment (downward)
        while (a < 6) {
            // this->th_ref.push_back(-M_PI/2);
            
            double x        = 9 + a * cos(-M_PI/2);
            double y        = (a-6) * sin(-M_PI/2);

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
            a+=this->dt;
        }
    }
    else if (_type == "infinite")
    {
        for (int i = 0; i < this->n_points; ++i)
        {
            double theta    = i * this->dt;
            double x        = 1 * sin(2 * theta / this->alpha);
            double y        = 0.0 + eta * sin(theta / (this->alpha));

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
        }
    }
    else
    {
        for (int i = 0; i < this->n_points; ++i)
        {
            double theta    = i * this->dt;
            double x        = _init[0] + eta * sin(theta/this->alpha);
            double y        = _init[1] + eta * cos(theta/this->alpha);

            this->x_ref.push_back(x);
            this->y_ref.push_back(y);
        }
    }

    // for (int i = 0; i < this->n_points; ++i)
    // {
    //     double theta    = i * this->dt;
    //     double x        = 1 * sin(theta / this->alpha);
    //     double y        = 0.0 + eta * sin(theta / (2 * this->alpha));

    //     this->x_ref.push_back(x);
    //     this->y_ref.push_back(y);
    // }

    // Vectors for first and second derivatives
    std::vector<double> dx_ref(this->x_ref.size()), dy_ref(this->y_ref.size());
    std::vector<double> ddx_ref(this->x_ref.size()), ddy_ref(this->y_ref.size());

    // Difference between the the values either side and dividing by 2, except for the boudaries
    // Taylor expansion second order
    for (size_t i = 1; i < this->x_ref.size()-1; ++i) {
        dx_ref[i] = (this->x_ref[i+1] - this->x_ref[i-1]) / (2*this->dt);
        dy_ref[i] = (this->y_ref[i+1] - this->y_ref[i-1]) / (2*this->dt);
    }

    // Difference for the first and last points
    // Taylor expansion first order
    dx_ref[0] = (this->x_ref[1] - this->x_ref[0]) / this->dt;
    dy_ref[0] = (this->y_ref[1] - this->y_ref[0]) / this->dt;
    dx_ref.back() = (this->x_ref.back() - this->x_ref[x_ref.size()-2]) / this->dt;
    dy_ref.back() = (this->y_ref.back() - this->y_ref[y_ref.size()-2]) / this->dt;

    // Calculate the orientation
    this->th_ref.push_back(std::atan2(dy_ref[0], dx_ref[0]));
    for (size_t i = 1; i < dx_ref.size(); ++i) {
        this->th_ref.push_back(std::atan2(dy_ref[i], dx_ref[i]));
        if (this->th_ref[i] - this->th_ref[i-1] > M_PI) {
            while(this->th_ref[i] - this->th_ref[i-1] > M_PI)
                this->th_ref[i] -= 2*M_PI;
        } else if (this->th_ref[i] - this->th_ref[i-1] < -M_PI) {
            while(this->th_ref[i] - this->th_ref[i-1] < -M_PI)
                this->th_ref[i] += 2*M_PI;
        }
    }

    // for (size_t i = 0; i < dx_ref.size(); i++)
    // {
    //     this->th_ref[i] = atan2(sin(this->th_ref[i]), cos(this->th_ref[i]));
    // }
    
    
    // Second derivatives calculate the same way as the first ones
    for (size_t i = 1; i < dx_ref.size()-1; ++i) {
        ddx_ref[i] = (dx_ref[i+1] - dx_ref[i-1]) / (2*this->dt);
        ddy_ref[i] = (dy_ref[i+1] - dy_ref[i-1]) / (2*this->dt);
    }
    ddx_ref[0] = (dx_ref[1] - dx_ref[0]) / this->dt;
    ddy_ref[0] = (dy_ref[1] - dy_ref[0]) / this->dt;
    ddx_ref.back() = (dx_ref.back() - dx_ref[dx_ref.size()-2]) / this->dt;
    ddy_ref.back() = (dy_ref.back() - dy_ref[dy_ref.size()-2]) / this->dt;

    // Velocities
    for (size_t i = 0; i < dx_ref.size(); ++i) {
        this->v_ref.push_back(std::sqrt(dx_ref[i]*dx_ref[i] + dy_ref[i]*dy_ref[i]));
        // To avoid division by zero
        if (this->v_ref.back() > 0.001) {
            this->w_ref.push_back((ddy_ref[i]*dx_ref[i] - ddx_ref[i]*dy_ref[i]) / 
                            (dx_ref[i]*dx_ref[i] + dy_ref[i]*dy_ref[i]));
        } else {
            this->w_ref.push_back(0);
        }
    }
}

void Trajectory::saveTrajectory(std::vector<std::vector<double>> _all
                                 , std::vector<double> _x
                                 , std::vector<double> _y
                                 , std::vector<double> _th
                                 , std::vector<double> _v
                                 , std::vector<double> _w
                                 , std::string _name)
{

    std::ofstream outFile(_name);
    outFile << "x y theta v w\n"; // Header

    if(_all.size())
    {
        for (size_t i = 0; i < _all.size(); ++i)
        {
            outFile << _all[i][0] << " " << _all[i][1] << " " << _all[i][2] << " "
                    << _all[i][3] << " " << _all[i][4] << "\n";
        }
        outFile.close();
        return;
    }

    for (size_t i = 0; i < _x.size(); ++i)
    {
        outFile << _x[i] << " " << _y[i] << " " << _th[i] << " "
                << _v[i] << " " << _w[i] << "\n";
    }
    outFile.close();

}