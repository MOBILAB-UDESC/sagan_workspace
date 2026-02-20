#include "trajectory.hpp"


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

void Trajectory::setTrajectoryFromWaypoints(
    const std::vector<std::pair<double, double>>& waypoints,
    double _dt,
    double desired_velocity,
    std::string interpolation_type)
{
    this->dt = _dt;
    this->x_ref.clear();
    this->y_ref.clear();
    this->th_ref.clear();
    this->v_ref.clear();
    this->w_ref.clear();

    if (waypoints.size() < 2) {
        std::cerr << "Need at least 2 waypoints" << std::endl;
        return;
    }

    // Calculate cumulative path length
    std::vector<double> path_lengths;
    path_lengths.push_back(0.0);
    for (size_t i = 1; i < waypoints.size(); ++i) {
        double dx = waypoints[i].first - waypoints[i-1].first;
        double dy = waypoints[i].second - waypoints[i-1].second;
        double segment_length = sqrt(dx*dx + dy*dy);
        path_lengths.push_back(path_lengths.back() + segment_length);
    }

    double total_length = path_lengths.back();
    double total_time = total_length / desired_velocity;
    this->n_points = static_cast<int>(total_time / this->dt);

    // Create parameter s (0 to 1) for interpolation
    for (int i = 0; i < this->n_points; ++i) {
        double t = i * this->dt;
        double s = (desired_velocity * t) / total_length;
        
        if (s > 1.0) s = 1.0; // Cap at the end

        // Find which segment we're in
        size_t segment_idx = 0;
        for (size_t j = 1; j < path_lengths.size(); ++j) {
            double normalized_s = path_lengths[j] / total_length;
            if (s <= normalized_s) {
                segment_idx = j - 1;
                break;
            }
        }

        // Linear interpolation within the segment
        double s0 = (segment_idx > 0) ? path_lengths[segment_idx] / total_length : 0.0;
        double s1 = path_lengths[segment_idx + 1] / total_length;
        double segment_s = (s - s0) / (s1 - s0);

        double x0 = waypoints[segment_idx].first;
        double y0 = waypoints[segment_idx].second;
        double x1 = waypoints[segment_idx + 1].first;
        double y1 = waypoints[segment_idx + 1].second;

        if (interpolation_type == "cubic") {
            // Cubic Hermite interpolation for smoother path
            double t2 = segment_s * segment_s;
            double t3 = t2 * segment_s;
            double h00 = 2*t3 - 3*t2 + 1;
            double h10 = t3 - 2*t2 + segment_s;
            double h01 = -2*t3 + 3*t2;
            double h11 = t3 - t2;

            // Use finite difference for tangents
            double m0x, m0y, m1x, m1y;
            if (segment_idx == 0) {
                m0x = (waypoints[1].first - waypoints[0].first) * 0.5;
                m0y = (waypoints[1].second - waypoints[0].second) * 0.5;
            } else {
                m0x = (waypoints[segment_idx].first - waypoints[segment_idx-1].first) * 0.5;
                m0y = (waypoints[segment_idx].second - waypoints[segment_idx-1].second) * 0.5;
            }

            if (segment_idx == waypoints.size() - 2) {
                m1x = (waypoints[segment_idx+1].first - waypoints[segment_idx].first) * 0.5;
                m1y = (waypoints[segment_idx+1].second - waypoints[segment_idx].second) * 0.5;
            } else {
                m1x = (waypoints[segment_idx+2].first - waypoints[segment_idx].first) * 0.5;
                m1y = (waypoints[segment_idx+2].second - waypoints[segment_idx].second) * 0.5;
            }

            this->x_ref.push_back(h00*x0 + h10*m0x + h01*x1 + h11*m1x);
            this->y_ref.push_back(h00*y0 + h10*m0y + h01*y1 + h11*m1y);
        } else {
            // Linear interpolation (simple but not smooth)
            this->x_ref.push_back(x0 + segment_s * (x1 - x0));
            this->y_ref.push_back(y0 + segment_s * (y1 - y0));
        }
    }

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