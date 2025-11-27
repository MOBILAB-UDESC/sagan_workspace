#include "mpc.hpp"

MPC::MPC(int N, double dt,
        std::vector<double> u_min,
        std::vector<double> u_max,
        std::vector<double> z_min,
        std::vector<double> z_max,
        std::vector<std::vector<double>> Q,
        std::vector<std::vector<double>> R)
{
    this->N = N;
    this->dt = dt;
    this->u_min = u_min;
    this->u_max = u_max;
    this->z_min = z_min;
    this->z_max = z_max;
    this->Q = Q;
    this->R = R;

    this->nz = z_min.size();
    this->nu = u_min.size();
}

bool MPC::setup_constant_mpc()
{
    // create casadi variables
    MX z = MX::sym("z", this->nz); // states
    MX u = MX::sym("u", this->nu); // control inputs

    // define the system dynamics
    MX rhs = MX::vertcat({
        u(0)*MX::cos(z(2)),
        u(0)*MX::sin(z(2)),
        u(1)
    });
    
    // nonlinear mapping function f(z,u)
    this->f = Function("f", {z, u}, {rhs});

    // Uncomment the following lines to test the function f with specific inputs
    // DMVector inputs = {      // == std::vector<DM>
    //     DM({0.0, 0.0, 0.0}), // Any state
    //     DM({1.0, 1.0})       // Any control input
    // };
    // std::cout << "f(z,u) = " << this->f(inputs)[0] << std::endl;

    MX U = MX::sym("U", this->nu, this->N);
    MX P = MX::sym("P", this->nz + this->N*(this->nz+this->nu)); // P = [z_init.T; ref_0; ref_1; ...; ref_{N-1}], where ref_k = [z_ref_k.T; u_ref_k.T]
    MX Z = MX::sym("Z", this->nz, this->N+1);

    MX obj = 0.0; // Cost function
    std::vector<MX> g; // Constraints vector

    MX z0 = Z(Slice(0, this->nz), 0); // initialize with the initial state of the system
    g.push_back(z0-P(Slice(0, this->nz))); // Initial state constraint. Note: P(Slice(0, this->nz)) == P[0:nz] no python

    // std::cout << z0.size() << std::endl;
    // std::cout << g << std::endl;

    for (int k = 0; k < this->N; ++k)
    {
        z0  = Z(Slice(0, this->nz), k);
        MX u0  = U(Slice(0, this->nu), k);

        // std::cout << z0.size() << std::endl;
        // std::cout << u0.size() << std::endl;

        MX diff_z = z0 - P(Slice((this->nz+this->nu)*k+this->nz, (this->nz+this->nu)*k+2*this->nz));
        MX diff_u = u0 - P(Slice((this->nz+this->nu)*k+2*this->nz, (this->nz+this->nu)*k+2*this->nz+this->nu));

        // std::cout << diff_z.size() << std::endl;
        // std::cout << diff_u.size() << std::endl;

        obj += MX::mtimes({diff_z.T(), DM(this->Q), diff_z});
        obj += MX::mtimes({diff_u.T(), DM(this->R), diff_u});

        auto z_next  = Z(Slice(0, 3), k+1);
        auto f_value = this->f(std::vector<MX>{z0, u0})[0];
        auto z_euler = z0 + (this->dt*f_value);
        g.push_back(z_next - z_euler); // System dynamics constraint
    }
    
    // Obstacle avoidance
    for (int obs = 0; obs < this->n_obstacles; ++obs){
        double obs_x    = this->obstacles[obs][0];
        double obs_y    = this->obstacles[obs][1];
        double obs_r    = this->obstacles[obs][2];
        double min_dist = this->obstacle_tol + this->robot_radius + obs_r;

        for (int k = 0; k < (N+1); ++k){
            MX dx = Z(0, k) - obs_x;
            MX dy = Z(1, k) - obs_y;
            MX dist = MX::sqrt(MX::pow(dx, 2) + MX::pow(dy, 2));
            g.push_back(-dist + min_dist);
        }
    }

    // std::cout << obj << std::endl;

    // Decision variables in the optimization problem
    MX OPT_variables = MX::vertcat({reshape(Z, this->nz*(this->N+1), 1), reshape(U, this->nu*(this->N), 1)});

    // std::cout << OPT_variables.size() << std::endl;

    // Nonlinear programming problem
    // MXDict nlp = {{"x", OPT_variables}, {"f", obj}, {"g", g}, {"p", P}};
    MXDict nlp = {{"x", OPT_variables}, {"f", obj}, {"g", MX::vertcat(g)}, {"p", P}};
    // MXDict opt = {{"ipopt", {{"print_level", 0}, {"max_iter", 2000}, {"acceptable_tol", 1e-8}, {"acceptable_obj_change_tol", 1e-6}}}, {"print_time", 0}};
    Dict opt = {{"print_time", 0}, {"ipopt", Dict{{"print_level", 0}, {"max_iter", 2000}, {"acceptable_tol", 1e-8}, {"acceptable_obj_change_tol", 1e-6}}}};

    this->solver = nlpsol("solver", "ipopt", nlp, opt);

    // initialize numeric arguments;
    int ng_dynamic = this->nz*(this->N+1);
    int ng_obstacl = this->n_obstacles*(this->N+1);

    this->args["lbg"] = DM::zeros(ng_dynamic + ng_obstacl, 1);
    for (int i = ng_dynamic; i < (ng_dynamic + ng_obstacl); ++i) {
        this->args["lbg"](i) = -DM::inf();
    }
    this->args["ubg"] = DM::zeros(ng_dynamic + ng_obstacl, 1);

    this->args["lbx"] = DM::ones(this->nz * (this->N+1) + this->nu * this->N, 1);
    this->args["ubx"] = DM::ones(this->nz * (this->N+1) + this->nu * this->N, 1);
    this->args["p"]   = DM::ones(this->nz * (this->N+1) + this->nu * this->N, 1);

    for (int k = 0; k <= N; ++k) {

        // Order = OPT_variables

        int idxz = k * this->nz;
        
        for (int i = 0; i < this->nz; ++i)
        {
            this->args["lbx"](idxz+i) = this->z_min[i];
            this->args["ubx"](idxz+i) = this->z_max[i];
        }

        if(k > 0)
        {
            int idxu = (k-1) * this->nu;
            for (int i = 0; i < this->nu; ++i)
            {
                this->args["lbx"](this->nz*(this->N+1)+idxu+i) = this->u_min[i];
                this->args["ubx"](this->nz*(this->N+1)+idxu+i) = this->u_max[i];
            }
        }
    }

    // std::cout << args["lbx"] << std::endl;
    // std::cout << args["ubx"] << std::endl;
    
    // MX Z0 = MX::repmat(z_init, N+1, 1);
    // MX U0 = MX::zeros(this->N, this->nu);
    // this->args["x0"] = MX::vertcat({reshape(Z0, this->nz*(this->N+1), 1), reshape(U0.T(), this->nu*(this->N), 1)});
    // std::cout << typeid(solver).name()  << std::endl;

    return true;
}

void MPC::set_reference(int mpc_iter)
{

    if(this->Z_ref.empty())
    {
        for (int k = 0; k < this->N; ++k)
        {
            double t_p = mpc_iter*this->dt + k*this->dt;
            double xr = 0.5*t_p;
            double yr = 1.0;
            double tr = 0.0;
            double vr = 0.5;
            double wr = 0.0;

            if (xr >= 12)
            {
                xr = 12.0;
                yr = 1.0;
                tr = 0.0;
                vr = 0.5;
                wr = 0.0;
            }

            std::vector<double> ref = {xr, yr, tr, vr, wr};

            for (int i = 0; i < (this->nz+this->nu); ++i)
            {
                this->args["p"](k*(this->nz+this->nu)+this->nz+i) = ref[i];
            }

        }

        return;
    }

    for (int k = 0; k < this->N; ++k)
    {
        // std::cout << "entró glob" << Z_ref.size() <<std::endl;
        if ((mpc_iter+k)<this->Z_ref.size())
        {
            // std::cout << "entró 1" << std::endl;
            for (int i = 0; i < (this->nz+this->nu); ++i)
            {
                this->args["p"](k*(this->nz+this->nu)+this->nz+i) = this->Z_ref[mpc_iter+k][i];
            }
        }
        else
        {
            // std::cout << "entró 2" << std::endl;
            std::vector<double> ref = this->Z_ref.back();
            ref[3] = 0.0;
            ref[4] = 0.0;
            for (int i = 0; i < (this->nz+this->nu); ++i)
            {
                this->args["p"](k*(this->nz+this->nu)+this->nz+i) = ref[i];
            }
        }

    }
    
}

void MPC::setup_variable_mpc(int mpc_iter, std::vector<double> z_init, std::vector<std::vector<double>> Z_init, std::vector<std::vector<double>> U_init)
{

    for (int k = 0; k < this->nz; ++k)
    {
        this->args["p"](k) = z_init[k];
    }

    this->set_reference(mpc_iter);

    // std::cout << args["p"] << std::endl;

    DM Z0;
    if (mpc_iter == 0)
    {
        Z0 = DM::repmat(z_init, N+1, 1);
    }else{
        Z0 = reshape(DM(Z_init).T(), this->nz*(N+1), 1);
    }

    DM U0 = DM::zeros(this->N, this->nu);
    for (int k = 0; k < this->N; ++k)
        for (int i = 0; i < this->nu; ++i)
            U0(k, i) = U_init[k][i];

    std::vector<DM> x0_vec;
    x0_vec.push_back(reshape(Z0, this->nz*(this->N+1), 1));
    x0_vec.push_back(reshape(U0.T(), this->nu*(this->N), 1));
    this->args["x0"] = DM::vertcat(x0_vec);

    // std::cout << "Iteration: " << mpc_iter << ", args[x0]: " << this->args["x0"] << std::endl;

}

void MPC::setup_obstacles(std::vector<double> obstacle)
{

    if(obstacle.size() != 3)
    {
        std::cout << "Wrong dimension. It should be a 3-element vector {cx,cy,radius}";
        return;
    }

    this->obstacles.push_back(obstacle);
    this->n_obstacles++;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MPC::solve_mpc()
{
    auto sol = this->solver(this->args);
    auto x_opt = static_cast<DM>(sol.at("x"));
    // std::cout << "XOPT: " << x_opt << std::endl;
    int idx_u = this->nz * (this->N + 1);

    std::vector<std::vector<double>> U_opt;
    std::vector<std::vector<double>> Z_opt;
    for (int k = 0; k < this->N+1; ++k) {
        // std::vector<double> uk(this->nu);
        std::vector<double> zk(this->nz);
        for (int i = 0; i < this->nz; ++i) {
            zk[i] = static_cast<double>(x_opt(k*this->nz + i));
        }
        Z_opt.push_back(zk);
    }
    for (int k = 0; k < this->N; ++k) {
        std::vector<double> uk(this->nu);
        for (int i = 0; i < this->nu; ++i) {
            uk[i] = static_cast<double>(x_opt(idx_u + k*this->nu + i));
        }
        U_opt.push_back(uk);
    }

    return {Z_opt, U_opt};
}

// void MPC::shift(double T, std::vector<double>& x0, std::vector<std::vector<double>>& U, Function& f) {
//     DM st = DM(x0);
//     DM con = DM(U[0]);
//     DM f_value = f(DMVector{st, con})[0];
//     st = st + T * f_value;
//     for (int i = 0; i < x0.size(); ++i) x0[i] = static_cast<double>(st(i));
//     t0 += T;

//     U.erase(U.begin());
//     U.push_back(U.back());
// }

void MPC::shift(double T, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& U) {
    
    U.erase(U.begin());
    U.push_back(U.back());

    for (int i = 0; i < U.size(); ++i)
    {
        DM st = DM(X[i]);
        DM con = DM(U[i]);
        DM f_value = this->f(DMVector{st, con})[0];
        st = st + T * f_value;
        X[i+1] = {static_cast<double>(st(0)), static_cast<double>(st(1)), static_cast<double>(st(2))};
    }
}


// int main()
// {
//     double t0 = 0.0;
//     float dt  = 0.1;
//     float sim_time = 15.0;
//     int mpc_iter = 0;
//     int N = 25;

//     Trajectory path;
//     path.setTrajectory(1.25, 5, dt, 1, {}, "infinite");

//     std::vector<double> z0 = {0.0, 0.0, 0.55};
//     std::vector<double> zg = {1.5, 1.5, 0.0};

//     std::vector<double> zmin = {-20.0, -20.0, -10e10};
//     std::vector<double> zmax = { 20.0,  20.0,  10e10};
//     std::vector<double> umin = { -1.2,  -3.1};
//     std::vector<double> umax = {  1.2,   3.1};

//     auto mpc = MPC(N, 2, dt,
//         umin,
//         umax,
//         zmin,
//         zmax,
//         {{1.0, 0.0, 0.0},
//         {0.0, 1.0, 0.0},
//         {0.0, 0.0, 0.1}},
//         {{0.5, 0.0},
//         {0.0, 0.05}}
//     );

//     mpc.Z_ref = path.getTrajectory();
//     // std::cout << "Ref: " << mpc.Z_ref << std::endl;

//     /*
//         Add obstacles here, before calling setup_constant_mpc
//     */
//     mpc.setup_obstacles({0.74, 0.5, 0.05});
//     mpc.setup_obstacles({0.0, 1.25, 0.05});


//     if (mpc.setup_constant_mpc())
//     {
//         std::cout << "MPC setup successful!" << std::endl;
//     }
//     else
//     {
//         std::cout << "MPC setup failed!" << std::endl;
//         return 0;
//     }

//     std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> res;
//     std::vector<std::vector<double>> U0(N, std::vector<double>(umin.size(), 0.0));
//     std::vector<std::vector<double>> Z0(N+1, std::vector<double>(zmin.size(), 0.0));

//     Z0[0][0] = z0[0];
//     Z0[0][1] = z0[1];
//     Z0[0][2] = z0[2];

//     float max_iter = mpc.Z_ref.size(); // sim_time/dt;
//     std::vector<std::vector<double>> mpc_output(max_iter, std::vector<double>((zmin.size()+umin.size())));

//     while(mpc_iter < max_iter)
//     {
//         mpc.setup_variable_mpc(mpc_iter, z0, Z0, U0);
        
//         res = mpc.solve_mpc();
//         Z0  = res.first;
//         U0  = res.second; 

//         Z0.erase(Z0.begin());
//         Z0.push_back(Z0.back());

//         shift(dt, t0, z0, U0, mpc.f);

//         // std::cout << "Iteration: " << mpc_iter << ", States: " << z0 << std::endl;
//         std::cout << "Iteration: " << mpc_iter << ", STATES: " << Z0 << std::endl;

//         mpc_output[mpc_iter][0] = z0[0];
//         mpc_output[mpc_iter][1] = z0[1];
//         mpc_output[mpc_iter][2] = z0[2];
//         mpc_output[mpc_iter][3] = U0[0][0];
//         mpc_output[mpc_iter][4] = U0[0][1];

//         mpc_iter++;

//     }

//     // std::cout << "Output: " << mpc_output << std::endl;

//     path.saveTrajectory(mpc_output, {}, {}, {}, {}, {}, "results.txt");
//     path.saveTrajectory(mpc.Z_ref, {}, {}, {}, {}, {}, "reference.txt");

//     for (int i = 0; i < mpc.n_obstacles; ++i)
//     {
//         path.setTrajectory(mpc.obstacles[i][2] + 0.1, 5, dt, 1, {mpc.obstacles[i][0], mpc.obstacles[i][1]}, "circle");
//         path.saveTrajectory(path.getTrajectory(), {}, {}, {}, {}, {}, "obstacle"+str(i)+".txt");
//     }

//     return 0;
// }