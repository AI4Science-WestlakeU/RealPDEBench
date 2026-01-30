using WaterLily
using StaticArrays

# Memory optimization settings
GC.gc()  # Clean up garbage at startup

function tapered_naca0025(p=7; Re=1e3, mem=Array, U=1, α=0.0)
    # Define simulation size and parameters
    n = 2^p
    chord_root = n*0.676     # Root chord length
    chord_tip = n*0.135      # Tip chord length (half of root)
    center = SA[n/4+5, n/2, n/2]  # Wing center position
    span = 2n - 6      # Span length
    ν = U*chord_root/Re  # Viscosity coefficient

    # NACA0025 parameters
    t = 0.25
    a0 = 0.2969
    a1 = 0.1260
    a2 = 0.3516
    a3 = 0.2843
    a4 = 0.1015

    # Convert angle to radians
    α_rad = α * π/180

    function get_thickness(x_c, chord)
        if x_c < 0 || x_c > 1
            return 0.0
        end
        x_sqrt = sqrt(max(x_c, 0.0))
        return 5*t*chord*(a0*x_sqrt - a1*x_c - a2*x_c^2 + a3*x_c^3 - a4*x_c^4)
    end

    body = AutoBody() do xyz,t
        # Transform to wing coordinate system
        x,y,z = xyz - center
        
        # Coordinate transformation considering angle of attack
        x_rot = x*cos(α_rad) + y*sin(α_rad)
        y_rot = -x*sin(α_rad) + y*cos(α_rad)
        
        # First check if within span range
        if abs(z) > span
            return abs(z) - span
        end
        
        # Calculate scaling factor at current z position (linearly decreasing from bottom to top)
        z_norm = (z + span)/(2*span)  # Normalize z to [0,1], z=-span is 0 (bottom), z=span is 1 (top)
        z_norm = min(max(z_norm, 0.0), 1.0)
        
        # Calculate local chord length at current position
        local_chord = chord_root + (chord_tip - chord_root) * z_norm
        
        # Normalize to local chord length
        x_c = x_rot/local_chord + 0.5  # Move to [0,1] interval
        
        # Calculate distance to airfoil surface
        if 0 ≤ x_c ≤ 1
            thickness = get_thickness(x_c, local_chord)
            # Calculate distance from point to airfoil surface
            return abs(y_rot) - thickness
        else
            # Outside airfoil, calculate distance to nearest edge
            dx = if x_c < 0
                x_rot + local_chord/2
            else
                x_rot - local_chord/2
            end
            return sqrt(dx^2 + y_rot^2)
        end
    end

    # Initialize simulation and return center for flow viz
    Simulation((2n,n,2n),(U,0,0),chord_root;ν,body,mem),center  # Horizontal incoming flow
end

using CUDA
using NPZ

reynolds_list = [10781, 11563, 12344, 13125, 13906, 14688, 15469, 16250, 17031, 17813]
aoa_list = [0.0, 5.0, 10.0, 15.0, 20.0]

for reynolds in reynolds_list
    for aoa in aoa_list
        println("reynolds = $reynolds, aoa = $aoa")
        
        # Clean up memory before each new parameter combination
        GC.gc()
        CUDA.reclaim()

        # Initialize simulation (add mem=CUDA.CuArray for GPU version)
        # Use CUDA to accelerate computation and specify GPU device
        CUDA.device!(0)  # Specify using GPU device 0
        sim, center = tapered_naca0025(7, Re=reynolds, mem=CUDA.CuArray, α=aoa)  # Set higher resolution and Reynolds number
        println("Initialization complete, grid size = ", size(sim.flow.p))

        # Main time loop: dimensionless time tU/L from t0 to t0+duration
        duration, step = 50, 0.01       
        # Record startup time
        t_start = time()

        t0 = 0

        # Create HDF5 file for streaming data saving (avoid memory overflow)
        using HDF5
        filename = "reynolds_$(reynolds)_aoa_$(aoa).h5"
        
        # Ensure directory exists
        mkpath(dirname(filename))
        
        # Memory monitoring function
        function check_memory_and_cleanup()
            free_mem_gb = Sys.free_memory() / 1024^3
            if free_mem_gb < 50  # If available memory is less than 50GB, force cleanup
                GC.gc()
                CUDA.reclaim()
                return true
            end
            return false
        end
        
        # Open HDF5 file for streaming write
        h5open(filename, "w") do file
            step_counter = 1
            for t in range(t0, t0 + duration; step)
                sim_step!(sim, t; remeasure = false)  # Static geometry, can disable remeasure
                @info "tU/L=$(round(t, digits=4))  Δt=$(round(sim.flow.Δt[end], digits=3))"
                
                # Create different group for each time step
                group_name = "timestep_$(step_counter)"
                g = create_group(file, group_name)
                
                # Only save cross-section data, use local variables to avoid memory accumulation
                pressure_slice = Array(sim.flow.p[:,:,95])
                velocity_slice = Array(sim.flow.u[:,:,95,:])
                vorticity_slice = Array(sim.flow.σ[:,:,95])
                
                g["pressure"] = pressure_slice
                g["velocity"] = velocity_slice
                g["vorticity"] = vorticity_slice
                g["time"] = t
                
                # Immediately clear local variables
                pressure_slice = nothing
                velocity_slice = nothing
                vorticity_slice = nothing
                
                # Force garbage collection and GPU memory reclaim
                GC.gc()
                CUDA.reclaim()  # Reclaim GPU memory
                
                # Print memory usage
                free_mem_gb = round(Sys.free_memory()/1024^3, digits=2)
                
                # Check and clean up memory every step
                if step_counter % 200 == 0  # Check memory every 200 steps
                    check_memory_and_cleanup()
                end
                
                step_counter += 1
            end
        end

        # Calculate and output elapsed time
        elapsed = time() - t_start
        println("Simulation elapsed time ", round(elapsed, digits = 2), " seconds")

        println("Calculation complete, total steps = ", length(sim.flow.Δt) - 1)
    end
end
