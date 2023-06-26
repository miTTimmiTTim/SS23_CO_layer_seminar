
"""
Util script for the TravelingSalesPerson problem with pickup and delivery.
Implementation based on https://github.com/PiotrZakrzewski/julia-tsp
"""
# Define Delivery struct
struct Delivery
    x::Float64
    y::Float64
end

function create_deliveries(tasks::Vector{DataFrame}, containers::Vector{DataFrame}, assignments::Vector{Matrix{Int64}})
    all_deliveries = []
    all_constraints = []
    # Loop through each task and container dataframe and corresponding assignment matrix
    for (task, container, assignment) in zip(tasks, containers, assignments)
        deliveries = []
        constraints = []

        # Loop through each row of the assignment matrix
        for i in 1:size(assignment, 1)
            # Get the index of the matching container (value = 1)
            container_idx = findfirst(x -> x == 1, assignment[i, :])
            
            # Check if there is a match
            if container_idx != nothing
                # Get the x, y coordinates from task and container dataframes
                task_x, task_y = task[i, :Baustelle_x_Koordinate], task[i, :Baustelle_y_Koordinate]
                container_x, container_y = container[container_idx, :Container_x_Koordinate], container[container_idx, :Container_y_Koordinate]
                # Create delivery objects and add them to the list
                pickup = Delivery(container_x, container_y)
                delivery = Delivery(task_x, task_y)
                # Add a contains part that checks if already in the lists 
                
                push!(deliveries, pickup)
                push!(deliveries, delivery)

                # Add the indices of the pickups and deliveries to the constraints list
                push!(constraints, (2i-1, 2i))
            end
        end

        # Add the deliveries and constraints for this task to the overall list
        push!(all_deliveries, deliveries)
        push!(all_constraints, constraints)
    end

    # Return the list of deliveries and constraints for all tasks
    return all_deliveries, all_constraints
end


# Calculate distance between two Delivery locations
function dist(del1::Delivery, del2::Delivery)
    return sqrt((del1.x - del2.x)^2 + (del1.y - del2.y)^2)
end

# Calculate the travel matrix between all delivery locations
function calc_travelmatrix(deliveries)
    tm = zeros(Float64, length(deliveries), length(deliveries))
    for i = 1:length(deliveries)
        for j = 1:length(deliveries)
            tm[i, j] = dist(deliveries[i], deliveries[j])
        end
    end
    return tm
end

# Function to find the shortest subtour in a matrix
function shortest_subtour(matrix::Matrix{Int64})
    g = Graphs.SimpleDiGraph(matrix)
    cycles = []
    max_cycle_len = Graphs.nv(g)
    for node in Graphs.vertices(g)
        push!(cycles, Graphs.neighborhood(g, node, max_cycle_len))
    end
    cycles = filter(x -> max_cycle_len > length(x) > 1, cycles)
    if isempty(cycles)
        return []
    end
    return sort(cycles, by=length)[1]
end

# Main function to solve TSP with pickup and delivery constraints
function solve_tsp(deliveries, constraints, solver)
    # Calculate the travel matrix
    travelmatrix = calc_travelmatrix(deliveries)
    
    # Initialize the optimization model
    model = JuMP.Model(solver)
    set_silent(model)
    # Number of locations
    n = length(deliveries)

    # Define binary decision variables for routes
    @variable(model, route[1:n, 1:n], Bin)

    # Define integer decision variables for "time" or sequence
    @variable(model, 1 <= time[1:n] <= n, Int)

    # Ensure each location is visited exactly once as a destination
    @constraint(model, [i = 1:n], sum(route[i, :]) == 1.0)

    # Ensure each location is visited exactly once as a source
    @constraint(model, [c = 1:n], sum(route[:, c]) == 1.0)

    # Disallow traveling to itself
    @constraint(model, [j = 1:n], route[j, j] == 0)

    # Enforce pickup before delivery and allow other locations in between
    for (pickup, delivery) in constraints
        @constraint(model, time[pickup] + 1 <= time[delivery])
    end

    # Calculate the total travel time
    traveltime = travelmatrix .* route

    # Objective function: minimize the total travel time
    @objective(model, Min, sum(traveltime))
    # Callback function for lazy constraints
    function callback(cb_data)
    status = callback_node_status(cb_data, model)
    if status == MOI.CALLBACK_NODE_STATUS_FRACTIONAL
        return
    end
    x_val = callback_value.(cb_data, route)
    x_val = round.(x_val)
    x_val = Int64.(x_val)
    cycle = shortest_subtour(x_val)
    sub_inds = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i != j]
    if length(sub_inds) > 0
        con = @build_constraint(sum(route[i, j] for (i,j) in sub_inds) <= length(cycle) - 1 )
        MOI.submit(model, MOI.LazyConstraint(cb_data), con)
    end
end
    # Set callback and optimize the model
    MOI.set(model, MOI.LazyConstraintCallback(), callback)
    optimize!(model)

    total_cost = JuMP.objective_value(model)
    
    # Get the optimized route values
    route_val = JuMP.value.(route)

    # Return the optimized route and total cost
    return (route_val, total_cost)
end

function display_solution(problem, route)
    plot_result = scatter(shape = :circle, markersize = 6, legend = false)

    for i in 1:length(problem)
        for j in 1:length(problem)
            val = route[i, j]
            if val > 0
                del1 = problem[i]
                del2 = problem[j]

                if i % 2 == 0
                    scatter!([del1.x, del2.x], [del1.y, del2.y], color = :blue)
                else
                    scatter!([del1.x, del2.x], [del1.y, del2.y], color = :red)
                end

                if (i+1) == j 
                    plot!([del1.x, del2.x], [del1.y, del2.y], color = :green)
                else
                    plot!([del1.x, del2.x], [del1.y, del2.y], color = :black)
                end
            end
        end
    end
    return plot_result
end
