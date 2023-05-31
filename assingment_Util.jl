
function remove_days_with_few_occurrences(df::DataFrame, t::Int)
    days_counts = combine(groupby(df, :Beginn), nrow => :count)
    filtered_counts = filter(row -> row.count > t, days_counts)
    cleaned_df = innerjoin(df, filtered_counts, on = :Beginn)
    select!(cleaned_df, Not(:count))
    return cleaned_df
end

function unique_Containers(df::DataFrame)
    return sort(select(df, [:Container, :Material, :Maschinen_Typ, :Baustelle_x_Koordinate, :Baustelle_y_Koordinate, :Beginn, :Werktage]) 
    |> unique!, [:Container, :Beginn])
end

function add_origin_positions(df::DataFrame, x::Float64, y::Float64)
    unique_containers = unique(df.Container)
    
    for container in unique_containers
        new_entry = (
            Container = container, 
            Material = C_NULL, 
            Maschinen_Typ = C_NULL, 
            Baustelle_x_Koordinate = x, 
            Baustelle_y_Koordinate = y, 
            Beginn = -1, 
            Werktage = 0
        )
    push!(df, new_entry)
    return df
    end
end

"""
add_container_coordinates(df::DataFrame)

The mehtod takes: 
    DataFrame df  
adds 2 new Columns to the df:
    Container_x_Koordinate, Container_y_Koordinate : Site coordinates of the Last 
    Row with this Container or the mean of all for the first Assingment of the Container.
"""
function add_container_coordinates(df::DataFrame)
    df = sort(df, [:Beginn])
    container_groups = groupby(df, :Container)
    for container in container_groups
        container[!, :Container_x_Koordinate] = insert!([container[i-1, :Baustelle_x_Koordinate] for i in 2:size(container, 1)],1, container[1, :Baustelle_x_Koordinate])
        container[!, :Container_y_Koordinate] = insert!([container[i-1, :Baustelle_y_Koordinate] for i in 2:size(container, 1)],1, container[1, :Baustelle_y_Koordinate])
    end
    return df
end
"""
Train test Split based on days 
"""
function split(df::DataFrame, ratio::Float64)
    days = unique(df, :Beginn).Beginn
    days_partition = MLJ.partition(days, ratio, multi=false);
    days_train, days_test = days_partition
    data_train = filter(row -> row.Beginn in days_train, df)
    data_test = filter(row -> row.Beginn in days_test, df)
    return data_train, data_test
end
"""
turn_data_into_assignment_matrix(df::DataFrame, df_containers::DataFrame)

The method takes:
    DataFrame df,
    DataFrame df_containers (unique Containers),
    Int n
Returns an assingment Matrix for each day with:
    the Orders,
    the Containers assinged on the day and 
    the n additional Containers that where the longest not assinged,
    1 where the container serves the Order,
    0 everywhere else.
"""
function turn_data_into_assignment_matrix(df::DataFrame, df_containers::DataFrame, n::Int64)
    day_groups = groupby(df, :Beginn)
    matrix_list = Vector{Array{Int, 2}}(undef, length(day_groups))

    for (day_index, day) in enumerate(day_groups)
        date = day[1, :Beginn]

        # Filter containers with 'Beginn' on the specified date
        on_date_containers = filter(row -> row.Beginn == date, df_containers)

        filtered_containers = filter(row -> (row.Beginn + row.Werktage <= date), df_containers)
        
        sorted_containers = sort(filtered_containers,  :Beginn)
    
        if size(sorted_containers, 1) > n
            sorted_containers = sorted_containers[1:n, :]
        end
        # Concatenate the two DataFrames, selecting the top 25 off-date containers
        selected_containers = vcat(on_date_containers, sorted_containers[1:min(n, nrow(sorted_containers)), :])
        selected_containers = sort(selected_containers,  :Beginn)
        num_assignments = nrow(day)
        num_containers = nrow(selected_containers)
        matrix = zeros(Int, num_assignments, num_containers)

        for i in 1:num_assignments
            for j in 1:num_containers
                if day[i, :Container] == selected_containers[j, :Container]
                    matrix[i, j] = 1
                end
            end 
        end
        matrix_list[day_index] = matrix
    end

    return matrix_list
end
"""
turn_data_into_input_values(df::DataFrame, df_containers::DataFrame)

The method takes:
    DataFrame df,
    DataFrame df_containers (unique Containers),
    Int n
Returns an :
    1. assingment Matrix for each day and 
    entrys for each Container Order Combination with 4 features:
        1. The Container is able to fullfill the Order,
        2. The Verbrauch_proTag of the Order
        3. The Werktage (duration) of the Order
        4. The Distances of Container_positions to Site_position
    With all Container assinged on the Day + the n additional free Containers
    2. DataFrame with the Orders.
    3. DataFrame with the Containers
    
"""
function turn_data_into_input_values(df::DataFrame, df_containers::DataFrame, n::Int64)
    day_groups = groupby(df, :Beginn)
    values_list = Vector{Array{Float32, 3}}(undef, length(day_groups))
    day_dataframes = Vector{DataFrame}(undef, length(day_groups))
    selected_containers_dataframes = Vector{DataFrame}(undef, length(day_groups))

    for (day_index, day) in enumerate(day_groups)
        date = day[1, :Beginn]

        on_date_containers = filter(row -> row.Beginn == date, df_containers)
        on_date_containers[!,:Baustelle_x_Koordinate], on_date_containers[!,:Baustelle_y_Koordinate] = on_date_containers[!,:Container_x_Koordinate], on_date_containers[!,:Container_y_Koordinate]
        filtered_containers = filter(row -> (row.Beginn + row.Werktage <= date), df_containers)


        sorted_containers = sort(filtered_containers,  :Beginn )

        if size(sorted_containers, 1) > n
            sorted_containers = sorted_containers[1:n, :]
        end
        selected_containers = vcat(on_date_containers, sorted_containers[1:min(n, nrow(sorted_containers)), :])
        selected_containers = sort(selected_containers,  :Beginn)
        num_assignments = nrow(day)
        num_containers = nrow(selected_containers)
        matrix = zeros(4, num_assignments, num_containers)
        for i in 1:num_assignments
            row = @view day[i, :]
            for j in 1:num_containers
                container = @view selected_containers[j, :]
                container_material = @view df_containers[df_containers.Container .== container.Container, :Material]
                container_m_typ = @view df_containers[df_containers.Container .== container.Container, :Maschinen_Typ]

                container_able = row.Material in container_material && row.Maschinen_Typ in container_m_typ
                matrix[1, i, j] = container_able ? 1 : 0
                matrix[2, i, j] = row.Verbrauch_proTag
                matrix[3, i, j] = row.Werktage
                matrix[4, i, j] = euclidean([row.Baustelle_x_Koordinate,  row.Baustelle_x_Koordinate],
                                            [container.Baustelle_x_Koordinate, container.Baustelle_x_Koordinate])
            end 
        end
        values_list[day_index] = matrix
        day_dataframes[day_index] = day
        selected_containers_dataframes[day_index] = selected_containers
    end

    return values_list, day_dataframes, selected_containers_dataframes
end

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
                container_x, container_y = container[container_idx, :Baustelle_x_Koordinate], container[container_idx, :Baustelle_y_Koordinate]
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

function display_solution_old(problem, route)
    x_pos = [c.x for c in problem]
    y_pos = [c.y for c in problem]
    plot_result = scatter(x_pos, y_pos, shape = :circle, markersize = 6)
    for i in 1:length(problem)
        for j in 1:length(problem)
            val = route[i, j]
            if val > 0
                del1 = problem[i]
                del2 = problem[j]
                plot!([del1.x, del2.x], [del1.y, del2.y], legend = false)
            end
        end
    end
    return plot_result
end