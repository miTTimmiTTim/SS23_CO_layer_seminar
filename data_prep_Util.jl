

"""
remove_days_with_few_occurrences(df::DataFrame, t::Int)

Inputs:
    DataFrame df
    Int t: threshhould
Filters out days where less then t assignments occured.
"""
function remove_days_with_few_occurrences(df::DataFrame, t::Int)
    days_counts = combine(groupby(df, :Beginn), nrow => :count)
    filtered_counts = filter(row -> row.count > t, days_counts)
    cleaned_df = innerjoin(df, filtered_counts, on = :Beginn)
    select!(cleaned_df, Not(:count))
    return cleaned_df
end
"""
min_max_Scaler(df_train::DataFrame, df_test::DataFrame)

Inputs:
    DataFrame train_df
    DataFrame test_df
Scales the DataFrames using the min max scaler based on train_df
"""
function min_max_Scaler(df_train::DataFrame, df_test::DataFrame)
    columns = [:Werktage, :Verbrauch_proTag, :Baustelle_x_Koordinate, :Baustelle_y_Koordinate, :Container_x_Koordinate, :Container_y_Koordinate]
    
    for colum in columns
        min, max = minimum(df_train[!, colum]), maximum(df_train[!,colum])
        df_train[!, colum] =  (df_train[!, colum] .- min)/(max-min)
        df_test[!, colum] =  (df_test[!, colum] .- min)/(max-min)
    end 
    return df_train, df_test
end
"""
add_container_coordinates(df::DataFrame)

Inputs:
    DataFrame df  
adds 2 new Columns to the df:
    Container_x_Koordinate, Container_y_Koordinate : Site coordinates of the Last 
    Row with this Container or the mean of all, for the first assignment of the Container.
"""
function add_container_coordinates(df::DataFrame)
    container_groups = groupby(df, :Container)
    for container in container_groups
        container[!, :Container_x_Koordinate] = insert!([container[i-1, :Baustelle_x_Koordinate] for i in 2:size(container, 1)],1, mean(df.Baustelle_x_Koordinate))
        container[!, :Container_y_Koordinate] = insert!([container[i-1, :Baustelle_y_Koordinate] for i in 2:size(container, 1)],1, mean(df.Baustelle_y_Koordinate))
    end
    return df
end
"""
Train test Split based on days.
Train contains ratio amount of days.
Test contains 1-ratio amount of days. 
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
turn_data_assignments(df::DataFrame, df_containers::DataFrame, n::Int64)

Inputs:
    DataFrame df
    DataFrame df_containers (unique Containers)
    int n
Returns an :
    1. assignment Matrix for each day and 
    entrys for each Container Order Combination with 4 features:
        1. The Container is able to fullfill the Order,
        2. The Verbrauch_proTag of the Order
        3. The Werktage (duration) of the Order
        4. The Distances of Container_positions to Site_position
    With all Container assinged on the Day + the n additional free Containers
    2. DataFrame with the Orders.
    3. DataFrame with the Containers
    4. assignment Matrix for each day
    Based on n samples from the df
"""
function turn_data_assignments(df::DataFrame, df_containers::DataFrame, n::Int64)
    day_groups = groupby(df, :Beginn)
    selected_days = sample(range(2,length(day_groups)-1), n; replace=false)
    assignment_list = Vector{Array{Int, 2}}(undef, length(selected_days))
    values_list = Vector{Array{Float32, 3}}(undef, length(selected_days))
    day_dataframes = Vector{DataFrame}(undef, length(selected_days))
    containers_dataframes = Vector{DataFrame}(undef, length(selected_days))

    for (num, day_index) in enumerate(selected_days)
        day = day_groups[day_index]
        close_days = day_groups[max(1, day_index-1):min(length(day_groups), day_index+1)]
        containers = reduce(vcat, close_days)

        num_assignments = nrow(day)
        num_containers = nrow(containers)
        assignment = zeros(Int, num_assignments, num_containers)
        values = zeros(4, num_assignments, num_containers)
        for i in 1:num_assignments
            row = @view day[i, :]
            for j in 1:num_containers
                container = @view containers[j, :]
                if row.Container == container.Container
                    assignment[i, j] = 1
                end
                container_material = @view df_containers[df_containers.Container .== container.Container, :Material]
                container_m_typ = @view df_containers[df_containers.Container .== container.Container, :Maschinen_Typ]

                container_able = row.Material in container_material && row.Maschinen_Typ in container_m_typ
                values[1, i, j] = container_able ? 1 : 0
                values[2, i, j] = row.Verbrauch_proTag
                values[3, i, j] = row.Werktage
                values[4, i, j] = euclidean([row.Baustelle_x_Koordinate,  row.Baustelle_x_Koordinate],
                                            [container.Container_x_Koordinate, container.Container_y_Koordinate])
            end 
        end
        values_list[num] = values
        day_dataframes[num] = day[!, [:Baustelle_x_Koordinate, :Baustelle_y_Koordinate]]
        containers_dataframes[num] = containers[!, [:Container_x_Koordinate, :Container_y_Koordinate]]
        assignment_list[num] = assignment
    end

    return values_list, day_dataframes, containers_dataframes, assignment_list
end