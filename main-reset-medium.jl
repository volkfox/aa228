
# run with > julia -p 7

@everywhere  begin

using DataFrames
using DataArrays
# medium policy code transition matrix
#using JLD

#println("medium program worker ",myid(), "starting")
infile="medium.csv"
f = open(infile)
df = readtable(infile)
nodes=names(df)
close(f)

transitions = Dict{Tuple{Int64,Int64,Int64},Tuple{Float64, Float64}}()

uniq=unique(df)
state_arr=by(uniq,[:s],nrow)[1]
sorted_state=sort(state_arr)
num_states=size(state_arr)[1]

# assumes rewards are always the same for every (s,a,sp) direction vector
# NOTE: nrow() function in DataFrames is totally broken for large datasets

k=1
for s in state_arr #FIXME
# for s in state_arr[1:10]

    subtable_s = uniq[ uniq[:s] .== s, :]
    for r_frame in eachrow(subtable_s)
       a = r_frame[:a]
       sp = r_frame[:sp]
       reward = r_frame[:r]
       ways=size( df[ (df[:s] .== s) & (df[:a].== a) & (df[:sp].== sp), :])[1]
       total_ways=size( df[ (df[:s] .== s) & (df[:a].== a), :])[1]
       
       transitions[(s,a,sp)]=((ways/total_ways),reward)
       
    end
    k=k+1
    if (k%1000) == 0
      println("worker:", myid(), " processing state ",k, " of ",num_states)
    end
end
println("Transition matrix read by worker: ",myid())


# precompute all lookup tables

# uncomment the below in production!
subtable_s = Dict{Int64,DataFrames.DataFrame}() 
subtable_s_a = Dict{Tuple{Int64,Int64},DataFrames.DataFrame}() 
#
for s in state_arr
  subtable_s[s]= uniq[ uniq[:s] .== s, :]
    for a in by(subtable_s[s],[:s,:a],nrow)[:a]
      subtable_s_a[(s,a)] = subtable_s[s][ subtable_s[s][:a] .== a, :]      
    end
end

println("Transition frames precomputed on worker: ",myid())

# main program for policy computation

function myrange(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0,size(q,1),nchunks+1)]
    splits[idx]+1:splits[idx+1]
end

function compute(iterations, gamma, total_states, U, Fin)

    range=myrange(U)
    println("I am worker:",myid()," and I got index: ", range)

#for i in 1:iterations
 i=1 
 while true
   for s in state_arr[range]
        results=Array{Array{Float64,2},1}()
        
        for a in by(subtable_s[s],[:s,:a],nrow)[:a]
            
            expected_reward=0.0
            
            for sp in subtable_s_a[(s,a)][:sp]
         
                probability=transitions[(s,a,sp)][1]
                reward=transitions[(s,a,sp)][2]

                index_sp=findfirst(state_arr, sp)
                
                if (index_sp == 0)
                    index_sp(x -> x>j, sorted_state)
                    if index_sp == 0
                      index_sp = findfirst(x -> x<j, sorted_state)
                    end
                    if index_sp == 0
                      println(myid(),": index ", index_sp, " cannot be inferred")
                      index_sp = 139
                    end
                end
                expected_reward += probability*(reward+U[index_sp,1])
                
            end
            push!(results, [expected_reward a])
        end
        best_utility=-Inf
        best_action=rand(1:7)
        
        for policy in results
        
            if policy[1] >= best_utility
                best_utility=policy[1]
                best_action=policy[2]
            end   
        end
        index_s=findfirst(state_arr, s) # this index should exist
        U[index_s, 1]=best_utility
        U[index_s, 2]=best_action
    end
    println("worker: ",myid()," iteration: ", i, " out of ", iterations)
    
    if i==iterations
       Fin[myid()]=1
       println("worker ",myid()," sent fin")
    end
   
    if  Fin[1]==1
       break
    end
    i=i+1

  end #while looop
end   #function


end 

# continue main program

U = SharedArray(Float64, (length(state_arr),2))
Fin = SharedArray(Int64, nprocs())
fill!(U,0)                         
iterations=10000
gamma=1.0
total_states=50000

function waiter(Fin)
  println("started fin waiter")
  while true 
    if countnz(Fin)==nworkers()
       Fin[1]=1
       break
    end 
    sleep(5)
  end
end

@sync begin
    for p in procs(U)
        @async begin
          remotecall_wait(compute, p, iterations, gamma, total_states, U, Fin)
        end
    end
    @async waiter(Fin)
end

f=open("medium.policy","w")

optimal_policy=U[:,2]
policy_dict=Dict(zip(state_arr, optimal_policy))
state_set=Set{Int64}(state_arr)

for j in 1:total_states
                
                if (!in(j,state_set))  # state with no information

                 copyindex=findfirst(x -> x>j, sorted_state)
                 if copyindex==0
                     copyindex=findfirst(x -> x<j, sorted_state)
                 end
                 if copyindex==0
                     copyindex=139
                 end
                 @printf(f, "%s\n", convert(Int64,policy_dict[state_arr[copyindex]]))
                else
                 @printf(f, "%s\n", convert(Int64,policy_dict[j]))
                end
    end   
    println("medium policy written")
    close(f)



