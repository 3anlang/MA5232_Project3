using Plots
using LinearAlgebra
using SparseArrays

function moments(V,g,h,N)
    # We apply composite trapozoidal rule
    Δv = V[2] - V[1]
    μ₀ = Δv*sum(g) - (g[1]+g[end])*Δv/2
    μ₁ = Δv*sum(V.*g) - (V[1]*g[1]+V[end]*g[end])*Δv/2
    μ₂ = Δv*sum(V.*V.*g) - (V[1]^2*g[1]+V[end]^2*g[end])*Δv/2
    ν = Δv*sum(h) - (h[1]+h[end])*Δv/2
    return [μ₀, μ₁, μ₂, ν]
end

function LJ(αβγ,V,μ)
    Δv = V[2] - V[1]
    g = exp.(αβγ[1] .+ αβγ[2].*V + αβγ[3].*V.^2)
    μ₀ = Δv*sum(g) - (g[1]+g[end])*Δv/2
    μ₁ = Δv*sum(g.*V) - (V[1]*g[1]+V[end]*g[end])*Δv/2
    μ₂ = Δv*sum(g.*V.^2) - (V[1]^2*g[1]+V[end]^2*g[end])*Δv/2
    L = [μ₀,μ₁] .- [μ[1],μ[2]]
    J = [μ₀ μ₁;
        μ₁ μ₂]
    return L,J
end


function one_D_Boltzmann(Δt,Ter,X,V,g₀,h₀,τ)
    # T, X, V: discretised point sets for t, x₁, v₁
    # assuming constant spacing
    Δx = X[2]-X[1]
    M, N = length(X), length(V)
    g = zeros(Ter,M,N)
    h = zeros(Ter,M,N)
    αβγ = zeros(M,3)
    μ = zeros(Ter,M,4)
    g[1,:,:] = g₀.(X,V')
    h[1,:,:] = h₀.(X,V')
    for i = 1:Ter-1
        # Find α, β, γ
        for j = 1:M
            μ[i,j,:] = moments(V,g[i,j,:],h[i,j,:],N)
            if min(μ[1],μ[3],μ[4]) < 0
                print("Negative moment: $([i j])")
                return g[1:i,:,:],h[1:i,:,:],μ[1:i,:,:]
            end
            a = (μ[i,j,1]*(μ[i,j,3]+μ[i,j,4])-μ[i,j,2])/(3*μ[i,j,1]^2)
            αβγ[j,3] = -1/(2*a)
            L,J = LJ(αβγ[j,:],V,μ)
            times = 0
            while norm(L) > 1e-7
                L,J = LJ(αβγ[j,:],V,μ)
                if times >= 1000
                    error("Divergent")
                end
                Δ = inv(J)*L
                αβγ[j,1:2] .-= Δ
                times += 1
            end
        end

        # Apply Euler's Method
        for k = 1:N
            A = zeros(M)
            for j = 1:M
                A[j] = Δt / τ(μ[i,j,:])
            end
            B = exp.(αβγ*[1,V[k],V[k]^2])
            Bₒ = B ./ (-αβγ[:,3])
            P = Tridiagonal(
                fill((V[k]+abs(V[k]))*Δt/(2*Δx),M-1),
                (1-Δt/Δx*abs(V[k])).- A,
                fill(-(V[k]-abs(V[k]))*Δt/(2*Δx),M-1)
            )
            P = sparse(P)
            g[i+1,:,k] = P*g[i,:,k] + A.*B
            h[i+1,:,k] = P*g[i,:,k] + A.*Bₒ

            g[i+1,1,k] = 2*g[i+1,2,k] - g[i+1,3,k]
            g[i+1,end,k] = 2*g[i+1,end-1,k] - g[i+1,end-2,k]
            h[i+1,1,k] = 2*h[i+1,2,k] - h[i+1,3,k]
            h[i+1,end,k] = 2*h[i+1,end-1,k] - h[i+1,end-2,k]

            if min(g[i+1,:,k]...) < -0.05 || min(h[i+1,:,k]...) < -0.05
                print("Negative Density: $([i+1 k])")
                return g[1:i+1,:,:],h[1:i+1,:,:],μ[1:i+1,:,:]
            end
        end
    end

    for j = 1:M
        μ[Ter,j,:] = moments(V,g[Ter,j,:],h[Ter,j,:],N)
    end

    return g,h,μ
end

function τ(μ)
    ρ = μ[1]
    θ = (μ[1]*(μ[3]+μ[4])-μ[2]^2)/(3*μ[1]^2)
    return (sqrt(π/2)*15/19.7936)/(ρ*θ^(0.28))
end

function Q3(Mₛ,Δt,Ter,a,b,M,N)
    ρ₁, ρ₂ = 1, 4*Mₛ^2/(Mₛ^2+3)
    u₁, u₂ = sqrt(5/3)*Mₛ, sqrt(5/3)*(Mₛ^2+3)/(4*Mₛ)
    θ₁, θ₂ = 1, (5*Mₛ^2-1)/(4*ρ₂)
    c = min(u₁-3.5*θ₁,u₂-3.5*θ₂)
    d = max(u₁+3.5*θ₁,u₂+3.5*θ₂)

    function g(x,v)
        if x <= 0
            return ρ₁/sqrt(2*π*θ₁)*exp(-(v-u₁)^2/(2*θ₁))
        else
            return ρ₂/sqrt(2*π*θ₂)*exp(-(v-u₂)^2/(2*θ₂))
        end
    end

    function h(x,v)
        if x <= 0
            return ρ₁*sqrt(2*θ₁/π)*exp(-(v-u₁)^2/(2*θ₁))
        else
            return ρ₂*sqrt(2*θ₂/π)*exp(-(v-u₂)^2/(2*θ₂))
        end
    end

    X = LinRange(a,b,M+2)
    V = LinRange(c,d,N+2)

    g,h,μ = one_D_Boltzmann(Δt,Ter,X,V,g,h,τ)

    if length(g[:,1,1]) < Ter
        return g,h,μ
    else
        y = (μ[end,:,1].-μ[end,1,1])./(μ[end,end,1].-μ[end,1,1])
        anim = @animate for i=1:Ter
            plot(X,(μ[i,:,1].-μ[i,1,1])./(μ[i,end,1].-μ[i,1,1]),
            title = "Mₛ=$(Mₛ)", xlabel = "x₁",
            ylabel = "Scaled Density",
            label = "t = $(round(i*Δt,digits=4))")
        end
        gif(anim, "Density Ms=$(Mₛ).gif", fps = 20)
        return g,h,μ
    end
end
