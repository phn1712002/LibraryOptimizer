classdef Particle < Member
    %{
    Particle - Represents a particle in Particle Swarm Optimization
    
    Extends the base Member class with velocity information for PSO.
    
    Attributes:
        position : array
            Current position in the search space
        fitness : float
            Fitness value of the current position
        velocity : array
            Velocity vector of the particle
    %}
    
    properties
        velocity
    end
    
    methods
        function obj = Particle(position, fitness, velocity)
            %{
            Particle constructor - Initialize a Particle with position, fitness, and velocity
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : float
                    Fitness value of the position
                velocity : array, optional
                    Velocity vector of the particle (default: zeros)
            %}
            
            % Call parent constructor
            obj@Member(position, fitness);
            
            % Set velocity
            if nargin < 3
                obj.velocity = zeros(size(position));
            else
                obj.velocity = velocity;
            end
        end
        
        function new_particle = copy(obj)
            %{
            copy - Create a deep copy of the Particle
            
            Returns:
                new_particle : Particle
                    A new Particle object with copied position, fitness, and velocity
            %}
            
            new_particle = Particle(obj.position, obj.fitness, obj.velocity);
        end
        
        function str = char(obj)
            %{
            char - String representation of the Particle
            
            Returns:
                str : string
                    Formatted string showing position, fitness, and velocity
            %}
            
            str = sprintf('Position: %s - Fitness: %.6f - Velocity: %s', ...
                mat2str(obj.position), obj.fitness, mat2str(obj.velocity));
        end
    end
end
