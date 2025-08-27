classdef ParticleMultiMember < MultiObjectiveMember
    %{
    ParticleMultiMember - Multi-objective particle with velocity and personal best
    
    Extends MultiObjectiveMember with PSO-specific attributes:
    - velocity: Current velocity vector
    - personal_best_position: Best position found by this particle
    - personal_best_fitness: Best fitness found by this particle
    %}
    
    properties
        velocity
        personal_best_position
        personal_best_fitness
    end
    
    methods
        function obj = ParticleMultiMember(position, fitness, velocity)
            %{
            ParticleMultiMember constructor
            
            Inputs:
                position : array
                    Current position vector
                fitness : array
                    Current fitness values (multiple objectives)
                velocity : array, optional
                    Current velocity vector (default: zeros)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set velocity and personal best
            if nargin < 3 || isempty(velocity)
                obj.velocity = zeros(size(position));
            else
                obj.velocity = velocity;
            end
            
            obj.personal_best_position = position;
            obj.personal_best_fitness = fitness;
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the particle
            
            Returns:
                new_member : ParticleMultiMember
                    Deep copy of this particle
            %}
            
            new_member = ParticleMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.velocity ...
            );
            
            % Copy additional properties
            new_member.dominated = obj.dominated;
            new_member.grid_index = obj.grid_index;
            new_member.grid_sub_index = obj.grid_sub_index;
            new_member.personal_best_position = obj.personal_best_position;
            new_member.personal_best_fitness = obj.personal_best_fitness;
        end
        
        function str = char(obj)
            %{
            char - String representation of the particle
            
            Returns:
                str : char
                    Formatted string showing position, fitness, and velocity
            %}
            
            str = sprintf('Position: %s - Fitness: %s - Velocity: %s', ...
                mat2str(obj.position), ...
                mat2str(obj.multi_fitness), ...
                mat2str(obj.velocity));
        end
    end
end
