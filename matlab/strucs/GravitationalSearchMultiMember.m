classdef GravitationalSearchMultiMember < MultiObjectiveMember
    %{
    GravitationalSearchMultiMember - Multi-objective gravitational search member with velocity
    
    Extends MultiObjectiveMember with GSA-specific attributes:
    - velocity: Current velocity vector for gravitational movement
    %}
    
    properties
        velocity
    end
    
    methods
        function obj = GravitationalSearchMultiMember(position, fitness, velocity)
            %{
            GravitationalSearchMultiMember constructor
            
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
            
            % Set velocity
            if nargin < 3 || isempty(velocity)
                obj.velocity = zeros(size(position));
            else
                obj.velocity = velocity;
            end
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the member
            
            Returns:
                new_member : GravitationalSearchMultiMember
                    Deep copy of this member
            %}
            
            new_member = GravitationalSearchMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.velocity ...
            );
            
            % Copy additional properties
            new_member.dominated = obj.dominated;
            new_member.grid_index = obj.grid_index;
            new_member.grid_sub_index = obj.grid_sub_index;
        end
        
        function str = char(obj)
            %{
            char - String representation of the member
            
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
