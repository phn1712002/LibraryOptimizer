classdef BacteriaMultiMember < MultiObjectiveMember
    %{
    BacteriaMultiMember - Multi-objective bacteria member with health attribute
    
    Extends MultiObjectiveMember with BFO-specific attributes:
    - health: Health value for reproduction selection (sum of fitness values across chemotaxis loops)
    %}
    
    properties
        health
    end
    
    methods
        function obj = BacteriaMultiMember(position, fitness, health)
            %{
            BacteriaMultiMember constructor
            
            Inputs:
                position : array
                    Current position vector
                fitness : array
                    Current fitness values (multiple objectives)
                health : float, optional
                    Health value for reproduction (default: 0.0)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set health value
            if nargin < 3 || isempty(health)
                obj.health = 0.0;
            else
                obj.health = health;
            end
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the member
            
            Returns:
                new_member : BacteriaMultiMember
                    Deep copy of this member
            %}
            
            new_member = BacteriaMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.health ...
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
                    Formatted string showing position, fitness, and health
            %}
            
            str = sprintf('Position: %s - Fitness: %s - Health: %.4f', ...
                mat2str(obj.position), ...
                mat2str(obj.multi_fitness), ...
                obj.health);
        end
    end
end
