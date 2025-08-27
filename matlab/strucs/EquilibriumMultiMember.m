classdef EquilibriumMultiMember < MultiObjectiveMember
    %{
    EquilibriumMultiMember - Multi-objective equilibrium member
    
    Extends MultiObjectiveMember for Equilibrium Optimizer algorithm.
    This is a simple wrapper class that maintains the same interface
    as MultiObjectiveMember but provides a specific copy method.
    %}
    
    methods
        function obj = EquilibriumMultiMember(position, fitness)
            %{
            EquilibriumMultiMember constructor
            
            Inputs:
                position : array
                    Current position vector
                fitness : array
                    Current fitness values (multiple objectives)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the member
            
            Returns:
                new_member : EquilibriumMultiMember
                    Deep copy of this member
            %}
            
            new_member = EquilibriumMultiMember(...
                obj.position, ...
                obj.multi_fitness ...
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
                    Formatted string showing position and fitness
            %}
            
            str = sprintf('Position: %s - Fitness: %s', ...
                mat2str(obj.position), ...
                mat2str(obj.multi_fitness));
        end
    end
end
