classdef GlowwormMultiMember < MultiObjectiveMember
    %{
    GlowwormMultiMember - Multi-objective glowworm member with luciferin and decision range
    
    Extends MultiObjectiveMember with GSO-specific attributes:
    - luciferin: Luciferin value for glowworm communication
    - decision_range: Decision range for neighbor detection
    %}
    
    properties
        luciferin
        decision_range
    end
    
    methods
        function obj = GlowwormMultiMember(position, fitness, luciferin, decision_range)
            %{
            GlowwormMultiMember constructor
            
            Inputs:
                position : array
                    Current position vector
                fitness : array
                    Current fitness values (multiple objectives)
                luciferin : float, optional
                    Luciferin value (default: 5.0)
                decision_range : float, optional
                    Decision range value (default: 3.0)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set luciferin and decision range
            if nargin < 3 || isempty(luciferin)
                obj.luciferin = 5.0;
            else
                obj.luciferin = luciferin;
            end
            
            if nargin < 4 || isempty(decision_range)
                obj.decision_range = 3.0;
            else
                obj.decision_range = decision_range;
            end
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the member
            
            Returns:
                new_member : GlowwormMultiMember
                    Deep copy of this member
            %}
            
            new_member = GlowwormMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.luciferin, ...
                obj.decision_range ...
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
                    Formatted string showing position, fitness, luciferin, and decision range
            %}
            
            str = sprintf('Position: %s - Fitness: %s - Luciferin: %.3f - Decision Range: %.3f', ...
                mat2str(obj.position), ...
                mat2str(obj.multi_fitness), ...
                obj.luciferin, ...
                obj.decision_range);
        end
    end
end
