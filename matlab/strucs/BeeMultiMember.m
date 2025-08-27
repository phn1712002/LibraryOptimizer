classdef BeeMultiMember < MultiObjectiveMember
    %{
    BeeMultiMember - Represents a bee individual for multi-objective Artificial Bee Colony algorithm
    
    Attributes:
        position : array
            Current position in the search space
        multi_fitness : array
            Multi-objective fitness values
        trial : int
            Counter for unsuccessful attempts, used for abandonment strategy
        dominated : bool
            Whether this solution is dominated by others
        grid_index : int
            Grid index for archive management
        grid_sub_index : array
            Sub-grid indices for each objective
    %}
    
    properties
        trial
    end
    
    methods
        function obj = BeeMultiMember(position, fitness, trial)
            %{
            BeeMultiMember constructor - Initialize a BeeMultiMember with position, fitness, and trial counter
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : array
                    Multi-objective fitness values
                trial : int
                    Trial counter for abandonment (default: 0)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set default values if not provided
            if nargin < 3
                obj.trial = 0;
            else
                obj.trial = trial;
            end
        end
        
        function new_bee = copy(obj)
            %{
            copy - Create a deep copy of the BeeMultiMember
            
            Returns:
                new_bee : BeeMultiMember
                    A new BeeMultiMember object with copied properties
            %}
            
            new_bee = BeeMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.trial ...
            );
            
            % Copy additional properties from parent
            new_bee.dominated = obj.dominated;
            new_bee.grid_index = obj.grid_index;
            if ~isempty(obj.grid_sub_index)
                new_bee.grid_sub_index = obj.grid_sub_index;
            end
        end
        
        function str = char(obj)
            %{
            char - String representation of the BeeMultiMember
            
            Returns:
                str : string
                    Formatted string showing position, fitness, and trial counter
            %}
            
            str = sprintf('Position: %s - Fitness: [%s] - Trial: %d - Dominated: %d', ...
                mat2str(obj.position), ...
                strjoin(string(num2str(obj.multi_fitness(:), '%.6g')), ' '), ...
                obj.trial, obj.dominated);
        end
    end
end
