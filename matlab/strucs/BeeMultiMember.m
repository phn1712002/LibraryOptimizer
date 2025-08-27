classdef BeeMultiMember < MultiObjectiveMember
    %{
    BeeMultiMember - Multi-objective bee member with trial counter
    
    Extends MultiObjectiveMember with ABC-specific attributes:
    - trial: Trial counter for scout bee mechanism
    %}
    
    properties
        trial
    end
    
    methods
        function obj = BeeMultiMember(position, fitness, trial)
            %{
            BeeMultiMember constructor
            
            Inputs:
                position : array
                    Current position vector
                fitness : array
                    Current fitness values (multiple objectives)
                trial : int, optional
                    Trial counter (default: 0)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set trial counter
            if nargin < 3 || isempty(trial)
                obj.trial = 0;
            else
                obj.trial = trial;
            end
        end
        
        function new_member = copy(obj)
            %{
            copy - Create a deep copy of the member
            
            Returns:
                new_member : BeeMultiMember
                    Deep copy of this member
            %}
            
            new_member = BeeMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.trial ...
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
                    Formatted string showing position, fitness, and trial counter
            %}
            
            str = sprintf('Position: %s - Fitness: %s - Trial: %d', ...
                mat2str(obj.position), ...
                mat2str(obj.multi_fitness), ...
                obj.trial);
        end
    end
end
