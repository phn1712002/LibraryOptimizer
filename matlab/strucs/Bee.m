classdef Bee < Member
    %{
    Bee - Represents a bee individual in the Artificial Bee Colony algorithm
    
    Attributes:
        position : array
            Current position in the search space
        fitness : float
            Fitness value of the current position
        trial : int
            Counter for unsuccessful attempts, used for abandonment strategy
    %}
    
    properties
        trial
    end
    
    methods
        function obj = Bee(position, fitness, trial)
            %{
            Bee constructor - Initialize a Bee with position, fitness, and trial counter
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : float
                    Fitness value of the position
                trial : int
                    Initial trial counter (default: 0)
            %}
            
            % Call parent constructor
            obj@Member(position, fitness);
            
            % Set trial counter
            if nargin < 3
                obj.trial = 0;
            else
                obj.trial = trial;
            end
        end
        
        function new_bee = copy(obj)
            %{
            copy - Create a deep copy of the Bee
            
            Returns:
                new_bee : Bee
                    A new Bee object with copied position, fitness, and trial counter
            %}
            
            new_bee = Bee(obj.position, obj.fitness, obj.trial);
        end
        
        function str = to_string(obj)
            %{
            to_string - String representation of the Bee
            
            Returns:
                str : string
                    Formatted string showing position, fitness, and trial counter
            %}
            
            str = sprintf('Position: %s - Fitness: %.6f - Trial: %d', ...
                mat2str(obj.position), obj.fitness, obj.trial);
        end
    end
end
