classdef BatMultiMember < MultiObjectiveMember
    %{
    BatMultiMember - Represents a bat individual for multi-objective Bat Algorithm
    
    Attributes:
        position : array
            Current position in the search space
        multi_fitness : array
            Multi-objective fitness values
        frequency : float
            Frequency used for velocity update
        velocity : array
            Velocity vector of the bat
        loudness : float
            Loudness parameter (controls acceptance of new solutions)
        pulse_rate : float
            Pulse emission rate (controls random walk probability)
        dominated : bool
            Whether this solution is dominated by others
        grid_index : int
            Grid index for archive management
        grid_sub_index : array
            Sub-grid indices for each objective
    %}
    
    properties
        frequency
        velocity
        loudness
        pulse_rate
    end
    
    methods
        function obj = BatMultiMember(position, fitness, frequency, velocity, loudness, pulse_rate)
            %{
            BatMultiMember constructor - Initialize a BatMultiMember with position, fitness, and bat-specific parameters
            
            Inputs:
                position : array
                    Position vector in search space
                fitness : array
                    Multi-objective fitness values
                frequency : float
                    Frequency parameter (default: 0.0)
                velocity : array
                    Velocity vector (default: zeros)
                loudness : float
                    Loudness parameter (default: 1.0)
                pulse_rate : float
                    Pulse rate parameter (default: 0.5)
            %}
            
            % Call parent constructor
            obj@MultiObjectiveMember(position, fitness);
            
            % Set default values if not provided
            if nargin < 3
                obj.frequency = 0.0;
            else
                obj.frequency = frequency;
            end
            
            if nargin < 4
                obj.velocity = zeros(size(position));
            else
                obj.velocity = velocity;
            end
            
            if nargin < 5
                obj.loudness = 1.0;
            else
                obj.loudness = loudness;
            end
            
            if nargin < 6
                obj.pulse_rate = 0.5;
            else
                obj.pulse_rate = pulse_rate;
            end
        end
        
        function new_bat = copy(obj)
            %{
            copy - Create a deep copy of the BatMultiMember
            
            Returns:
                new_bat : BatMultiMember
                    A new BatMultiMember object with copied properties
            %}
            
            new_bat = BatMultiMember(...
                obj.position, ...
                obj.multi_fitness, ...
                obj.frequency, ...
                obj.velocity, ...
                obj.loudness, ...
                obj.pulse_rate ...
            );
            
            % Copy additional properties from parent
            new_bat.dominated = obj.dominated;
            new_bat.grid_index = obj.grid_index;
            if ~isempty(obj.grid_sub_index)
                new_bat.grid_sub_index = obj.grid_sub_index;
            end
        end
        
        function str = char(obj)
            %{
            char - String representation of the BatMultiMember
            
            Returns:
                str : string
                    Formatted string showing position, fitness, and bat parameters
            %}
            
            str = sprintf('Position: %s - Fitness: [%s] - Frequency: %.3f - Loudness: %.3f - Pulse Rate: %.3f - Dominated: %d', ...
                mat2str(obj.position), ...
                strjoin(string(num2str(obj.multi_fitness(:), '%.6g')), ' '), ...
                obj.frequency, obj.loudness, obj.pulse_rate, obj.dominated);
        end
    end
end
